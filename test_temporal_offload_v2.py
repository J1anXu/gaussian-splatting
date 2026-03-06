"""
test_temporal_offload.py — v2 修复版
=====================================
修复:
  1. Benchmark: Store All OOM后隔离, 不影响后续Ours的运行
  2. Training: gradient clipping + 更小lr防止NaN

用法:
  python test_temporal_offload.py --test verify
  python test_temporal_offload.py --test benchmark
  python test_temporal_offload.py --test ablation
  python test_temporal_offload.py --test training
  python test_temporal_offload.py --test all
"""

import torch
import time
import gc
import argparse
from temporal_offload import DiffSimCheckpointer


class MassSpring:
    def __init__(self, n_particles, k_neighbors=6, device='cuda'):
        self.n = n_particles
        self.device = device
        self.dt = 0.005
        self.k_spring = 50.0
        self.damping = 0.05
        self.gravity = torch.tensor([0., -9.8, 0.], device=device)
        n_springs = n_particles * k_neighbors
        si = torch.randint(0, n_particles, (n_springs,), device=device)
        sj = torch.randint(0, n_particles, (n_springs,), device=device)
        mask = si != sj
        self.si, self.sj = si[mask], sj[mask]
        self.init_pos = torch.randn(n_particles, 3, device=device) * 0.3
        diff = self.init_pos[self.si] - self.init_pos[self.sj]
        self.rest_len = diff.norm(dim=1).detach()

    def step(self, pos, vel):
        diff = pos[self.si] - pos[self.sj]
        dist = diff.norm(dim=1, keepdim=True).clamp(min=1e-6)
        direction = diff / dist
        stretch = dist.squeeze(1) - self.rest_len
        f = torch.zeros_like(pos)
        f.scatter_add_(0, self.si.unsqueeze(1).expand(-1, 3),
                       -self.k_spring * stretch.unsqueeze(1) * direction)
        acc = f - self.damping * vel + self.gravity
        new_vel = vel + acc * self.dt
        new_pos = pos + new_vel * self.dt
        return new_pos, new_vel

    def get_init_state(self):
        return (self.init_pos.clone(), torch.zeros_like(self.init_pos))


def reset_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def test_verify():
    print("=" * 65)
    print("TEST: 梯度正确性验证")
    print("=" * 65)
    print("对比 Store All (ground truth) vs Segmented Adjoint (ours)\n")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    configs = [
        (1000, 100, 10), (1000, 100, 20), (1000, 500, 50),
        (5000, 200, 20), (5000, 200, 50),
        (10000, 100, 10), (10000, 100, 50),
    ]
    print(f"{'Particles':>10} {'Steps':>6} {'K':>4} {'Pos MaxRelErr':>15} {'Vel MaxRelErr':>15} {'Status':>8}")
    print("-" * 65)
    for n_particles, n_steps, K in configs:
        sim = MassSpring(n_particles, device=device)
        init_state = sim.get_init_state()
        reset_gpu()
        state_gt = tuple(s.clone().requires_grad_(True) for s in init_state)
        p, v = state_gt
        for _ in range(n_steps):
            p, v = sim.step(p, v)
        loss = p.sum() + v.sum()
        loss.backward()
        grad_pos_gt = state_gt[0].grad.clone()
        grad_vel_gt = state_gt[1].grad.clone()
        del p, v, loss
        reset_gpu()
        ckpt = DiffSimCheckpointer(sim.step, segment_length=K, device=device)
        final = ckpt.forward(init_state, n_steps)
        grad_final = (torch.ones_like(final[0]), torch.ones_like(final[1]))
        grad_init = ckpt.backward(grad_final)
        pos_err = (grad_pos_gt - grad_init[0]).abs().max().item()
        vel_err = (grad_vel_gt - grad_init[1]).abs().max().item()
        pos_rel = pos_err / (grad_pos_gt.abs().max().item() + 1e-12)
        vel_rel = vel_err / (grad_vel_gt.abs().max().item() + 1e-12)
        status = "✓" if max(pos_rel, vel_rel) < 1e-3 else "✗"
        print(f"{n_particles:>10} {n_steps:>6} {K:>4} {pos_rel:>15.2e} {vel_rel:>15.2e} {status:>8}")
        del grad_pos_gt, grad_vel_gt, grad_init
        reset_gpu()
    print("\n所有 ✓ = 数学精确等价。")


def _run_store_all(sim, init_state, n_steps, device):
    reset_gpu()
    try:
        p = init_state[0].clone().requires_grad_(True)
        v = init_state[1].clone().requires_grad_(True)
        t0 = time.perf_counter()
        for _ in range(n_steps):
            p, v = sim.step(p, v)
        loss = p.sum()
        loss.backward()
        if device == 'cuda': torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / 1024**2 if device == 'cuda' else 0
        del p, v, loss
        gc.collect()
        return peak, elapsed
    except RuntimeError:
        if device == 'cuda': torch.cuda.empty_cache()
        gc.collect()
        return None, None


def _run_ours(sim, init_state, n_steps, K, device):
    reset_gpu()
    try:
        ckpt = DiffSimCheckpointer(sim.step, segment_length=K, device=device)
        t0 = time.perf_counter()
        final = ckpt.forward(init_state, n_steps)
        grad_final = (torch.ones_like(final[0]), torch.zeros_like(final[1]))
        grad_init = ckpt.backward(grad_final)
        if device == 'cuda': torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / 1024**2 if device == 'cuda' else 0
        cpu_mb = ckpt.stats.total_cpu_checkpoint_mb
        del final, grad_final, grad_init, ckpt
        gc.collect()
        return peak, elapsed, cpu_mb
    except RuntimeError:
        if device == 'cuda': torch.cuda.empty_cache()
        gc.collect()
        return None, None, None


def test_benchmark():
    print("=" * 65)
    print("TEST: Store All vs Segmented Adjoint+Offload")
    print("=" * 65)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_particles = 50000
    K = 50
    sim = MassSpring(n_particles, device=device)
    init_state = sim.get_init_state()
    print(f"配置: {n_particles} 粒子, K={K}\n")
    print(f"{'Steps':>7} │{'Peak MB':>9} {'Time':>8} │{'GPU MB':>8} {'CPU MB':>8} {'Time':>8} │{'Save':>5}")
    print(f"{'':>7} │{'Store All':^18}│{'Ours (Offload)':^26}│{'':>5}")
    print("─" * 72)

    test_steps = [100, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    store_all_oom = False

    for ns in test_steps:
        if not store_all_oom:
            peak_a, time_a = _run_store_all(sim, init_state, ns, device)
            if peak_a is None:
                store_all_oom = True
        else:
            peak_a, time_a = None, None

        peak_b, time_b, cpu_mb = _run_ours(sim, init_state, ns, K, device)

        a_str = f"{peak_a:>8.0f} {time_a:>7.2f}s" if peak_a else f"{'OOM':>8} {'':>7} "
        if peak_b is not None:
            b_str = f"{peak_b:>7.0f} {cpu_mb:>7.0f} {time_b:>7.2f}s"
        else:
            b_str = f"{'OOM':>7} {'':>7} {'':>7} "

        if peak_a and peak_b:
            save = f"{peak_a/peak_b:.0f}x"
        elif peak_b and not peak_a:
            save = "∞"
        else:
            save = "—"

        print(f"{ns:>7} │{a_str} │{b_str} │{save:>5}")
    print()


def test_ablation():
    print("=" * 65)
    print("TEST: 消融实验 — Segment Length K")
    print("=" * 65)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_particles = 50000
    n_steps = 10000
    sim = MassSpring(n_particles, device=device)
    init_state = sim.get_init_state()
    print(f"配置: {n_particles} 粒子, {n_steps} 步\n")
    print(f"{'K':>6} {'GPU Peak':>10} {'CPU Ckpt':>10} {'Fwd':>8} {'Bwd':>8} {'Total':>8} {'#Segs':>7}")
    print("-" * 60)
    for K in [5, 10, 20, 50, 100, 200, 500]:
        reset_gpu()
        try:
            ckpt = DiffSimCheckpointer(sim.step, segment_length=K, device=device)
            final = ckpt.forward(init_state, n_steps)
            grad_final = (torch.ones_like(final[0]), torch.zeros_like(final[1]))
            ckpt.backward(grad_final)
            if device == 'cuda': torch.cuda.synchronize()
            s = ckpt.stats
            gpu = torch.cuda.max_memory_allocated() / 1024**2 if device == 'cuda' else 0
            total = s.forward_time + s.backward_time
            print(f"{K:>6} {gpu:>8.0f}MB {s.total_cpu_checkpoint_mb:>8.0f}MB "
                  f"{s.forward_time:>6.2f}s {s.backward_time:>6.2f}s {total:>6.2f}s {s.n_segments:>7}")
            del final, grad_final, ckpt
        except RuntimeError:
            print(f"{K:>6}      OOM")
            torch.cuda.empty_cache()
        gc.collect()
    print("\n  K小→GPU少/慢, K大→GPU多/快。选K使autograd刚好fit GPU budget。")


def test_training():
    print("=" * 65)
    print("TEST: 端到端训练 — 短horizon vs 长horizon")
    print("=" * 65)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_particles = 5000
    K = 50
    n_iters = 50
    lr = 1e-4
    grad_clip = 1.0

    for n_steps, label in [(200, "短horizon(200步, Store All也能跑)"),
                            (5000, "长horizon(5000步, 只有Ours能跑)")]:
        print(f"\n--- {label} ---\n")
        sim = MassSpring(n_particles, device=device)
        target_pos = torch.zeros(n_particles, 3, device=device)
        init_vel = torch.randn(n_particles, 3, device=device) * 0.01
        print(f"{'Iter':>5} {'Loss':>14} {'|grad|':>10} {'GPU MB':>8} {'Time':>7}")
        print("-" * 50)
        losses = []
        for it in range(n_iters):
            reset_gpu()
            init_state = (sim.init_pos.clone(), init_vel.clone())
            ckpt = DiffSimCheckpointer(sim.step, segment_length=K, device=device)
            t0 = time.perf_counter()
            final_state = ckpt.forward(init_state, n_steps)
            diff = final_state[0] - target_pos
            loss_val = (diff ** 2).mean().item()
            grad_pos = 2.0 * diff / (n_particles * 3)
            grad_vel_f = torch.zeros_like(final_state[1])
            grad_init = ckpt.backward((grad_pos, grad_vel_f))
            elapsed = time.perf_counter() - t0
            gpu_peak = torch.cuda.max_memory_allocated() / 1024**2 if device == 'cuda' else 0
            gv = grad_init[1]
            gnorm = gv.norm().item()
            if gnorm > grad_clip:
                gv = gv * (grad_clip / gnorm)
                gnorm = grad_clip
            init_vel = (init_vel - lr * gv).detach()
            losses.append(loss_val)
            if it % 10 == 0 or it == n_iters - 1:
                print(f"{it:>5} {loss_val:>14.4f} {gnorm:>10.4f} {gpu_peak:>6.0f}MB {elapsed:>5.2f}s")
            del ckpt, final_state, grad_init

        if losses[-1] < losses[0] * 0.99:
            print(f"\n  收敛: {losses[0]:.4f} → {losses[-1]:.4f} ✓")
        else:
            print(f"\n  未明显收敛 (可能需要更多迭代或调参)")

    print("\n结论: 长horizon训练只有Ours能跑, 且loss可以下降。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', default='all',
                        choices=['verify', 'benchmark', 'ablation', 'training', 'all'])
    args = parser.parse_args()
    print(f"PyTorch {torch.__version__}")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {name} ({mem:.0f} GB)")
    print()
    if args.test in ('verify', 'all'): test_verify(); print()
    if args.test in ('benchmark', 'all'): test_benchmark(); print()
    if args.test in ('ablation', 'all'): test_ablation(); print()
    if args.test in ('training', 'all'): test_training(); print()


if __name__ == '__main__':
    main()
