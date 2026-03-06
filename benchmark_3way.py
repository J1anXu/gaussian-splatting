"""
benchmark_3way.py — 三方对比: Store All vs GPU Checkpoint vs Ours (CPU Offload)
=================================================================================

对比三种策略的GPU peak内存和运行时间:
  A) Store All:       标准PyTorch autograd, 存所有中间变量
  B) GPU Checkpoint:  每K步存checkpoint到GPU, 反向时重算 (现有方案)
  C) CPU Offload:     每K步存checkpoint到CPU, 反向时load回GPU (我们的方案)

python benchmark_3way.py
"""

import torch
import time
import gc
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


def reset_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# ================================================================
# Strategy A: Store All
# ================================================================
def run_store_all(sim, init_state, n_steps, device):
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


# ================================================================
# Strategy B: GPU-only Checkpoint (现有方案)
# ================================================================
def run_gpu_checkpoint(sim, init_state, n_steps, K, device):
    """
    checkpoint存在GPU上, 不offload到CPU。
    Forward: no_grad, 每K步存GPU checkpoint
    Backward: 逐段从GPU checkpoint重建autograd
    """
    reset_gpu()
    try:
        t0 = time.perf_counter()

        # Forward: 存checkpoint到GPU
        pos = init_state[0].clone()
        vel = init_state[1].clone()
        gpu_checkpoints = [(pos.clone(), vel.clone())]  # 全存GPU!
        boundaries = [0]

        with torch.no_grad():
            for t in range(n_steps):
                pos, vel = sim.step(pos, vel)
                if (t + 1) % K == 0 or t == n_steps - 1:
                    gpu_checkpoints.append((pos.clone(), vel.clone()))  # GPU上
                    boundaries.append(t + 1)

        # Backward: 逐段重算
        n_seg = len(gpu_checkpoints) - 1
        grad_pos = torch.ones_like(pos)
        grad_vel = torch.zeros_like(vel)

        for seg in reversed(range(n_seg)):
            seg_len = boundaries[seg + 1] - boundaries[seg]
            cp_pos, cp_vel = gpu_checkpoints[seg]
            p = cp_pos.requires_grad_(True)
            v = cp_vel.requires_grad_(True)
            for _ in range(seg_len):
                p, v = sim.step(p, v)
            torch.autograd.backward([p, v], [grad_pos, grad_vel])
            grad_pos = cp_pos.grad.clone() if cp_pos.grad is not None else torch.zeros_like(cp_pos)
            grad_vel = cp_vel.grad.clone() if cp_vel.grad is not None else torch.zeros_like(cp_vel)
            del p, v

        if device == 'cuda': torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / 1024**2 if device == 'cuda' else 0
        del gpu_checkpoints, grad_pos, grad_vel
        gc.collect()
        return peak, elapsed
    except RuntimeError:
        if device == 'cuda': torch.cuda.empty_cache()
        gc.collect()
        return None, None


# ================================================================
# Strategy C: CPU Offload (我们的方案)
# ================================================================
def run_cpu_offload(sim, init_state, n_steps, K, device):
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


# ================================================================
# Main Benchmark
# ================================================================
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"PyTorch {torch.__version__}")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {name} ({mem:.0f} GB)")

    n_particles = 50000
    K = 50
    sim = MassSpring(n_particles, device=device)
    init_state = (sim.init_pos.clone(), torch.zeros_like(sim.init_pos))
    state_mb = n_particles * 6 * 4 / 1024**2

    print(f"\n配置: {n_particles} 粒子, K={K}, 每步状态={state_mb:.2f} MB")
    print(f"=" * 95)
    print(f"{'':>7} │{'(A) Store All':^18}│{'(B) GPU Checkpoint':^22}│{'(C) Ours CPU Offload':^28}│")
    print(f"{'Steps':>7} │{'GPU MB':>8} {'Time':>7} │{'GPU MB':>8} {'Time':>7}   │{'GPU MB':>8} {'CPU MB':>8} {'Time':>7} │")
    print(f"{'─'*95}")

    test_steps = [100, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    store_all_oom = False

    for ns in test_steps:
        # A) Store All
        if not store_all_oom:
            pa, ta = run_store_all(sim, init_state, ns, device)
            if pa is None: store_all_oom = True
        else:
            pa, ta = None, None

        # B) GPU Checkpoint
        pb, tb = run_gpu_checkpoint(sim, init_state, ns, K, device)

        # C) CPU Offload (ours)
        pc, tc, cpu_mb = run_cpu_offload(sim, init_state, ns, K, device)

        # Format
        a_str = f"{pa:>7.0f} {ta:>6.2f}s" if pa else f"{'OOM':>7} {'':>6} "
        b_str = f"{pb:>7.0f} {tb:>6.2f}s  " if pb else f"{'OOM':>7} {'':>6}   "
        if pc is not None:
            c_str = f"{pc:>7.0f} {cpu_mb:>7.0f} {tc:>6.2f}s"
        else:
            c_str = f"{'OOM':>7} {'':>7} {'':>6} "

        print(f"{ns:>7} │{a_str} │{b_str}│{c_str} │")

    print(f"{'─'*95}")

    print("""
解读:
  (A) Store All:       PyTorch autograd存所有中间变量 → GPU = O(T × 10x), 几千步OOM
  (B) GPU Checkpoint:  checkpoint全存GPU → GPU = O(T/K × state) + O(K × 10x), 内存随T增长
  (C) Ours:            checkpoint存CPU → GPU = O(K × 10x) 恒定, 不随T增长

  关键对比:
    B vs C: 同样的checkpoint策略, 唯一区别是checkpoint存GPU还是CPU
    B的GPU内存随T增长 (checkpoint占GPU), C的GPU内存恒定 (checkpoint在CPU)
    当T足够大, B也会OOM, 但C永远不会 (受限于CPU内存, 通常64-256GB)
""")


if __name__ == '__main__':
    main()
