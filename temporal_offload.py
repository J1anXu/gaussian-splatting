"""
temporal_offload.py — Temporal Memory Offloading for Differentiable Simulation
================================================================================

把长时间可微物理模拟的内存从 O(T) 降到 O(K):
  - Forward阶段: 不建autograd, 每K步把checkpoint存到CPU
  - Backward阶段: 逐段从CPU load checkpoint, 重建K步autograd, 求梯度, 释放

用法:

    from temporal_offload import DiffSimCheckpointer

    sim = YourPhysicsSimulator()
    ckpt = DiffSimCheckpointer(sim_step_fn=sim.step, segment_length=50)

    # Forward
    state = (pos, vel)
    final_state = ckpt.forward(state, n_steps=10000)

    # 定义loss
    loss = final_state[0].sum()

    # Backward (自动offload + segmented adjoint)
    grads = ckpt.backward(loss_grad={'pos': torch.ones_like(final_state[0]),
                                      'vel': torch.zeros_like(final_state[1])})

    # grads 是对初始state的梯度

兼容任何 PyTorch-based 的可微模拟, 不需要改模拟器代码。
"""

import torch
import time
from typing import Tuple, List, Dict, Callable, Optional, Union
from dataclasses import dataclass, field


@dataclass
class OffloadStats:
    """记录offloading的统计信息"""
    n_steps: int = 0
    n_segments: int = 0
    segment_length: int = 0
    forward_time: float = 0.0
    backward_time: float = 0.0
    peak_gpu_mb: float = 0.0
    total_cpu_checkpoint_mb: float = 0.0
    gpu_to_cpu_transfer_time: float = 0.0
    cpu_to_gpu_transfer_time: float = 0.0

    def summary(self) -> str:
        return (
            f"Steps: {self.n_steps}, Segments: {self.n_segments}, K: {self.segment_length}\n"
            f"Forward: {self.forward_time:.3f}s, Backward: {self.backward_time:.3f}s\n"
            f"Peak GPU: {self.peak_gpu_mb:.0f} MB, CPU Checkpoints: {self.total_cpu_checkpoint_mb:.0f} MB\n"
            f"Transfer: GPU→CPU {self.gpu_to_cpu_transfer_time*1000:.1f}ms, "
            f"CPU→GPU {self.cpu_to_gpu_transfer_time*1000:.1f}ms total"
        )


class DiffSimCheckpointer:
    """
    可微物理模拟的时序内存管理器。

    核心思想:
      Forward时不建autograd, 每K步存checkpoint到CPU pinned memory。
      Backward时逐段从CPU load checkpoint, 重跑K步with autograd, 
      backward这一段, 释放autograd, 传递梯度到上一段。

    参数:
      sim_step_fn: 模拟器的单步函数。
                   签名: (state_tuple) -> new_state_tuple
                   其中state_tuple是(tensor1, tensor2, ...)形式。
                   必须是纯函数(给相同输入返回相同输出)。
      segment_length: 每K步存一个checkpoint。K越大GPU内存越多但重算越多。
      device: GPU设备。
      use_pinned_memory: 是否用pinned memory加速CPU<->GPU传输。
      verbose: 是否打印进度。
    """

    def __init__(
        self,
        sim_step_fn: Callable,
        segment_length: int = 50,
        device: str = 'cuda',
        use_pinned_memory: bool = True,
        verbose: bool = False,
    ):
        self.step_fn = sim_step_fn
        self.K = segment_length
        self.device = device
        self.use_pinned = use_pinned_memory
        self.verbose = verbose

        # 内部状态
        self._cpu_checkpoints: List[Tuple[torch.Tensor, ...]] = []
        self._n_steps: int = 0
        self._segment_boundaries: List[int] = []  # 每个checkpoint对应的时间步
        self._stats = OffloadStats()

    # ================================================================
    # Public API
    # ================================================================

    def forward(
        self,
        init_state: Tuple[torch.Tensor, ...],
        n_steps: int,
    ) -> Tuple[torch.Tensor, ...]:
        """
        执行forward模拟, 把checkpoint存到CPU。

        参数:
          init_state: 初始状态, tuple of tensors, 比如 (pos, vel)
          n_steps: 总模拟步数

        返回:
          final_state: 最终状态 (detached, 在GPU上)
        """
        self._n_steps = n_steps
        self._cpu_checkpoints = []
        self._segment_boundaries = []

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        t_start = time.perf_counter()
        t_transfer = 0.0

        # 存初始状态
        t0 = time.perf_counter()
        self._save_checkpoint(init_state, step=0)
        t_transfer += time.perf_counter() - t0

        # Forward: 不建autograd
        state = tuple(s.detach().clone() for s in init_state)
        with torch.no_grad():
            for t in range(n_steps):
                state = self.step_fn(*state)

                # 每K步或最后一步存checkpoint
                if (t + 1) % self.K == 0 or t == n_steps - 1:
                    t0 = time.perf_counter()
                    self._save_checkpoint(state, step=t + 1)
                    t_transfer += time.perf_counter() - t0

                    if self.verbose and (t + 1) % (self.K * 10) == 0:
                        print(f"  Forward: {t+1}/{n_steps} steps, "
                              f"{len(self._cpu_checkpoints)} checkpoints saved")

        self._stats.n_steps = n_steps
        self._stats.n_segments = len(self._cpu_checkpoints) - 1
        self._stats.segment_length = self.K
        self._stats.forward_time = time.perf_counter() - t_start
        self._stats.gpu_to_cpu_transfer_time = t_transfer
        self._stats.total_cpu_checkpoint_mb = self._total_checkpoint_mb()

        if self.verbose:
            print(f"  Forward done: {n_steps} steps, "
                  f"{len(self._cpu_checkpoints)} checkpoints, "
                  f"{self._stats.total_cpu_checkpoint_mb:.0f} MB on CPU")

        return tuple(s.detach() for s in state)

    def backward(
        self,
        grad_final: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, ...]:
        """
        执行segmented adjoint backward, 返回对初始状态的梯度。

        参数:
          grad_final: 对最终状态的梯度, tuple of tensors
                      比如 loss=final_pos.sum() 则 grad_final = (ones_like_pos, zeros_like_vel)

        返回:
          grad_init: 对初始状态的梯度, tuple of tensors
        """
        assert len(self._cpu_checkpoints) >= 2, \
            "Need at least 2 checkpoints (init + final). Did you call forward()?"

        t_start = time.perf_counter()
        t_transfer = 0.0

        n_segments = len(self._cpu_checkpoints) - 1
        grad_state = tuple(g.clone() for g in grad_final)

        for seg in reversed(range(n_segments)):
            seg_start_step = self._segment_boundaries[seg]
            seg_end_step = self._segment_boundaries[seg + 1]
            seg_len = seg_end_step - seg_start_step

            # ① Load checkpoint from CPU
            t0 = time.perf_counter()
            checkpoint = self._load_checkpoint(seg)
            t_transfer += time.perf_counter() - t0

            # ② Re-run segment WITH autograd
            state = tuple(s.requires_grad_(True) for s in checkpoint)
            for _ in range(seg_len):
                state = self.step_fn(*state)

            # ③ Backward this segment
            torch.autograd.backward(
                tensors=list(state),
                grad_tensors=list(grad_state),
            )

            # ④ Extract gradients at segment start, propagate to previous segment
            grad_state = tuple(
                cp.grad.clone() if cp.grad is not None else torch.zeros_like(cp)
                for cp in checkpoint
            )

            # ⑤ Free autograd graph for this segment
            del state, checkpoint

            if self.verbose and (n_segments - seg) % 10 == 0:
                print(f"  Backward: segment {seg}/{n_segments} done")

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self._stats.peak_gpu_mb = torch.cuda.max_memory_allocated(self.device) / 1024**2

        self._stats.backward_time = time.perf_counter() - t_start
        self._stats.cpu_to_gpu_transfer_time = t_transfer

        return grad_state

    def forward_backward(
        self,
        init_state: Tuple[torch.Tensor, ...],
        n_steps: int,
        loss_fn: Callable,
    ) -> Tuple[float, Tuple[torch.Tensor, ...]]:
        """
        一步完成 forward + loss + backward。

        参数:
          init_state: 初始状态
          n_steps: 模拟步数
          loss_fn: 接收final_state, 返回 (loss_value, grad_tuple)
                   其中 grad_tuple 是对 final_state 每个tensor的梯度

        返回:
          (loss_value, grad_init_state)
        """
        final_state = self.forward(init_state, n_steps)
        loss_val, grad_final = loss_fn(final_state)
        grad_init = self.backward(grad_final)
        return loss_val, grad_init

    @property
    def stats(self) -> OffloadStats:
        return self._stats

    # ================================================================
    # Gradient Correctness Verification
    # ================================================================

    @staticmethod
    def verify_gradients(
        sim_step_fn: Callable,
        init_state: Tuple[torch.Tensor, ...],
        n_steps: int,
        segment_length: int = 20,
        device: str = 'cuda',
    ) -> Dict[str, float]:
        """
        验证segmented adjoint的梯度与PyTorch autograd完全一致。

        跑两次:
          1. Store All: 标准PyTorch autograd (ground truth)
          2. Segmented Adjoint: 我们的方法

        返回每个state tensor的梯度最大绝对误差和相对误差。
        """
        print(f"Verifying gradients: {n_steps} steps, K={segment_length}")

        # --- Ground truth: Store All ---
        state_gt = tuple(s.clone().requires_grad_(True) for s in init_state)
        state = state_gt
        for _ in range(n_steps):
            state = sim_step_fn(*state)
        loss_gt = sum(s.sum() for s in state)
        loss_gt.backward()
        grads_gt = tuple(s.grad.clone() for s in state_gt)
        del state
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # --- Our method: Segmented Adjoint ---
        ckpt = DiffSimCheckpointer(
            sim_step_fn=sim_step_fn,
            segment_length=segment_length,
            device=device,
        )
        final_state = ckpt.forward(init_state, n_steps)
        grad_final = tuple(torch.ones_like(s) for s in final_state)
        grads_ours = ckpt.backward(grad_final)

        # --- Compare ---
        results = {}
        for i, (g_gt, g_ours) in enumerate(zip(grads_gt, grads_ours)):
            abs_err = (g_gt - g_ours).abs().max().item()
            rel_err = abs_err / (g_gt.abs().max().item() + 1e-12)
            results[f'state_{i}_max_abs_err'] = abs_err
            results[f'state_{i}_max_rel_err'] = rel_err
            status = "✓ PASS" if rel_err < 1e-4 else "✗ FAIL"
            print(f"  State {i}: max_abs_err={abs_err:.2e}, "
                  f"max_rel_err={rel_err:.2e}  {status}")

        return results

    # ================================================================
    # Internal Methods
    # ================================================================

    def _save_checkpoint(self, state: Tuple[torch.Tensor, ...], step: int):
        """把一个state tuple存到CPU pinned memory"""
        if self.use_pinned:
            cpu_state = tuple(s.detach().cpu().pin_memory() for s in state)
        else:
            cpu_state = tuple(s.detach().cpu() for s in state)
        self._cpu_checkpoints.append(cpu_state)
        self._segment_boundaries.append(step)

    def _load_checkpoint(self, seg_idx: int) -> Tuple[torch.Tensor, ...]:
        """从CPU load一个checkpoint回GPU"""
        cpu_state = self._cpu_checkpoints[seg_idx]
        return tuple(s.to(self.device) for s in cpu_state)

    def _total_checkpoint_mb(self) -> float:
        total = 0
        for ckpt in self._cpu_checkpoints:
            for t in ckpt:
                total += t.nelement() * t.element_size()
        return total / 1024**2
