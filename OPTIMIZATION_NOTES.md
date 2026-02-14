# Gaussian Splatting 训练优化笔记

## 问题背景

当前训练代码（`train.py`）采用CPU-GPU混合策略来严格限制GPU内存占用：
- Phase 1: 正常训练直到达到 SPLIT_SIZE
- Phase 2: 将模型分割成多个 submodel，每个 submodel 存储在 CPU 上，只在需要时移动到 GPU

**主要问题**：训练速度非常慢

## 性能瓶颈分析

### Phase 2 的主要瓶颈

1. **频繁的 CPU-GPU 数据传输**
   - `move_and_activate_subset()` 每次都要从 CPU 移动数据到 GPU
   - 梯度计算后又要移回 CPU (`sub_viewspace_point_tensor.grad.cpu()`)
   - visibility_filter 和 radii 也要移回 CPU

2. **重复渲染**
   - 第一次：无梯度渲染所有可见 submodel（用于 merge，第 283-311 行）
   - 第二次：对每个可见 submodel 再渲染一次（带梯度，用于反向传播，第 325-368 行）
   - 每个 submodel 在一个 iteration 中被渲染 2 次

3. **串行处理**
   - 所有 submodel 依次处理，无法并行
   - 每个 submodel 单独做 forward、backward、optimizer step

4. **Frustum Culling 开销**
   - 每个 iteration 都要对所有 submodel 做 frustum culling
   - CPU 上的 tensor 操作增加额外开销

## 优化方案

### 1. Gradient Checkpointing（梯度检查点）

**原理**：不保存所有中间激活值，在 backward 时重新计算需要的部分

**实现**：
```python
from torch.utils.checkpoint import checkpoint

# 在 render 时使用
render_pkg = checkpoint(render, viewpoint_cam, submodel, pipe, bg, ...)
```

**优点**：
- 减少内存占用，无需把所有数据放 CPU
- PyTorch 自动处理

**缺点**：
- 会增加一些计算时间（重新计算）

---

### 2. 批量处理 Submodel（推荐优先实施）

**原理**：一次在 GPU 上加载多个 submodel，批量处理

**实现**：
```python
BATCH_SIZE = 3  # 根据 GPU 内存调整

for i in range(0, len(visible_submodel_id_list), BATCH_SIZE):
    batch_submodels = visible_submodel_id_list[i:i+BATCH_SIZE]

    # 批量移动到 GPU
    for submodel_id in batch_submodels:
        submodel_list[submodel_id].move_and_activate_subset()

    # 批量渲染
    for submodel_id in batch_submodels:
        # render and backward
        ...

    # 批量 optimizer step
    for submodel_id in batch_submodels:
        submodel_list[submodel_id].optimizer.step()
```

**预期加速**：3-5x

---

### 3. 异步数据传输（Pinned Memory + CUDA Streams）

**原理**：使用 CUDA streams 实现计算和数据传输的 overlap

**实现**：
```python
# 初始化时使用 pinned memory
for submodel in submodel_list:
    submodel._xyz = submodel._xyz.pin_memory()
    submodel._opacity = submodel._opacity.pin_memory()
    # ... 其他参数

# 训练时使用 stream
stream = torch.cuda.Stream()
with torch.cuda.stream(stream):
    submodel.move_and_activate_subset()
```

**优点**：
- 数据传输和计算可以并行
- 减少等待时间

---

### 4. 减少重复渲染

**原理**：第一次无梯度渲染的结果可以部分复用

**实现**：
```python
# 第一次渲染时保存中间状态
with torch.no_grad():
    render_pkg = render(...)
    image_no_grad = render_pkg["render"].detach()

# 第二次使用 saved tensors 避免重复计算
```

---

### 5. Lazy Frustum Culling（推荐优先实施）

**原理**：不是每个 iteration 都做 frustum culling，相机视角变化不大时可以复用

**实现**：
```python
# 每 5 个 iteration 更新一次
if iteration % 5 == 0:
    for model in submodel_list:
        visible_mask = frustum_culling(model._xyz, viewpoint_cam.full_proj_transform)
        model.visible_indices = torch.nonzero(visible_mask, as_tuple=True)[0]
```

**预期加速**：1.5-2x

---

### 6. 混合精度训练（AMP）

**原理**：使用 float16 进行计算，减少内存和计算时间

**实现**：
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    render_pkg = render(...)
    loss = compute_loss(...)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**优点**：
- 减少内存占用 ~50%
- 加速计算 ~2x（在支持 Tensor Core 的 GPU 上）

---

### 7. 提高贡献度阈值

**当前实现**：
```python
if contributed_percent < 0.05:
    continue
```

**优化**：
```python
if contributed_percent < 0.1:  # 提高到 10%
    continue
```

**效果**：跳过更多不重要的 submodel

---

### 8. Gradient Accumulation

**原理**：多个 iteration 累积梯度后再更新，减少 optimizer step 次数

**实现**：
```python
ACCUMULATION_STEPS = 4

for i, submodel in enumerate(visible_submodels):
    loss = compute_loss(...)
    loss = loss / ACCUMULATION_STEPS
    loss.backward()

    if (i + 1) % ACCUMULATION_STEPS == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

### 9. Streaming Cache（长期优化）

**原理**：维护一个 GPU cache，保留最近使用的 submodel，使用 LRU 策略替换

**实现**：
```python
class SubmodelCache:
    def __init__(self, max_gpu_submodels=3):
        self.cache = {}  # submodel_id -> gpu_tensor
        self.lru = []
        self.max_size = max_gpu_submodels

    def get(self, submodel_id, submodel_cpu):
        if submodel_id not in self.cache:
            if len(self.cache) >= self.max_size:
                # 移除最久未使用的
                evict_id = self.lru.pop(0)
                self.cache[evict_id].to_cpu()
                del self.cache[evict_id]

            # 加载到 GPU
            self.cache[submodel_id] = submodel_cpu.to_gpu()

        # 更新 LRU
        if submodel_id in self.lru:
            self.lru.remove(submodel_id)
        self.lru.append(submodel_id)

        return self.cache[submodel_id]
```

**预期加速**：10-15x（配合其他优化）

---

## 早停策略（Early Stopping for Submodels）

### 核心思想

某些 submodel 在某些视角下已经收敛，继续训练带来的收益非常小，可以直接跳过更新。

### 判断收敛的指标

#### 1. 基于梯度范数（最可靠）

```python
class SubmodelConvergenceTracker:
    def __init__(self, num_submodels, window_size=50, grad_threshold=1e-5):
        self.grad_history = [[] for _ in range(num_submodels)]
        self.window_size = window_size
        self.grad_threshold = grad_threshold
        self.converged = [False] * num_submodels
        self.frozen_since = [None] * num_submodels

    def update(self, submodel_id, grad_norm):
        """更新梯度历史"""
        self.grad_history[submodel_id].append(grad_norm)
        if len(self.grad_history[submodel_id]) > self.window_size:
            self.grad_history[submodel_id].pop(0)

    def check_convergence(self, submodel_id, iteration):
        """检查是否收敛"""
        if len(self.grad_history[submodel_id]) < self.window_size:
            return False

        # 计算最近 window_size 个 iteration 的平均梯度
        avg_grad = sum(self.grad_history[submodel_id]) / self.window_size

        # 计算梯度的标准差（稳定性）
        std_grad = np.std(self.grad_history[submodel_id])

        # 收敛条件：平均梯度小且稳定
        if avg_grad < self.grad_threshold and std_grad < self.grad_threshold * 0.5:
            if not self.converged[submodel_id]:
                self.converged[submodel_id] = True
                self.frozen_since[submodel_id] = iteration
                print(f"Submodel {submodel_id} converged at iteration {iteration}")
            return True

        return False
```

**优点**：
- 直接反映优化进度
- 理论基础扎实

**缺点**：
- 需要计算梯度范数（有一定开销）

---

#### 2. 基于 Loss 改善率

```python
class LossImprovementTracker:
    def __init__(self, num_submodels, window_size=100, improvement_threshold=0.001):
        self.loss_history = [[] for _ in range(num_submodels)]
        self.window_size = window_size
        self.improvement_threshold = improvement_threshold

    def update(self, submodel_id, loss_value):
        self.loss_history[submodel_id].append(loss_value)
        if len(self.loss_history[submodel_id]) > self.window_size:
            self.loss_history[submodel_id].pop(0)

    def check_convergence(self, submodel_id):
        if len(self.loss_history[submodel_id]) < self.window_size:
            return False

        # 比较前半段和后半段的平均 loss
        mid = self.window_size // 2
        first_half = sum(self.loss_history[submodel_id][:mid]) / mid
        second_half = sum(self.loss_history[submodel_id][mid:]) / mid

        # 改善率
        improvement = (first_half - second_half) / (first_half + 1e-8)

        return improvement < self.improvement_threshold
```

**优点**：
- 直接反映训练效果
- 计算简单

---

#### 3. 基于渲染贡献变化（最高效）

```python
class RenderContributionTracker:
    def __init__(self, num_submodels, window_size=30):
        self.contribution_history = [[] for _ in range(num_submodels)]
        self.window_size = window_size

    def update(self, submodel_id, contribution_percent):
        """contribution_percent: 该 submodel 对最终图像的贡献百分比"""
        self.contribution_history[submodel_id].append(contribution_percent)
        if len(self.contribution_history[submodel_id]) > self.window_size:
            self.contribution_history[submodel_id].pop(0)

    def check_convergence(self, submodel_id):
        if len(self.contribution_history[submodel_id]) < self.window_size:
            return False

        # 如果贡献率持续很低，说明这个块不重要
        avg_contribution = sum(self.contribution_history[submodel_id]) / self.window_size
        if avg_contribution < 0.02:  # 贡献小于 2%
            return True

        # 如果贡献率变化很小，说明已经稳定
        std_contribution = np.std(self.contribution_history[submodel_id])
        if std_contribution < 0.005:  # 变化小于 0.5%
            return True

        return False
```

**优点**：
- 无需额外计算（代码中已有 contributed_percent）
- 直观易懂

---

### 组合策略（推荐）

```python
class SmartSubmodelScheduler:
    def __init__(self, num_submodels, config):
        self.num_submodels = num_submodels
        self.grad_tracker = SubmodelConvergenceTracker(num_submodels)
        self.loss_tracker = LossImprovementTracker(num_submodels)
        self.contrib_tracker = RenderContributionTracker(num_submodels)

        # 收敛状态
        self.frozen = [False] * num_submodels
        self.frozen_iterations = [0] * num_submodels

        # 唤醒机制
        self.global_loss_history = []
        self.wake_up_threshold = 0.05  # 如果全局 loss 上升 5%，唤醒所有块

    def should_skip_training(self, submodel_id, iteration):
        """判断是否应该跳过这个 submodel 的训练"""
        if not self.frozen[submodel_id]:
            return False

        # 每隔一段时间检查是否需要唤醒
        if iteration - self.frozen_iterations[submodel_id] > 500:
            # 检查全局 loss 是否上升
            if self._should_wake_up():
                self.frozen[submodel_id] = False
                print(f"Wake up submodel {submodel_id} at iteration {iteration}")
                return False

        return True

    def update_and_check(self, submodel_id, iteration, grad_norm, loss_value, contribution):
        """更新统计信息并检查是否应该冻结"""
        if self.frozen[submodel_id]:
            return

        self.grad_tracker.update(submodel_id, grad_norm)
        self.loss_tracker.update(submodel_id, loss_value)
        self.contrib_tracker.update(submodel_id, contribution)

        # 组合判断：满足任意两个条件就冻结
        conditions = [
            self.grad_tracker.check_convergence(submodel_id, iteration),
            self.loss_tracker.check_convergence(submodel_id),
            self.contrib_tracker.check_convergence(submodel_id)
        ]

        if sum(conditions) >= 2:
            self.frozen[submodel_id] = True
            self.frozen_iterations[submodel_id] = iteration
            print(f"🔒 Freeze submodel {submodel_id} at iteration {iteration}")
            print(f"   Grad converged: {conditions[0]}, Loss converged: {conditions[1]}, Contrib stable: {conditions[2]}")

    def _should_wake_up(self):
        """检查是否需要唤醒冻结的 submodel"""
        if len(self.global_loss_history) < 100:
            return False

        recent_loss = sum(self.global_loss_history[-20:]) / 20
        old_loss = sum(self.global_loss_history[-100:-80]) / 20

        # 如果 loss 上升超过阈值，唤醒所有块
        if recent_loss > old_loss * (1 + self.wake_up_threshold):
            return True

        return False

    def update_global_loss(self, loss):
        self.global_loss_history.append(loss)
        if len(self.global_loss_history) > 1000:
            self.global_loss_history.pop(0)

    def get_stats(self):
        """获取统计信息"""
        num_frozen = sum(self.frozen)
        return {
            "frozen_submodels": num_frozen,
            "active_submodels": self.num_submodels - num_frozen,
            "frozen_ratio": num_frozen / self.num_submodels
        }
```

### 集成到训练代码

```python
# 在 training_phase_2 开始时初始化
scheduler = SmartSubmodelScheduler(
    num_submodels=len(submodel_list),
    config={
        'grad_threshold': 1e-5,
        'improvement_threshold': 0.001,
        'window_size': 50
    }
)

# 在训练循环中
for submodel_id, rank_map in zip(visible_submodel_id_list, block_rank):
    submodel: GaussianModel = submodel_list[submodel_id]

    # 🔥 检查是否应该跳过
    if scheduler.should_skip_training(submodel_id, iteration):
        continue

    submodel.move_and_activate_subset()
    render_pkg = render(...)

    # ... 计算 loss ...

    loss.backward()

    with torch.no_grad():
        # 计算梯度范数
        grad_norm = 0.0
        for param in [submodel._xyz_gpu, submodel._opacity_gpu,
                      submodel._scaling_gpu, submodel._rotation_gpu]:
            if param.grad is not None:
                grad_norm += param.grad.norm().item() ** 2
        grad_norm = grad_norm ** 0.5

        # 计算贡献度（代码中已有）
        contribution = contributed_percent

        # 更新并检查收敛
        scheduler.update_and_check(
            submodel_id, iteration,
            grad_norm, loss.item(), contribution
        )

        # Optimizer step
        if iteration < opt.iterations:
            submodel.optimizer.step()
            submodel.optimizer.zero_grad(set_to_none=True)

# 更新全局 loss
scheduler.update_global_loss(ema_loss_for_log)

# 定期打印统计
if iteration % 100 == 0:
    stats = scheduler.get_stats()
    print(f"Frozen: {stats['frozen_submodels']}/{scheduler.num_submodels} ({stats['frozen_ratio']*100:.1f}%)")
```

### 预期效果

- **早期（0-5k iterations）**：几乎所有块都在训练
- **中期（5k-15k iterations）**：30-50% 的块被冻结（主要是背景和远处区域）
- **后期（15k-30k iterations）**：60-80% 的块被冻结

**额外加速比**：2-5x

---

## 优化优先级建议

### 立即实施（最大收益/最小改动）
1. **批量处理 Submodel**（方案 2）
2. **Lazy Frustum Culling**（方案 5）
3. **早停策略**（基于渲染贡献）

**预期总加速**：5-10x

### 中期实施
4. **异步数据传输**（方案 3）
5. **混合精度训练**（方案 6）

**预期总加速**：10-15x

### 长期优化
6. **Streaming Cache**（方案 9）
7. **Per-View 收敛追踪**

**预期总加速**：15-20x

---

## 参考文献和相关工作

- **Mega-NeRF**: 大规模场景的分块训练策略
- **Block-NeRF**: 城市级别场景的分块渲染
- **Instant-NGP**: 使用 occupancy grid 跳过空区域
- **Neural Radiance Fields**: Importance sampling 策略
- **PyTorch Gradient Checkpointing**: 内存优化技术

---

## 实验记录

### 实验 1：Baseline
- **配置**：原始代码
- **速度**：[待填写]
- **GPU 内存**：[待填写]

### 实验 2：批量处理 + Lazy FC
- **配置**：BATCH_SIZE=3, FC_INTERVAL=5
- **速度**：[待填写]
- **加速比**：[待填写]

### 实验 3：加入早停策略
- **配置**：grad_threshold=1e-5, window_size=50
- **速度**：[待填写]
- **冻结比例**：[待填写]
- **加速比**：[待填写]

---

## 注意事项

1. **梯度累积可能影响收敛**：如果使用 gradient accumulation，需要调整学习率
2. **混合精度可能影响数值稳定性**：某些操作（如 softmax）需要保持 float32
3. **早停策略需要调参**：不同场景的最优阈值可能不同
4. **唤醒机制很重要**：避免过早冻结导致欠拟合

---

## TODO

- [ ] 实现批量处理 Submodel
- [ ] 实现 Lazy Frustum Culling
- [ ] 实现早停策略（基于渲染贡献）
- [ ] 实验并记录加速效果
- [ ] 实现异步数据传输
- [ ] 实现混合精度训练
- [ ] 实现 Streaming Cache
- [ ] 对比不同优化策略的效果

---

*最后更新：2026-02-13*
