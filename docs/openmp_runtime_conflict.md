# OpenMP Runtime 冲突：堆损坏崩溃的根因与修复

## 现象

训练到几千 iter 随机崩溃：

```
Training progress: 12%|█▏ | 3475/30000 [01:07<09:49]
free(): corrupted unsorted chunks
```

其他可能出现的变体：
- `corrupted size vs. prev_size`
- `corrupted double-linked list`
- PyTorch 内部 assertion failure (`TensorAdvancedIndexing.cpp`)

非确定性出现，有时跑几百 iter 就崩，有时跑几千 iter。

---

## 根因：同一进程里加载了两个 OpenMP runtime

| 来源 | 库 | 路径 |
|---|---|---|
| PyTorch / MKL (conda) | **libomp.so** (LLVM) | `~/miniconda3/envs/.../lib/libomp.so` |
| cpu_adam.cpp (GCC `-fopenmp`) | **libgomp.so** (GNU) | `/usr/lib/gcc/x86_64-linux-gnu/11/libgomp.so` |

两个 runtime 各自维护线程池和内部 malloc/free，互相踩内存 → glibc 检测到堆损坏 → abort。

### 为什么 conda 的重定向没生效？

conda `_openmp_mutex` 包做了一个巧妙的重定向：

```
$CONDA_ENV/lib/libgomp.so.1  →  libomp.so  (symlink)
```

如果程序链接的是 conda lib 目录下的 `libgomp.so.1`，实际用的就是 libomp，不会冲突。

**但问题是**：GCC 的 `-fopenmp` flag 在链接时，直接找到了系统 GCC 自带的 libgomp：

```bash
$ gcc -fopenmp -print-file-name=libgomp.so
/usr/lib/gcc/x86_64-linux-gnu/11/libgomp.so    # ← 真正的 GNU libgomp，不是 conda 的
```

所以 conda 的重定向被完全绕过了。

### 环境信息

```
conda env: partgs_fix_2
_openmp_mutex: 4.5  7_kmp_llvm    (选择 LLVM OpenMP)
llvm-openmp:   22.1.0             (提供 libomp.so)
pytorch:       2.0.1 py3.8_cuda11.7  (内部用 libomp)
系统 GCC:      11 (/usr/lib/gcc/x86_64-linux-gnu/11/)
```

---

## 修复方案

### 核心思路

**C++ 代码完全不动**。所有 `#pragma omp parallel for`、`omp_set_num_threads(64)`、`omp_get_thread_num()` 原样保留。

**只改 `setup.py` 的链接方式**：编译时保留 `-fopenmp`（让 GCC 认 pragma），链接时换成 conda 的 `-lomp`。

LLVM 的 libomp 内置了完整的 GOMP 兼容层（263 个 `GOMP_*` 符号），GCC 编译出的代码可以直接跑在 libomp 上。

### setup.py 具体改动

```python
# ===== 以前（有问题）=====
extra_compile_args={"cxx": ["-O3", "-fopenmp", "-std=c++17", "-march=native"]},
extra_link_args=["-fopenmp"]
#                ^^^^^^^^^ GCC 链接时自动找到 /usr/lib/gcc/.../libgomp.so

# ===== 现在（修复后）=====
import sys
conda_lib = os.path.join(sys.prefix, 'lib')  # → ~/miniconda3/envs/partgs_fix_2/lib
omp_link_args = [f"-L{conda_lib}", "-lomp", f"-Wl,-rpath,{conda_lib}",
                 "-Wl,--no-as-needed"]

extra_compile_args={"cxx": ["-O3", "-fopenmp", "-std=c++17", "-march=native"]},
extra_link_args=omp_link_args
```

各 flag 作用：
- `-fopenmp`（仅在 compile args）：GCC 识别 `#pragma omp`，生成 `GOMP_*` 调用
- `-L{conda_lib}`：链接时优先搜索 conda 的 lib 目录
- `-lomp`：链接 libomp.so（不是 libgomp）
- `-Wl,-rpath,...`：运行时也从 conda lib 加载
- `-Wl,--no-as-needed`：确保 libomp 不被 linker 优化掉

### 重新编译

```bash
conda activate partgs_fix_2
pip install submodules/diff-gaussian-rasterization
```

### 验证

```bash
# 编译后检查 .so 链接了哪个 OpenMP：
ldd $(python -c "import diff_gaussian_rasterization_wenqi_tam._C as m; print(m.__file__)") | grep omp
# 应该看到：libomp.so → ~/miniconda3/envs/.../lib/libomp.so
# 不应该看到：libgomp.so → /usr/lib/...
```

---

## 为什么之前的方案不好

| 方案 | 问题 |
|---|---|
| 去掉 `-fopenmp` | `#pragma omp` 变 no-op，全部单线程，adam/frustum culling 极慢 |
| 换 `at::parallel_for` | 可行但明显慢于原生 OpenMP：dispatch 开销大，不支持 `collapse(2)`、`num_threads(64)` 等 |
| Python fallback frustum culling | 比 C++ OpenMP 版本慢很多 |

当前方案（链接 libomp）保留了所有 OpenMP 特性，速度零损失。

---

## 注意事项

1. **submodule reset 会丢失修改**：`git submodule update --init` 会覆盖 setup.py，需要重新应用
2. **Ada 6000 服务器**：可能不受此问题影响（GCC/PyTorch 版本匹配），但此修复无害
3. **升级 PyTorch 或 GCC 后**：需要重新验证 libomp 兼容性（通常没问题）
