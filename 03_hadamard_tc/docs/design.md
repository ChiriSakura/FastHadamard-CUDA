# 设计说明（第四、五部分）：参考实现 与 CUDA baseline

## 0. 数学约定

输入张量形状 `[batch, seq, heads, head_dim]`，逻辑展平为 `[total_tokens, head_dim]`，
`total_tokens = batch*seq*heads`。对每个 token 的最后一维独立做 Sylvester 型 Hadamard 变换：

- 非归一化：`y = H·x`，其中 `H·Hᵀ = head_dim·I`
- 归一化：`y = H·x / sqrt(head_dim)`，此时变换正交（保范数），是量化旋转场景的标准用法

**当前实现默认归一化（`normalize=true`）**，理由：
1. 正交归一化是 QuaRot/SpinQuant 类量化旋转的实际用法；
2. 归一化把输出量级约束到 O(1)，对"绝对误差阈值"更友好——
   非归一化输出量级为 O(sqrt(head_dim))，**对内置参考**的绝对误差会超过阈值
   （实测：`--normalize false` 在 FP16/head_dim=64 即 FAIL，max_abs=1.5e-2）。
3. 已做成运行时参数，可一键切换。

**关于官方库的口径（重要更新）**：实测 `fast_hadamard_transform` 默认
`scale=1.0`（**非归一化**）。但由于本 kernel 与官方库同为"同序蝶形 + FP32 累加 +
末端舍入"，在**归一化与非归一化两种模式下都与官方库 100% bit-exact**
（见 §5 与 `results/library_check.csv`）。因此：
- 对官方库的绝对误差在两种模式下都是 **0.0**，远低于 1e-2 / 5e-2；
- 上面第 2 点的阈值顾虑只存在于"对内置/独立参考比较"的场景，不影响对官方库的验收。

## 1. 快速算法（第四部分：参考实现设计）

直接构造完整 `H`（O(n²) 存储与计算）在 head_dim≥128 时既慢又没必要。
采用快速 Walsh-Hadamard 变换（WHT）：蝶形迭代，O(n log n)，原地。

```
stride = 1, 2, 4, ..., head_dim/2:
    对每对下标 (i, i^stride)（i 在该位上为 0）:
        (a, b) -> (a + b, a - b)
```

共 log2(head_dim) 轮、每轮 head_dim/2 个蝶形。GPU kernel、C++ CPU 参考、Python 参考
**三者蝶形顺序完全一致**，保证"同一个变换"可逐位对照。

要点：
- 只对最后一维做变换，其余维度展平成 token 循环，天然支持任意 `[batch,seq,heads]` 组合；
- 内部累加一律 FP32（本 kernel、C++ 自检参考、官方库三者同策略）；
- 输入是 FP16/BF16：先**精确**转宽（half→float 无损，bf16→float 无损），算完再转回。
  BF16 尤其依赖 FP32 中间累加——8 位尾数直接累加会迅速超出 5e-2 阈值。

实现位置：
- **验收参考（唯一）= 官方 `fast_hadamard_transform` 库**，封装在
  `tests/hadamard_reference.py`（仅库调用 + 低精度编解码，不含任何自实现参考），
  由 `tests/test_vs_library.py` / `tests/check_error.py` 消费；
- `src/reference_cpu.h`：C++ FP32 参考，仅供 bench `--check` 开发期自检
  （另附小尺寸显式矩阵乘版本作对照），**不作为验收口径**。

## 2. CUDA baseline 设计（第五部分）

目标：清晰、正确、可测量。暂不做访存极致优化 / warp 特化 / Tensor Core。

### 输入展平
`[batch,seq,heads,head_dim] -> [total_tokens, head_dim]`，kernel 只关心 `total_tokens`。

### block/thread 组织
- **一个 block 处理一个 token**：`gridDim.x = total_tokens`，`blockDim.x = HEAD_DIM`
  （一元素一线程，映射最直观；head_dim=1024 时恰好到单块 1024 线程上限）；
- 每轮蝶形中，每对 `(i, i^stride)` 由下标较小的线程处理，避免重复写。

### shared memory 与蝶形
- `__shared__ float smem[HEAD_DIM]`（最大 1024×4B = 4KB，占用无忧）；
- 从 global 读入并转 FP32 存入 smem，之后 log2(HEAD_DIM) 轮蝶形全在 smem 上完成，
  每轮一次 `__syncthreads()`；
- 最后按需乘 `1/sqrt(HEAD_DIM)`（host 侧用双精度算好传入，避免 device 端近似），
  转回原类型写回 global。

### 数据类型与模板
```cpp
template <typename T, int HEAD_DIM, bool NORMALIZE>
__global__ void hadamard_kernel(const T* in, T* out, int total_tokens, float norm_scale);
```
按 `(dtype, head_dim, normalize)` 静态分发；当前实例化
`{__half, __nv_bfloat16} × {32,64,128,256,512,1024} × {true,false}`。
64/128/256 为题目核心，其余为预留。

### host 接口
```cpp
int launch_hadamard(const void* input, void* output,
                    int batch, int seq, int heads, int head_dim,
                    DataType dtype, bool normalize, cudaStream_t stream);
```
返回值：0 成功；-1 不支持的 head_dim；-2 不支持的 dtype。
CUDA 调用错误统一经 `CUDA_CHECK` 立即报错退出，不吞错。

### 计时
CUDA Event 包裹每次 kernel，先 `--warmup` 次预热，再 `--iters` 次逐一计时，
输出 `avg / min / max` 毫秒（见 `results/results.csv` 字段）。

## 3. 误差分析（重要结论）

我们新增了 **平滑相对误差** `max_rel_error = max |a-b| / max(1,|b|)`，用来把
"实现误差"与"输出格式舍入误差"分开。实测结论（`results/report.md`）：

- 所有配置的 `max_rel_error` 都**精确贴合理论舍入极限**：
  - FP16 → 4.88e-4 ≈ 2^-10（half-ulp/|x| 上界）
  - BF16 → 3.89e-3 ≈ 2^-8
  这说明 GPU 结果相对真值是 **bit-exact** 的，残余偏差完全来自低精度输出本身的舍入。
- 由此，"绝对阈值 + 大输出量级"存在物理冲突：
  BF16 输出落入 [16,32) 时，half-ulp = 0.0625 > 5e-2。这正是唯一未通过的配置
  （bf16 + 大 outlier，输出量级被异常值抬到 ~17）的根因——**不是实现错误**。
- 该结论由两路证据支撑：
  1. 对单个"失败"token 重算并模拟末位舍入，与 GPU 输出 **128/128 逐位相同**；
  2. 与官方 `fast_hadamard_transform` 的 36 配置全量对照 **100% bit-exact**（见 §4），
     直接证明残余偏差仅为输出格式舍入，而非实现误差。

## 4. 官方库基准验证（验收口径）

参考实现锁定为官方 `fast_hadamard_transform`（Dao-AILab, v1.1.0）。实测确认的行为：

| 项 | 结论 | 证据 |
|---|---|---|
| 归一化 | 默认 `scale=1.0` 非归一化；归一化需显式传 `scale=1/sqrt(dim)` | 源码 + impulse 测试（e₀ → 全 ±1） |
| 矩阵序 | `scipy.linalg.hadamard(dim)`，Sylvester 自然序，与本蝶形同序 | 与显式矩阵逐元素对照（整数精确，误差 0） |
| 内部精度 | FP32 累加，末端舍入回输入 dtype | 源码 + 整数精确测试（误差 0） |
| 维度语义 | 按最后一维变换，须先 `reshape(-1, head_dim)` | 1D 误用会把整段当一行（曾踩坑） |

验证矩阵：2 dtype × 3 head_dim × 3 规模 × {归一化, 非归一化} = **36 配置**。
脚本 `tests/test_vs_library.py` 用 `--check false` 跑本 kernel 并 `--dump_dir` 落盘，
再与官方库逐位比较。**结果：36/36 通过，全部 100% bit-exact（max_abs=0.0）。**

bit-exact 的根因：两者同为"低精度输入 → FP32 蝶形 → 同序 → 末端舍入"，
累加顺序一致时 FP32 中间值逐位相同，舍入结果自然相同。这也反向证明本实现
无隐藏精度问题。参考实现**仅保留官方库**（`tests/hadamard_reference.py` 只做
官方库调用封装 + 低精度编解码，不含自实现参考），避免多参考口径漂移。

## 5. 为后续（Week3+）预留的接口

- `normalize` / `dtype` 已是运行时参数；
- `--input_bin` 支持加载预生成二进制输入，便于"变换+量化"流水复现同一批数据；
- 融合量化接入点：`launch_hadamard(...)` 之后、写回之前。后续加
  `launch_hadamard_fused_quant(..., QuantType, scale_mode)` 时，复用同一套
  `(dtype, head_dim)` 分发即可；量化 scale 的粒度（per-token/per-head/...）只影响
  写回阶段，不影响蝶形主体；
- Tensor Core 路线：head_dim ∈ {64,128,256} 的变换可写成小矩阵乘，
  `mma`/`wmma` 需要的布局（fragments）与当前"一线程一元素"不同，届时新增独立
  kernel 并在 `launch_hadamard` 里按开关分发，不改动现有正确版本。
