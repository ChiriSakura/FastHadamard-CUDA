# Hadamard 变换加速项目总结报告

> **2026-09-09 审计修订**：下文历史性能数据保留，但撤回“TC 普遍正确舍入等价”
> 和“37 个配置全面验收通过”的结论。严格全张量验收、尾部修复、A/B 调优的最新
> 状态以 [审计与优化记录](audit_optimization.md) 为准。NCU 已在 H200 成功采集，
> 旧节点的 DCGM 阻塞日志不是当前总体状态。

> 项目：2026 夏季训练营 CUDA 方向项目·选题三
>
> 当前阶段：Week1/Week2 baseline + 9.1 kernel 优化 + 9.2 融合 INT4
> + warp-per-token 再优化 + Tensor Core 对比
>
> 最近实测：H200 A/B `17249309`、`17251073`；H100 参考库验收 `17251286`、
> NCU 缓存策略对照 `17251063`，全部已完成。历史 L40S 主表来自
> 2026-09-06 job `17083405`，不与本轮跨 GPU 混算。
>
> 报告状态：baseline、优化 FHT、融合量化、warp-per-token FHT 与 Tensor Core
> 对比均已填写并附原始产物

## 1. 摘要

后续参考库性能基线与四阶段执行记录见 [下一轮优化](next_optimization.md)：
job `17289704` 在 H200 完成 80 配置的同边界 Graph 回放比较和输出检查。
它补充“相对外部参考库”的性能证据，不替代本文既有 CUDA Event／NCU 计时口径。
后续 `17290172` 的七组 warp/INT4 参数共 1848 行验证全过；`17290602` 定位了
11 个 split 失败输入的第一 MMA、残差表示、后续累加和最终舍入误差。
`17290837` 的独立 PTX 重排在 L40S d=64 上，相对向量化 WMMA 的 FP16 fast/split
再快 1.30×/1.41×，与对应 WMMA 的 108 配置输出哈希一致。TC 严格精度失败未修复，
不切换默认；当时新 PTX 的 L40S NCU 尝试被采集资源占用阻塞，不能冒充已有 H100 数据。

后续补测已在 H200 `gh108` 取得新 PTX/WMMA/warp 的 **30 项 NCU 原始 CSV**，
覆盖 D=64/128/256 和 cache-control=all/none。D=128 PTX fast 的 dynamic shared
由 WMMA 的 9728 B/block 降到 0，short-scoreboard 由 6.164 降到 0.976，
该次 NCU 时间由 6.304 降到 4.992 μs。完整采集、配额故障恢复及精度实验记录见
[Hopper 缺口补测](gap_closure.md)。旧 L40S 的阻塞不是当前总体状态。

最终补跑 `17292654`、`17294196` 均 COMPLETED。D1024 的选择性宽写回已实现，
H200 FP16/BF16 融合 INT4 分别为 170.600→160.623 μs（1.062×）、
172.215→162.919 μs（1.057×），D512 保持基本不变；528 行输出检查全过。
新的保护式 PTX split 通过 108/108 核心配置、192/192 压力测试及三类 sanitizer，
但 FP16 D64/128/256 为 20.927/39.334/72.718 μs，均慢于同卡严格 warp，
所以没有替换默认。纯 TC 的精度/速度兼得仍未实现；保护通过不能包装成纯 TC 成功。
[最终保护成本与量化收益表](../tensor_core/results/guard_quant_17294196/summary.md)。

2026-09-09 第二轮新增实测：H200 选中组合 `WARPS=0, TC_WARPS=4, TC_VECTOR=1`
使 FP16 d=64 warp 从 24.8502 μs 降到 16.1386 μs（1.540×），其融合 INT4 从
25.2122 μs 降到 22.5171 μs（1.120×）；d=128 TC fast/split 分别为
46.9066→36.2342 μs（1.295×）、57.1277→39.9834 μs（1.429×），d=256 split
为 108.3900→73.7037 μs（1.471×）。均为同后端、同 GPU 三次扫描中位数。
裁剪末尾同步仅带来约 ±0.55% 波动，未采用。可选构建 `make -C tensor_core h200-tuned`，
不切换通用默认、不把 warp 自动替换为数值不同的 TC。

五组参数分别完成 132 配置独立 FP32 参考和 132 配置 Dao 参考库验收，两类结果一致：
baseline/optimized/warp 均 132/132、绝对误差 0；TC fast/split 分别为 96/132、121/132，
最大绝对误差 1/0.5，不能宣称全部配置合规。融合与同算法 unfused 的 packed INT4 和
scale 均逐位一致（非 TC 每组 132 配置，TC 每组 108 配置，TC d>256 不支持）。
跨参数输出哈希差异为 0；75 项 sanitizer 无错误；H100 上新增 40 次 NCU 采集全部成功。
TC fast 的 short-scoreboard 由 12.922 降到 5.994，支持 shared 中转优化方向。
[完整优化历程](audit_optimization.md)、
[最终性能/参考库验收/NCU/Roofline 汇总](../tensor_core/results/verify_17251073/final_library/summary.md)。

本项目实现了面向 `[batch_size, seq_len, num_heads, head_dim]` 激活张量的 CUDA
快速 Walsh-Hadamard Transform（FHT）。实现支持 FP16、BF16，核心 `head_dim` 为
64/128/256，并额外实例化 32/512/1024。除每 block 一个 token、全程 shared-memory
蝶形的教学 baseline 外，9.1 版本使用 packed I/O、warp shuffle 和多 token/block；
9.2 版本进一步融合逐 token 对称 INT4 量化，直接输出 packed nibbles 和 FP32 scale。

以 Dao-AILab `fast_hadamard_transform` 1.1.0 为验收参考，在 2 种 dtype、3 种
`head_dim`、3 档输入规模、归一化/非归一化两种语义组成的 36 个配置上，当前实现
全部通过，`max_abs_error=0`，输出逐位一致率为 100%。在 L40S、131072 tokens 上，
优化 FHT 相对 baseline 加速 1.24x–3.42x；融合 INT4 相对 unfused pipeline 加速
1.26x–2.33x。六个核心配置均同时通过 optimized-vs-baseline、fused-vs-unfused、
GPU-vs-CPU quantizer 三项 bit-exact 检查。

本轮进一步补齐两项。第一项是把 9.1 的 “每线程 2 元素” 提升为 **warp-per-token**：
每线程持 `head_dim/32` 个元素，蝶形全部落在寄存器与 `__shfl_xor_sync` 上，
shared memory 与 `__syncthreads()` 用量降到 0；变换 kernel 在 d=128 上再快
1.68x，融合 INT4 相对已验收的 9.2 融合 kernel 快 1.07x–3.34x（核心 d=64/128/256 为
2.13x–3.34x），且与 9.1 输出逐位一致。第二项是 **Tensor Core 分支**（[`tensor_core/`](../tensor_core/README.md)）：
用 `H_d = H_{d/16} ⊗ H_16` 把 FHT 改写成两次 16x16x16 WMMA，而不是退化成稠密
GEMM；配套的 hi/lo 三次 mma 变体（`tc_split`）降低中间舍入误差，
这组 FP16 d<=256 样本逐位一致，但其他配置不保证同值。Tensor Core 在 d=64 上确实
最快（变换 1.10x–1.16x、融合 INT4 1.11x–1.19x，均相对最好的非 TC 实现），
d>=128 则输给 warp 版本。H200 上采到的硬件计数器给出了直接原因：warp 版本的
`stall barrier` 精确为 0、主导 stall 是 global memory 延迟（访存受限的正常形态），
而两条 TC kernel 的主导 stall 换成了 `short scoreboard` 与 `mio throttle`
（shared memory / MIO 压力）—— 代价不在 mma，而在 WMMA 必须做的那次
accumulator → matrix_b shared memory 中转。

### 1.1 交付状态总览

| 交付项 | 状态 | 主要证据 |
|---|---|---|
| FP16/BF16 baseline | 完成 | 官方库 36/36，逐位一致 |
| 9.1 optimized FHT | 完成 | L40S 1.24x–3.42x；核心 6/6 bit-exact |
| 9.2 fused INT4 | 完成 | L40S 1.26x–2.33x；三类一致性检查全过 |
| 扩展 shape/normalize | 完成 | 24/24 配置三项 bit-exact |
| Nsight Systems | 完成 | kernel timeline、launch 形状、register/shared-memory、API 汇总 |
| Roofline | 完成（逻辑 I/O 与实测 DRAM 分列） | 历史 L40S 逻辑图；新增 H100 all/none 实测图 |
| Nsight Compute 硬件计数器 | **完成（H200）** | optimized/fused 的 throughput、occupancy、cache、stall 已实测 |
| warp-per-token FHT | 完成 | d=128 再快 1.68x；与 9.1 输出 100% 逐位一致 |
| warp-per-token 融合 INT4 | 完成 | 相对 9.2 融合 kernel 快 2.13x–3.34x（d=64/128/256），逐位一致 |
| Tensor Core 变换（WMMA） | 实现完成，严格检查检出失败 | 独立 FP32 与真实参考库均 fast 96/132、split 121/132 |
| Tensor Core 融合 INT4 | 实现完成，尾部已修复 | 已测配置与同算法 unfused 一致；d>256 由调用方选 warp |
| Tensor Core vs 非 TC 对比 | 完成 | 同一进程/stream/输入下五条变换 + 五条融合路径 A/B |
| warp/TC 的 NCU 硬件计数器 | **完成（H200）** | stall 构成直接印证 shared 中转是 TC 的瓶颈 |
| 新增 NCU 缓存策略与 Roofline | 完成（H100） | job 17251063，40/40 all/none 采集与实际 DRAM 图 |

L40S 与 A100 节点仍被 DCGM 占用 counter，但 H200 `gh112` 成功完成 18-pass replay，
因此不再把 NCU 总项标成未完成。跨 GPU 数值不混入 L40S A/B 加速表：L40S 用于最终
端到端性能，H200 用于真实硬件计数器诊断，两者实验目的和硬件均明确标注。

## 2. 背景与目标

激活中的少量异常值会放大量化 scale，挤压大多数普通值的有效量化区间。正交旋转把
能量扩散到多个通道，可降低单通道动态范围，同时保持全精度网络的等价性。QuaRot
展示了旋转后端到端 4 bit 权重、激活和 KV cache 量化；SpinQuant 进一步学习旋转矩阵；
FlashAttention-3 则将 incoherent processing 与 FP8 block quantization 用于 Hopper
注意力计算。相关一手资料见文末参考文献。

项目目标不是只追求单次最快数字，而是建立可验证、可复现的优化闭环：

1. 支持题目要求的形状、FP16/BF16 和 64/128/256 维度；
2. 与官方 CUDA 参考库对齐变换次序、归一化和输出 dtype；
3. 建立可复现的正确性矩阵、CUDA Event 性能日志和 profiler 入口；
4. 用 profiler 数据驱动 shuffle/vectorized/multi-token 优化；
5. 固定量化协议并验证 fused 与 unfused bit-exact；
6. 在 Tensor Core 上换算法（而不是换后端）实现同一变换，并与做到位的非 Tensor
   Core 实现在同环境下对比误差、单 kernel 时间和融合后的端到端时间。

## 3. 数学定义与算法

令 `d=head_dim`，且 `d` 为 2 的幂。Sylvester Hadamard 矩阵满足：

```text
H_1 = [1]
H_2d = [[H_d,  H_d],
        [H_d, -H_d]]
H_d H_d^T = d I
```

实现同时支持：

- 非归一化：`y = H_d x`；
- 归一化：`y = H_d x / sqrt(d)`，此时为正交变换，也是默认量化旋转语义。

直接矩阵乘的复杂度为 `O(d^2)`。FHT 利用 Kronecker 结构进行蝶形分解，每轮执行：

```text
(a, b) -> (a + b, a - b)
stride = 1, 2, 4, ..., d/2
```

因此每个 token 的计算复杂度降为 `O(d log d)`，额外存储为 `O(d)`。

## 4. CUDA baseline 实现

### 4.1 数据布局与线程映射

前 3 个维度展平成 `total_tokens=batch*seq*heads`。当前映射为“一 block 一 token、
一线程一元素”：`gridDim.x=total_tokens`，`blockDim.x=head_dim`。这种方案结构直接，
便于先证明正确性；它不是最终最优映射。

### 4.2 数值路径

1. FP16/BF16 从 global memory 读入；
2. 精确转换为 FP32 并写入 shared memory；
3. 在 shared memory 上完成全部蝶形，每轮后 block 同步；
4. 可选乘 `1/sqrt(d)`；
5. 只在最终写回时舍入到 FP16/BF16。

FP32 中间计算是 BF16 达到误差要求的关键。若直接以 BF16 逐轮累加，误差会随蝶形
级数快速传播。

### 4.3 工程结构

- `include/hadamard.cuh`：稳定的 host API、dtype 与返回码；
- `src/hadamard.cu`：baseline、9.1 optimized、9.2 fused INT4 的模板 kernel 和
  `(dtype, head_dim, normalize)` 静态分发；
- `src/main.cu`：输入生成、参数解析、CUDA Event 计时、dump 与 CSV；
- `src/hadamard_warp.cu`：warp-per-token FHT 与其融合 INT4 版本（本轮新增，
  独立编译单元，不改动已验收路径）；
- `src/advanced_bench.cu`：9.1/9.2 A/B 计时、CPU INT4 reference 和 bit-exact 检查；
- `tensor_core/`：Tensor Core (WMMA) 实验分支，含 kernel、对照基准、扫描脚本、
  Slurm 任务和结果，详见 [`tensor_core/README.md`](../tensor_core/README.md)；
- `src/reference_cpu.h`：开发期 FP32 自检；
- `tests/test_vs_library.py`：正式验收矩阵；
- `scripts/run_tests.sh`：一键正确性与性能扫描；
- `scripts/profile.sh`：ncu/nsys 采集和旧版 qdstrm 导入兼容。

关键函数均注明张量语义、精度策略、同步原因和错误码。所有 CUDA Runtime 调用统一经
`CUDA_CHECK` 检查，kernel launch 后检查异步错误，不静默吞错。

### 4.4 9.1 optimized FHT

每个线程负责相邻两个元素，使用 `half2`/`__nv_bfloat162` 做 32-bit 合并 load/store。
stride=1 的加减留在同一线程的两个 FP32 寄存器中；stride=2–32 通过
`__shfl_xor_sync` 交换，不再为每一轮访问 shared memory 和执行 block barrier；只有
stride>=64 的跨 warp 阶段才使用 shared memory。d=64 每 block 处理 4 tokens，d=128
每 block 处理 2 tokens，减少 block 数并提高每个 block 的有效工作量。

原 `launch_hadamard` 不变，新增 `launch_hadamard_optimized`，从而保证 baseline 与优化
版本使用同一输入、同一 stream、同一 Event 计时边界，且能做逐位回归。

### 4.5 9.2 fused INT4

量化协议固定为逐 token 对称 INT4：

```text
scale = max(abs(x)) / 7                     # 全零 token 使用 scale=1
q = clamp(round_to_nearest_even(x/scale), -7, 7)
```

相邻两个 `q` 按 two's-complement 分别放入一个 byte 的低/高 nibble；每 token 另存一个
FP32 scale。unfused 路径为 `optimized FHT -> 低精度中间张量 -> quantize`；fused 路径
在 FHT 寄存器结果上先模拟一次输入 dtype 的舍入，再直接求 max/scale/INT4，避免中间
张量写回和重读。这个显式舍入是 fused 与 unfused 可严格 bit-exact 的关键。

量化 kernel 中每个线程处理相邻两个元素：先求本地 `max(abs(x0),abs(x1))`，再用
`__shfl_down_sync` 做 warp max；每个 warp 仅写一个 shared-memory 最大值，thread 0
完成最终 token max 和 scale，最后所有线程并行舍入、饱和并打包。fused kernel 采用
每 block 一个 token，因为 scale 依赖整个 token；相比 unfused，它少一次完整低精度
中间张量 global write、一次 global read 和一次 kernel launch。

### 4.6 warp-per-token FHT（本轮新增）

9.1 版本的结构性瓶颈来自 “每线程只持 2 个元素”：一个 token 需要 `d/2` 个线程，
所以 `d=128/256` 时 token 跨多个 warp，stride>=64 的蝶形必须走 shared memory +
`__syncthreads()`，而且那些阶段用 `if (partner > local_pair)` 只让一半线程干活。
d=256 因此只拿到约 1.25x 加速。

本轮把每线程负责的元素数提升到 `ELEMS = head_dim / 32`，于是**一个 warp 恰好
处理一个 token**：

- 全局下标 = `lane * ELEMS + i`；
- `stride < ELEMS` 的低阶蝶形完全在同一线程的寄存器数组内完成；
- `stride >= ELEMS` 的阶段配对元素落在 `lane ^ (stride/ELEMS)` 上，`i` 不变，
  因此恒定是 **5 次** `__shfl_xor_sync`（因为 `d/ELEMS = 32`），与 d 无关；
- 全程 **0 shared memory、0 `__syncthreads()`**，所有线程每一级都在干活；
- global 访问宽度提升到 `ELEMS*2` 字节：d=64 为 32-bit、d=128 为 64-bit、
  d=256 及以上为 128-bit（多个 `int4` chunk）。

融合 INT4 版本受益更大。9.2 的融合 kernel 是 block-per-token，per-token max 需要
一次 warp shuffle + shared memory 跨 warp 归约 + 两次 `__syncthreads()`；
warp-per-token 之下一个 warp 就是一个 token，per-token max 退化成纯 warp
`__shfl_xor_sync` 归约，**shared memory 和 barrier 全部消失**，写回也从每线程
1 byte 提升到每 lane `ELEMS/2` 字节的一次 32-bit store。

接口按加法方式新增（`launch_hadamard_warp`、
`launch_hadamard_fused_quant_int4_warp`），`launch_hadamard`、
`launch_hadamard_optimized`、`launch_hadamard_fused_quant_int4` 与其数值语义
完全不变，因此 36/36 官方库验收结论继续有效，同时可以做严格逐位回归。
限制是 `head_dim >= 64`（否则一个 warp 装不下 2 元素/线程），d=32 仍走 9.1 路径。

### 4.7 Tensor Core 路径（本轮新增）

**先说不该做的事**：把 Hadamard 写成稠密 `H_d @ x` 交给 Tensor Core 是错的。
FHT 是 `O(d log d)`，稠密 GEMM 是 `O(d^2)`；d=256 时每 token 的算术量从 2048 次
加减涨到 65536 次乘加，涨 32 倍，而 Tensor Core 相对 FP32 SIMT 的峰值优势只有
一个量级左右，且 FHT 本身访存受限（逻辑算术强度仅 1.5–2.0 FLOP/B）。所以
Tensor Core 版本必须**换算法**，而不是换后端。

本实现用 Sylvester 矩阵的 Kronecker 递归，取 `b = 16`（正好是 WMMA
`m16n16k16` 的形状）：

```text
H_d = H_{d/16} (kron) H_16          (d >= 64)
```

把 token 内下标写成 `j = 16*rr + c`，则

```text
y[16*rr+c] = sum_{rr',c'} H_{d/16}[rr,rr'] * H_16[c,c'] * x[16*rr'+c']
         ==>  Y = H_{d/16} @ X @ H_16^T ,   H_16 对称
```

为了让每条 mma 都吃满 16x16x16，实现固定 “**一个 warp 处理 256 个连续元素
（一个 16x16 tile）**”，而不是固定一个 warp 一个 token：

| head_dim | 一个 tile 的内容 | 左乘常量矩阵 `A` |
|---:|---|---|
| 64 | 4 个 token，每 4 行一个 | `I_4 (kron) H_4` |
| 128 | 2 个 token，每 8 行一个 | `I_2 (kron) H_8` |
| 256 | 1 个 token，16 行 | `H_16` |
| 512 / 1024 | 一个 token 跨 2 / 4 个 tile | `H_16` |

左乘因子统一是块对角常量矩阵 `A = I_k (kron) H_m`（`m = min(d,256)/16`、
`k = 16/m`），于是**任何 head_dim 都归约成同一个两步计算**：

```text
P = X @ H_16     第 1 次 mma；X 是原始低精度输入，FP32 累加
Y = A @ P        第 2 次 mma
```

`d > 256` 时一个 token 跨 `d/256` 个 tile，对应
`H_d = H_{d/256} (kron) H_256`：先在每个 tile 内做上面两步，再对 accumulator
**逐元素**做 `log2(d/256)` 级蝶形。同一 fragment 类型的第 `e` 个元素在各 tile 中
对应相同的 `(row, col)`，所以这一级完全在寄存器里完成。

**精度是这条路径的真正难点**。WMMA 的 accumulator 是 FP32，但 operand 必须是
FP16/BF16。第 1 次 mma 直接使用原始输入，但不保证任意动态范围下精确；中间结果 `P` 必须
舍入回低精度才能喂给第 2 次 mma —— 这是 FHT 路径没有的额外舍入。因此提供两种
模式：

| 模式 | mma 次数 | 中间结果 | 效果 |
|---|---:|---|---|
| `tc_fast` | 2 | `P_hi = round(P)` | 最快，多一次低精度舍入 |
| `tc_split` | 3 | `P_hi = round(P)`，`P_lo = round(P - P_hi)` | 等效尾数 FP16 11→22 bit、BF16 8→16 bit |

`tc_split` 把 `A @ P` 拆成 `A @ P_hi + A @ P_lo` 累加到同一个 FP32 accumulator。
`A` 的元素只有 `0, ±1`，残差项能显著减小中间误差，但这不保证与 FP32 FHT
逐位一致。历史 token-peak `max_ulp = 0.500` 也不能证明逐元素正确舍入。

融合 INT4 侧：每个 lane 固定负责 8 个连续元素，因此 `d <= 256` 时一个 token 恰好
由连续的 `d/8` 个 lane 覆盖，per-token max 就是一次宽度为 `d/8` 的 sub-warp
`__shfl_xor_sync` 归约，无 shared memory、无 barrier。`d > 256` 时一个 token 跨
多个 tile，该归约不成立，接口显式返回不支持并由调用方回落到非 TC 融合 kernel ——
这是刻意的限制而不是未实现。

## 5. 正确性验证

### 5.1 环境

| 项目 | 实测值 |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU，SM 8.6，6 GiB |
| Driver | 592.00 |
| CUDA Toolkit / nvcc | 12.0 / 12.0.140 |
| PyTorch | 2.4.1+cu121 |
| 官方参考库 | fast_hadamard_transform 1.1.0 |
| 编译 | nvcc `-O3 -std=c++17 -gencode arch=compute_86,code=sm_86` |
| 随机种子 | 42 |

### 5.2 正式验收方法

每个配置由 C++ bench 生成目标低精度输入，运行本项目 kernel 并 dump 原始输入/输出；
Python 将输入严格 reshape 为 `[-1, head_dim]`，再调用官方库。双方使用相同 scale，
并在 FP32 域比较低精度输出。阈值为 FP16 `<1e-2`、BF16 `<5e-2`。

测试矩阵：

- dtype：FP16、BF16；
- head_dim：64、128、256；
- token 规模：1024、16384、131072；
- normalize：true、false；
- 合计：`2*3*3*2=36` 个配置。

### 5.3 结果

| dtype | 配置数 | max_abs_error 最大值 | bit-exact | 验收 |
|---|---:|---:|---:|---:|
| FP16 | 18 | 0 | 100% | 18/18 PASS |
| BF16 | 18 | 0 | 100% | 18/18 PASS |
| 合计 | 36 | 0 | 100% | 36/36 PASS |

完整逐配置记录见 `results/library_check.csv`。

开发期另对 CPU FP32 未舍入真值运行了 22 个配置。常规 normal/uniform 及 FP16 outlier
均过阈值；唯一 `BF16 + head_dim=128 + outlier(scale=20)` 的最大绝对误差为
`6.248e-2`。该项与官方 BF16 输出仍然 bit-exact，原因是输出落入 `[16,32)` 后 BF16
的 half-ULP 已达 `0.0625`，大于题目固定绝对阈值。这是输出格式舍入边界，不是 CUDA
计算错误。正式验收应以同 dtype 官方库结果为准。

### 5.4 优化与融合一致性

`hadamard_advanced_bench` 对每个 dtype/head_dim 同时执行三类逐位验证：

1. optimized FHT 与保留的 baseline 低精度输出完全一致；
2. fused INT4 的 packed bytes 和 FP32 scales 与 unfused pipeline 完全一致；
3. GPU unfused quantizer 与独立 CPU reference 的 packed bytes 和 scales 完全一致。

2026-09-04 L40S 全部 6 个核心性能配置均为 PASS；进一步覆盖 FP16/BF16、normalize
true/false 与 d=32/64/128/256/512/1024 的 24 个小规模配置也全部通过三项检查。
CPU reference 还计算反量化误差；normal 输入下 MAE 为 0.0912–0.1084，RMSE 为
0.1075–0.1263。bit-exact 用于验证实现语义，MAE/RMSE 用于描述 INT4 本身的有损
误差，两者不能相互替代。

### 5.5 warp / Tensor Core 路径的一致性检查

`tensor_core/src/tc_bench.cu` 对本轮新增的路径执行三条互补检查，它们同样不能
相互替代：

1. **全张量低精度外部参考的绝对误差** —— 对应题目的 FP16 `<1e-2`、BF16 `<5e-2`。
   历史 FP64 采样误差只作诊断；新实现不再使用相对误差绕过阈值；
2. **与 9.1 optimized FHT 输出的逐位一致率** —— 说明该实现是否严格等价于已经用
   官方库验收过的路径。`warp` 与 `tc_split` 分别达到 100% 与 99.94%–100%；
3. **融合 INT4 与“同算法先变换后量化”的逐位比较** —— 对应题目 “融合量化的结果需
   与先变换后量化的结果一致”。这条必须**按算法配对**：`fused_tc_fast` 与
   `tc_fast` 的输出比较，否则比较的是两种算法的数值差异，而不是融合是否正确。

历史 CSV 覆盖 10 个主配置、3 个 FP16 `normalize=false` 配置和 1 个尾部配置，
共 14 个唯一输入配置。旧尾部测试跳过了 TC 融合；`sweep_exit_code=0` 包含
相对误差放宽，不能当作 PDF 全面验收通过。新增严格测试见审计记录。

## 6. 性能结果与分析

### 6.1 RTX 3060 Laptop baseline

计时只包围 kernel，使用同一 CUDA stream 上的 Event；预热 5 次，正式测量 20 次。
下表采用 2026-09-02 最新全量复跑中的大规模 normal 配置：

| dtype | head_dim | tokens | avg ms | min ms | max ms | 有效全局带宽 GB/s | 算法吞吐 GOP/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP16 | 64 | 131072 | 0.3497 | 0.3461 | 0.3666 | 95.95 | 143.93 |
| FP16 | 128 | 131072 | 0.7613 | 0.7444 | 0.7916 | 88.15 | 154.26 |
| FP16 | 256 | 131072 | 1.6550 | 1.6333 | 1.7551 | 81.10 | 162.20 |
| BF16 | 64 | 131072 | 0.3475 | 0.3451 | 0.3562 | 96.55 | 144.83 |
| BF16 | 128 | 131072 | 0.7468 | 0.7432 | 0.7506 | 89.86 | 157.26 |
| BF16 | 256 | 131072 | 1.6723 | 1.6320 | 1.7295 | 80.26 | 160.52 |

“有效全局带宽”只按一次输入读取和一次输出写回计算，即
`tokens*d*4 bytes / time`，不包含 shared-memory 流量；“算法吞吐”按每层每元素一次
加/减，即 `tokens*d*log2(d) / time`。二者用于同一实现不同维度间比较，不等同于
硬件 DRAM 峰值或 Tensor Core FLOPS。

观察：

1. FP16 与 BF16 时间基本一致，说明当前瓶颈不在二者的转换差异；
2. `d` 从 64 翻倍到 128/256 时，时间约增加 2.17 倍，符合 `d log d` 工作量增长；
3. 有效全局带宽随维度从约 96 GB/s 降到 80 GB/s，说明同步、shared-memory 访问和
   每 block 线程数增长开始占更大比例；
4. 当前一轮蝶形一次 `__syncthreads()`，`d=256` 需要 8 次 block barrier，是下一步
   首要优化对象；
5. 尚未与官方库、warp-shuffle 版本或 Tensor Core 版本做同环境性能对比，因此不能
   声称当前实现优于这些方案。

完整结果见 `results/results.csv`；官方库逐配置结果见
`results/library_check.csv`。

### 6.2 L40S 9.1/9.2 A/B 结果

下表使用 131072 tokens、normal、normalize=true，预热 20 次、CUDA Event 正式测量
100 次。unfused 时间包含 optimized FHT 和独立 quantize 两个 kernel；fused 时间为一个
kernel，因此这是题目要求的端到端量化融合收益，而不只是量化子步骤时间。

| dtype | d | baseline us | optimized us | FHT speedup | unfused INT4 us | fused INT4 us | fusion speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP16 | 64 | 73.43 | 21.50 | 3.42x | 92.68 | 73.34 | 1.26x |
| FP16 | 128 | 109.37 | 38.62 | 2.83x | 110.50 | 73.41 | 1.51x |
| FP16 | 256 | 240.75 | 193.64 | 1.24x | 240.77 | 104.47 | 2.30x |
| BF16 | 64 | 73.55 | 21.51 | 3.42x | 92.79 | 73.43 | 1.26x |
| BF16 | 128 | 109.53 | 38.91 | 2.82x | 110.56 | 73.54 | 1.50x |
| BF16 | 256 | 240.99 | 192.82 | 1.25x | 240.69 | 103.40 | 2.33x |

INT4 加上每 token 一个 FP32 scale 后，输出相对 FP16/BF16 中间张量压缩
3.56x/3.76x/3.88x（d=64/128/256）。d=64/128 优化幅度最大，吻合“减少 warp 内
barrier + 多 token/block”的设计目标。d=256 仍需两轮跨 warp shared-memory 交换，
因此优化只有约 1.25x。另一方面，d 越大，中间张量 write/read 在 unfused pipeline
中的占比越高，所以融合收益从约 1.26x 增长到约 2.33x。

d=64/128 的输入+输出工作集分别约 32/64 MiB，小于 L40S 的 96 MiB L2；重复 warmup
后按逻辑最小 I/O 算出的“有效带宽”可超过 864 GB/s HBM 理论值。这是 warm-cache/L2
复用，不是超越显存物理上限。因此正式报告保留 baseline 的 HBM Roofline，同时不把
优化版本的逻辑 I/O 冒充 NCU 实测 DRAM bytes。

完整产物与可视化见 `results/optimization_16970010/`。

![9.1/9.2 优化与融合性能](../results/optimization_16970010/figures/advanced_performance.png)

### 6.3 warp-per-token 与 Tensor Core 的同环境 A/B（2026-09-06，L40S，job `17083405`）

本轮 A/B 由 `tensor_core/src/tc_bench.cu` 在**同一进程、同一 stream、同一份输入**
上依次评测五条变换路径和五条融合路径，因此各行可以直接相减。规模仍为
131072 tokens、normalize=true，预热 20 次、CUDA Event 正式测量 100 次。

变换 kernel（avg us）：

| dtype | d | baseline | optimized (9.1) | warp | tc_fast | tc_split | warp/optimized | 最快 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| FP16 | 64 | 73.43 | 21.31 | 21.41 | **18.38** | 21.73 | 1.00x | tc_fast |
| FP16 | 128 | 109.80 | 38.73 | **23.01** | 32.13 | 38.71 | 1.68x | warp |
| FP16 | 256 | 240.66 | 193.00 | **185.31** | 190.17 | 190.98 | 1.04x | warp |
| FP16 | 512 | 535.94 | 408.04 | **398.28** | 411.84 | 412.89 | 1.02x | warp |
| FP16 | 1024 | 1692.54 | 819.10 | 814.22 | 816.45 | **807.51** | 1.01x | tc_split |
| BF16 | 64 | 73.36 | 21.38 | 21.40 | **19.44** | 23.27 | 1.00x | tc_fast |
| BF16 | 128 | 109.55 | 38.72 | **23.02** | 34.42 | 42.05 | 1.68x | warp |
| BF16 | 256 | 240.89 | 192.91 | **185.18** | 190.17 | 191.23 | 1.04x | warp |
| BF16 | 512 | 536.53 | 407.62 | **398.46** | 410.94 | 412.46 | 1.02x | warp |
| BF16 | 1024 | 1699.60 | 817.35 | 813.50 | 814.51 | **805.48** | 1.00x | tc_split |

融合 INT4 端到端（avg us；`unfused` = warp FHT + 独立 quantize 两个 kernel）：

| dtype | d | unfused | fused_opt (9.2) | fused_warp | fused_tc_fast | fused_tc_split | fused_warp / 9.2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP16 | 64 | 92.81 | 73.38 | 22.09 | **18.58** | 21.70 | 3.32x |
| FP16 | 128 | 94.37 | 73.55 | **28.82** | 32.85 | 39.39 | 2.55x |
| FP16 | 256 | 237.67 | 103.85 | **48.82** | 62.01 | 75.83 | 2.13x |
| FP16 | 512 | 621.32 | 251.85 | **235.50** | n/a | n/a | 1.07x |
| FP16 | 1024 | 1276.55 | 573.32 | **469.59** | n/a | n/a | 1.22x |
| BF16 | 64 | 92.99 | 73.41 | 21.99 | **19.88** | 23.61 | 3.34x |
| BF16 | 128 | 94.17 | 73.66 | **28.68** | 35.16 | 42.89 | 2.57x |
| BF16 | 256 | 237.65 | 103.45 | **48.32** | 66.58 | 86.24 | 2.14x |
| BF16 | 512 | 621.11 | 251.41 | **235.25** | n/a | n/a | 1.07x |
| BF16 | 1024 | 1277.13 | 556.81 | **468.59** | n/a | n/a | 1.19x |

`fused_opt` 一列是已验收的 9.2 融合 kernel，在本轮复现出 73.38 / 73.55 /
103.85 us，与 §6.2 的 73.34 / 73.41 / 104.47 us 相差 0.1%–0.6%。这条锚点说明
本轮数字与历史数字可比，差异不是环境漂移。

三点结论：

1. **warp-per-token 兑现了 §9.4 里点出的 d=128/256 跨 warp 同步问题**。d=128
   去掉 1 轮 shared 交换和 2 次 barrier，直接换来 1.68x；d=64 本来就没有 barrier
   所以持平；d=256 只有 1.04x，因为该维度已经撞到访存上限（见第 3 点）。
2. **融合路径的收益比变换路径大得多**。9.2 的融合 kernel 是 block-per-token +
   shared 归约 + 2 次 barrier，warp-per-token 把它全部去掉，于是 d=64/128/256
   分别快 3.32x / 2.55x / 2.13x。这也说明 9.2 时期 “fused 只比 unfused 快
   1.26x–2.33x” 的瓶颈并不在融合思路，而在 per-token 归约的实现方式。
3. **d >= 256 已经访存受限，任何计算侧优化都改变不了时间**。d=256 的最小逻辑
   I/O 是 131072 x 256 x 2 B x 2 = 134 MB，按同节点 864 GB/s 的理论上限也需要
   155 us，实测 185 us（达到 roof 的 84%）。所以 d=256/512/1024 上五条路径全部
   收敛到 1%–3% 以内，`tc_split` 在 d=1024 上那 1% 的领先属于噪声量级。

### 6.4 Tensor Core 精度

前 4096 个 token 与 FP64 CPU 参考（同 stage 顺序、全程 double）对比。
`max_ulp` 是以 “该 token 峰值处输出 dtype 的 ULP” 为单位的误差；不是每个元素
自己的 ULP，`<= 0.5` 不能证明正确舍入。此表为 FP64 采样诊断，不是 PDF 验收。

| dtype | d | 实现 | max_abs_error | max_ulp | 与 optimized 逐位一致率 |
|---|---:|---|---:|---:|---:|
| FP16 | 64 | warp | 1.901e-03 | 0.500 | 100.00% |
| FP16 | 64 | tc_fast | 2.138e-03 | 1.244 | 59.80% |
| FP16 | 64 | tc_split | 1.901e-03 | 0.500 | 100.00% |
| FP16 | 128 | warp | 1.813e-03 | 0.500 | 100.00% |
| FP16 | 128 | tc_fast | 2.119e-03 | 0.995 | 58.68% |
| FP16 | 128 | tc_split | 1.813e-03 | 0.500 | 100.00% |
| FP16 | 256 | warp | 1.762e-03 | 0.500 | 100.00% |
| FP16 | 256 | tc_fast | 2.249e-03 | 0.924 | 57.99% |
| FP16 | 256 | tc_split | 1.762e-03 | 0.500 | 100.00% |
| FP16 | 1024 | tc_split | 1.949e-03 | 0.500 | 99.99% |
| BF16 | 64 | warp | 1.495e-02 | 0.500 | 100.00% |
| BF16 | 64 | tc_fast | 1.935e-02 | 1.051 | 59.76% |
| BF16 | 64 | tc_split | 1.495e-02 | 0.500 | 99.97% |
| BF16 | 128 | tc_fast | 1.669e-02 | 1.013 | 58.67% |
| BF16 | 128 | tc_split | 1.553e-02 | 0.500 | 99.97% |
| BF16 | 256 | tc_fast | 2.066e-02 | 0.962 | 57.97% |
| BF16 | 256 | tc_split | 1.559e-02 | 0.500 | 99.94% |

- `tc_split` 在 FP16 d=64/128/256 的这组样本中 100% 逐位一致，BF16 和更大维度
  存在不一致；不能宣称普遍等价。
- `tc_fast` 的采样 FP64 误差不替代全张量低精度参考的绝对误差验收。
- 历史 14 个配置中已执行的融合检查逐位一致；旧尾部配置跳过了 TC 融合。

PDF 要求固定绝对误差，不允许用相对误差替代。与未舍入 FP64 参考比较时包含输出
dtype 自身的舍入误差，而与低精度参考库输出比较时，两者可以完全同值。
旧实现将这两种参考混淆并使用 `PASS(rel)`；2026-09-08 已改为全张量低精度参考
的严格绝对误差检查，FP64/相对/token-peak ULP 仅作诊断。

![Tensor Core 与非 TC 对比](../tensor_core/results/tc_17083405/figures/tc_performance.png)

## 7. nsys / ncu 使用与实测分析

### 7.1 一键入口

```bash
bash scripts/profile.sh nsys
bash scripts/profile.sh ncu
```

代表配置为 FP16、`head_dim=128`、16384 tokens。原始产物保存在 `results/`。

### 7.2 Nsight Systems

本机 Nsight Systems 2022.4.2 首先只产生 `.qdstrm`；使用同版本
`QdstrmImporter` 后已生成：

- `results/nsys_fp16_hd128_baseline.nsys-rep`
- `results/nsys_fp16_hd128_baseline.sqlite`

CUDA API 汇总观察到 26 次 `cudaLaunchKernel`，与 1 次启动校验、5 次 warmup、20 次
计时迭代完全一致；同时有 40 次 `cudaEventRecord` 和 20 次
`cudaEventSynchronize`，符合逐迭代计时逻辑。最近一次开启 nsys 后该配置 CUDA
Event 平均时间为 0.1047 ms，与无 profiler 的最近 sweep 0.1047 ms 一致；仍应以
无 profiler 的重复测量作为正式性能口径。

这套 2022.4 profiler 在当前 592 驱动上没有导出 GPU kernel 明细，`gpukernsum` 明确
报告 `does not contain CUDA kernel data`。因此本报告不据此虚构 SM 利用率、占用率
或 kernel overlap；应升级 nsys 后重采 GPU timeline。`.qdstrm` 转 `.nsys-rep` 必须
使用相同版本 importer，这是 NVIDIA 官方文档明确要求的兼容规则。

### 7.3 Nsight Compute

本机 ncu 2022.4.1 能启动目标程序，但硬件计数器被系统策略禁止，报错：

```text
ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters
```

因此当前没有可用的 occupancy、DRAM throughput、shared bank conflict 或 barrier stall
数值。管理员可按 NVIDIA 官方说明使用具备 `CAP_SYS_ADMIN` 的用户运行，或设置
`NVreg_RestrictProfilingToAdminUsers=0` 后重启/重载驱动。权限开放后重点采集：

- achieved occupancy 与 registers/thread；
- DRAM read/write throughput；
- shared load/store throughput 与 bank conflicts；
- barrier stall、long scoreboard stall；
- FP32 pipe utilization。

预期判断：若 barrier stall 高，优先将 warp 内阶段改为 shuffle；若 DRAM 吞吐低且
load/store transaction 不理想，优先 half2/bfloat162 向量化和多 token/block；若 shared
bank conflict 高，再调整共享内存布局或改寄存器交换。

### 7.4 2026-09-04 集群补采（L40S）

为补齐旧版 profiler 没有 kernel timeline 的缺口，使用 Slurm job `16967596` 在
NVIDIA L40S（SM 8.9）上重编译并采集。环境为 driver 580.82.07、CUDA 13.0.88、
Nsight Systems 2025.3.2、Nsight Compute 2025.3.1。以下均为 131072 tokens、normal、
normalize=true；无 profiler 计时使用 20 次 warmup 和 100 次正式迭代：

| dtype | head_dim | avg us | min us | max us | 有效逻辑 I/O GB/s | 算法吞吐 GFLOP/s |
|---|---:|---:|---:|---:|---:|---:|
| FP16 | 64 | 73.93 | 73.06 | 83.07 | 453.86 | 680.78 |
| FP16 | 128 | 110.01 | 109.15 | 112.77 | 610.02 | 1067.53 |
| FP16 | 256 | 241.17 | 240.38 | 244.32 | 556.53 | 1113.05 |
| BF16 | 64 | 73.93 | 73.18 | 79.78 | 453.89 | 680.83 |
| BF16 | 128 | 110.97 | 109.66 | 115.84 | 604.78 | 1058.36 |
| BF16 | 256 | 241.12 | 240.32 | 245.76 | 556.65 | 1113.29 |

新版 Nsight Systems 已成功记录 GPU kernel 明细。代表配置 FP16/d=128 的 trace 含
56 次 `hadamard_kernel`，严格对应 1 次分发校验、5 次 warmup 和 50 次计时迭代；
kernel 平均 108.05 us，中位数 108.06 us，范围 107.62–108.96 us。launch 形状为
grid `131072x1x1`、block `128x1x1`，每线程 18 registers，静态 shared memory 512 B。
按 Systems 导出的设备资源上限估算，该配置的 resource-limit occupancy 为 100%；
这是 launch/resource 推导值，不是硬件计数器测得的 achieved occupancy。

CUDA Runtime 侧共记录 56 次 `cudaLaunchKernel`，中位调用开销 4.31 us；首次 JIT/module
路径把总体平均抬到 8.87 us。50 次 `cudaEventSynchronize` 平均 110.15 us，与 kernel
timeline 和无 profiler CUDA Event 计时互相吻合。

Roofline 使用同节点导出的 91.61 TFLOP/s FP32 上限与 864.10 GB/s 显存带宽上限。
当前 FHT 的逻辑算术强度为 `log2(d)/4`，即 1.5/1.75/2.0 FLOP/B，远低于约
106 FLOP/B 的 ridge point，因此三个维度在算法层面均处于 memory-side。以一次 16-bit
输入读取和一次 16-bit 输出写回的逻辑最小 I/O 计，达到该 roof 的 52.5%–70.6%。图见
`results/profile_16967596/figures/roofline.png` 和
`results/profile_16967596/figures/profiler_metrics.png`；kernel 逐次稳定性和 CUDA API
开销另见 `results/profile_16967596/figures/nsys_timeline.png`。

本轮 `ncu --set full` 在 6 个配置上均已实际尝试，但节点返回
`Profiling failed because a driver resource was unavailable`，提示 DCGM/另一采集器占用
performance counter。故本报告仍不填造 achieved occupancy、真实 DRAM bytes、bank
conflict 或 barrier stall；当前图的横轴明确使用逻辑最小 I/O，而非冒充 NCU 实测 DRAM
流量。脚本已保留原始错误日志，并增加 `--clock-control none` 和跨节点重试入口。

![L40S baseline 逻辑 I/O Roofline](../results/profile_16967596/figures/roofline.png)

### 7.5 优化/融合 Nsight trace

Slurm job `16970010` 在另一块 L40S 上成功记录 FP16/d=128、16384 tokens 的四路径
Systems trace。汇总如下：

| kernel | calls | avg us | grid | block | registers/thread | static shared | resource-limit occupancy |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline FHT | 26 | 14.86 | 16384 | 128 | 18 | 512 B | 100% |
| optimized FHT | 52 | 5.83 | 8192 | 128 | 20 | 1024 B | 100% |
| standalone INT4 | 26 | 9.91 | 16384 | 64 | 18 | 12 B | 100% |
| fused FHT+INT4 | 26 | 10.27 | 16384 | 64 | 18 | 524 B | 100% |

optimized 有 52 次是因为 standalone 测试和 unfused pipeline 都调用它；其余 26 次对应
1 次验证、5 次 warmup、20 次测量。该 trace 证明 kernel 形状和相对结构，最终性能仍
采用无 profiler 的 131072-token Event 结果。resource-limit occupancy 是由 block、
寄存器和 shared-memory 上限推导，不等于 achieved occupancy。

同一 job 对 optimized 和 fused 各执行一次 NCU full-set 采集，仍返回
`driver resource was unavailable`，明确指向 DCGM/其他采集器占用。日志见
`results/optimization_16970010/ncu/`；报告不据此猜测硬件计数器数值。

### 7.6 H200 NCU 硬件计数器（已闭环）

跨节点重试中，L40S `gl021` 与 A100 `ga024` 仍被 DCGM 占用，但 H200 `gh112`
（job `16975372`）成功完成 optimized 与 fused 各 18 passes。配置为 FP16/d=128、
16384 tokens，采集 SpeedOfLight、MemoryWorkloadAnalysis、Occupancy、WarpStateStats：

| kernel | NCU duration us | SM throughput | DRAM throughput | DRAM GB/s | achieved occupancy | L2 hit | registers/thread | static shared |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| optimized FHT | 7.552 | 34.73% | 11.63% | 555.90 | 58.39% | 53.99% | 19 | 1024 B |
| fused FHT+INT4 | 12.896 | 34.12% | 6.79% | 325.82 | 46.69% | 34.83% | 20 | 524 B |

optimized 的 SM throughput 明显高于 DRAM throughput，selected stall 中 long scoreboard
最高（7.73），barrier 为 1.55；说明 warp shuffle 已降低 barrier 影响，剩余主要问题是
数据依赖/内存等待而非 HBM 峰值带宽饱和。fused 的 long scoreboard 降至 4.99，但
token-wide max/scale/packing 增加依赖，使 achieved occupancy 从 58.39% 降为 46.69%。

以 NCU bandwidth×duration 积分，两次 capture 的实际 DRAM 流量均约 4.20 MB，低于
逻辑张量流量；不能据此归因于 validation 预热，因为 NCU kernel replay 默认刷新
缓存。需要分别检查 DRAM read/write、L2 流量和缓存控制策略。这些 H200 计数器
不与 L40S 性能混算。完整
`.ncu-rep`、raw CSV、精简表和图在 `results/ncu_retry_16975372/`。

![H200 NCU 硬件指标](../results/ncu_retry_16975372/ncu_hardware_metrics.png)

### 7.7 warp / Tensor Core 的 Nsight trace（job `17083405`）

同一条 Systems trace（FP16、d=128、16384 tokens）同时覆盖本轮全部十条路径。
launch 结构由 `scripts/summarize_nsys.py` 从 sqlite 导出：

| kernel | calls | avg us | grid | block | registers/thread | shared/block | resource-limit occupancy |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline FHT | 26 | 14.831 | 16384 | 128 | 18 | 512 B static | 100% |
| optimized FHT (9.1) | 26 | 5.827 | 8192 | 128 | 20 | 1024 B static | 100% |
| **warp FHT** | 52 | **3.743** | 4096 | 128 | 21 | **0** | 100% |
| tc_fast | 26 | 4.931 | 2048 | 128 | 34 | 9728 B dynamic | 83.3% |
| tc_split | 26 | 5.834 | 2048 | 128 | 31 | 12800 B dynamic | 66.7% |
| standalone INT4 quant | 29 | 9.906 | 16384 | 64 | 18 | 12 B static | 100% |
| fused INT4 (9.2) | 26 | 10.225 | 16384 | 64 | 18 | 524 B static | 100% |
| tc_fast + INT4 | 26 | 5.191 | 2048 | 128 | 30 | 9728 B dynamic | 83.3% |
| tc_split + INT4 | 26 | 6.210 | 2048 | 128 | 30 | 12800 B dynamic | 66.7% |
| **warp FHT + INT4** | 26 | **4.762** | 4096 | 128 | 19 | **0** | 100% |

warp FHT 是 52 次，因为独立测试和 unfused pipeline 都会调用它；其余 26 次对应
1 次校验 + 5 次 warmup + 20 次计时。CUDA Runtime 侧记录 263 次
`cudaLaunchKernel` 和 180 次 `cudaEventSynchronize`，与 benchmark 的十条路径
逐条对应。resource-limit occupancy 由 block/寄存器/shared-memory 上限推导，
不是硬件计数器实测 achieved occupancy。

**这张表直接解释了 Tensor Core 为什么只在 d=64 赢。** WMMA 的代价不在 mma 本身，
而在 accumulator 与 matrix_b 的 fragment 内部布局不同，第 1 次 mma 的结果必须经
shared memory 中转。这块 scratch 让每个 block 占用 9.7–12.8 KB shared memory，
把 resource-limit occupancy 从 100% 压到 83.3%/66.7%，同时多出一次 shared 写、
一次 shared 读和两次 `__syncwarp()`；寄存器也从 20–21 涨到 30–34。而 warp FHT
的蝶形全在寄存器和 shuffle 里，shared memory 用量是 0。

于是收益窗口很清楚：d=64 的 FHT 需要 6 级蝶形，两条 mma 一次替掉全部 6 级、
并且一个 warp 顺带处理 4 个 token，Tensor Core 净赚；d=128/256 时 FHT 只增加到
7/8 级（仍然是 5 级 shuffle + 2/3 级寄存器），mma 能省下的绝对时间几乎不变，
但 shared 中转成本不变甚至更高。于是变换路径上 TC 在 d=128 落后 1.40x、
d=256 落后 1.03x（该维度已接近访存上限，差异被压平）；融合路径上分别落后
1.14x 与 1.27x（融合 kernel 的计算占比更高，shared 中转的相对代价也更大）。

L40S 分区对本轮三个 kernel 的 NCU 采集仍返回
`Profiling failed because a driver resource was unavailable`（DCGM 占用
performance counter），原始日志保留在
`tensor_core/results/tc_17083405/ncu/`。硬件计数器因此改在 Hopper 节点单独采集，
见下节。

### 7.8 warp / Tensor Core 的 H200 硬件计数器（job `17083350`，已闭环）

H200 `gh115`（driver 580.82.07、CUDA 13.0.88、Nsight Compute 2025.3.1）对
warp / tc_fast / tc_split 三个 kernel 全部采集成功（`ncu_exit_status=0`）。
配置为 FP16、`head_dim=128`、16384 tokens，section 为 SpeedOfLight、
MemoryWorkloadAnalysis、Occupancy、WarpStateStats、SchedulerStats：

| 指标 | warp | tc_fast | tc_split |
|---|---:|---:|---:|
| NCU duration us | 4.992 | 7.712 | 9.248 |
| SM throughput % | 33.88 | 21.65 | 21.99 |
| DRAM throughput % | 17.58 | 11.36 | 9.47 |
| DRAM GB/s | 841.1 | 544.5 | 454.2 |
| achieved occupancy % | 65.25 | 72.01 | 65.67 |
| L2 hit % | 53.44 | 53.46 | 53.73 |
| registers/thread | 21 | 30 | 32 |
| dynamic shared KB/block | 0.000 | 9.728 | 12.800 |
| stall long scoreboard | 10.64 | 4.10 | 2.97 |
| stall short scoreboard | 1.90 | **12.94** | **11.07** |
| stall mio throttle | 1.34 | **10.52** | **12.82** |
| stall barrier | **0.00** | 1.06 | 1.32 |

**stall 构成给出了 §7.7 那个解释的直接硬件证据**：

1. **warp FHT 的 `stall barrier` 精确为 0**，与 “零 shared memory、零
   `__syncthreads()`” 的设计一一对应；它的主导 stall 是 `long scoreboard`
   （10.64，global memory 延迟），也就是说这个 kernel 已经在等显存 —— 这正是一个
   访存受限算子该有的样子。它把 DRAM 推到 841 GB/s，SM throughput 33.9%。
2. **两条 Tensor Core kernel 的主导 stall 完全换了一批**：`short scoreboard`
   （12.94 / 11.07）和 `mio throttle`（10.52 / 12.82）都是 shared memory / MIO
   通路的压力指标，而 `long scoreboard` 反而降到 4.10 / 2.97。也就是说 TC 版本
   并没有在等显存，而是在等自己那次 accumulator → matrix_b 的 shared memory
   中转，DRAM 只推到 544.5 / 454.2 GB/s，SM throughput 掉到约 21.7%。
3. `tc_split` 多的那次 mma 让 shared 压力进一步上升（`mio throttle` 从 10.52 升到
   12.82，`dynamic shared` 从 9.7 KB 升到 12.8 KB），寄存器也从 30 涨到 32，
   duration 相应从 7.71 增到 9.25 us —— 精度换性能的代价在计数器上是可见的。

注意 achieved occupancy 一列不能单独读：`tc_fast` 的 72.01% 反而高于 warp 的
65.25%，但它的 block 数只有 warp 的一半（2048 vs 4096），单位工作量下的有效
并行度更低；真正解释时间差的是 stall 构成和 DRAM 吞吐，不是 occupancy 数字本身。

口径与 §7.6 一致：**计数器采自 H200，端到端时间采自 L40S，两组数字不混算**；
也不用 NCU 的 duration 替代无 profiler 的 CUDA Event 计时（NCU 会串行化 kernel
并引入采集开销，其 warp:tc_fast:tc_split = 1 : 1.55 : 1.85 与 L40S nsys 的
1 : 1.32 : 1.56 排序一致但比例更陡）。完整 `.ncu-rep`、610 列 raw CSV 和精简表见
[`tensor_core/results/tc_ncu_17083350/`](../tensor_core/results/tc_ncu_17083350/README.md)。

## 8. 优化历程与开发中发现的问题

### 8.1 数据驱动的迭代记录

| 阶段 | 观测/问题 | 修改 | 验证结果 |
|---|---|---|---|
| 教学 baseline | 每轮都经过 shared memory 和 `__syncthreads()` | 保留为正确性锚点 | 官方库 36/36 bit-exact |
| L40S baseline profile | d=128 kernel 108.05 us；逻辑 AI 仅 1.75 FLOP/B | 优先处理同步与数据搬运 | 排除 Tensor Core 作为第一优先级 |
| warp 内蝶形 | d=64/128 大部分 stage 不需要跨 warp | stride 1 寄存器化，2–32 shuffle | d=64/128 加速 3.42x/2.83x |
| packed I/O | 相邻元素天然成对参与 stride 1 | `half2`/`bfloat162` 合并读写 | FP16/BF16 保持 bit-exact |
| 多 token/block | d=64/128 单 token block warp 数偏少 | 每 block 4/2 tokens | grid 分别缩小 4x/2x |
| d=256 回归 | 仍有 stride 64/128 跨 warp 同步 | 保留 shared fallback | 稳定加速约 1.25x，明确后续热点 |
| unfused INT4 | 中间低精度张量需写回再读入 | FHT、max、scale、pack 单 kernel 融合 | d=256 端到端加速 2.30x–2.33x |
| fused 一致性 | 直接量化 FP32 寄存器会偏离 unfused dtype 舍入 | 寄存器内模拟 FP16/BF16 舍入 | fused/unfused packed bytes/scales bit-exact |
| warp-per-token | 每线程 2 元素导致 d>=128 必须跨 warp 走 shared+barrier，且跨 warp 阶段半数线程空转 | 每线程 `d/32` 元素、一 warp 一 token，蝶形只用寄存器+5 次 shuffle | d=128 变换再快 1.68x；shared/barrier 归零；与 9.1 100% 逐位一致 |
| warp 融合归约 | 9.2 融合 kernel 的 per-token max 用 shared + 2 次 barrier | 一 warp 一 token 后改为纯 warp shuffle 归约，写回升到 32-bit store | 融合 INT4 相对 9.2 快 2.13x–3.34x |
| Tensor Core 选型 | 稠密 `H_d @ x` 会把 `O(d log d)` 退化成 `O(d^2)` | 改用 `H_d = H_{d/16} ⊗ H_16`，一个 16x16 tile 两条 mma | d=64 变换 1.16x、融合 1.19x 优于最好的非 TC 实现 |
| Tensor Core 精度 | 中间结果必须舍入回低精度才能喂第 2 次 mma | 拆成 `P_hi + P_lo` 三次 mma 累加到同一 FP32 accumulator | `max_ulp` 从 ~1.0 降到 0.500，FP16 d<=256 100% 逐位一致 |
| Tensor Core 上限 | d>=128 时 TC 反而更慢 | 由 nsys launch 结构定位到 shared scratch 压 occupancy 到 83%/67% | 明确 TC 收益窗口只在小 d；d>=256 已访存受限 |

### 8.2 开发中发现的问题与处理

1. **先锁定数学语义**：官方库默认 `scale=1.0`，项目默认正交归一化。测试显式传
   scale，同时覆盖两种语义，避免“数值都像对的但差一个 `sqrt(d)`”。
2. **修正最后一维语义**：官方库按最后一维变换；一维 dump 必须 reshape 为
   `[-1, head_dim]`，否则会把全部 token 当一行。
3. **统一累加精度和顺序**：低精度输入转 FP32、同序蝶形、末端一次舍入，使本实现
   与官方库达到 bit-exact。
4. **区分实现误差和格式误差**：对未舍入高精度参考的误差包含输出格式的舍入误差；
   相对误差仅作诊断，验收仍严格比较同 dtype 参考输出的绝对误差。
5. **Profiler 环境也是实验条件**：ncu 权限和 nsys/driver 版本会决定能否得到硬件指标；
   当前报告只写实际拿到的数据和明确失败原因。
6. **由 trace 指向同步优化**：baseline 每轮 shared-memory 往返和 barrier 是明确热点结构；
   将 warp 内阶段改成寄存器/shuffle 后，d=64/128 加速 2.8x–3.4x。
7. **融合价值随中间张量增长**：d=256 的 FHT 单独优化较小，但消除中间 write/read 后
   fused pipeline 仍达到 2.3x，说明“单 kernel 更快”和“端到端更省流量”要分开衡量。
8. **缓存会改变 Roofline 解释**：d=64/128 的重复工作集可落入 L40S 96 MiB L2，按
   逻辑 I/O 计算的带宽会超过 HBM roof。报告因此不把 warm-cache 逻辑带宽称为 DRAM
   带宽，并保留 NCU 实际 bytes 缺失标记。
9. **集群 profiler 是共享资源**：NCU 能 attach 并定位到目标 kernel，但 PM counter 在
   replay 开始时返回 `driver resource was unavailable`；这与代码编译、权限位或 kernel
   correctness 无关，不能通过修改 kernel“修复”。
10. **Tensor Core 不是“更快的后端”，而是另一种算法**：直接把 FHT 换成稠密
    `H_d @ x` 会把算术量涨 32 倍（d=256），远超 Tensor Core 的峰值优势。必须先用
    Kronecker 分解把变换重排成 16x16 的形状，才谈得上用 WMMA。题目 Note 里
    “实现算法不一定相同” 说的就是这件事。
11. **融合正确性检查必须按算法配对**：一开始把 `fused_tc_fast` 的 packed bytes 直接
    与 “warp FHT -> quantize” 的参考比较，结果必然 MISMATCH —— 那比较的是两种算法的
    数值差异，而不是融合本身是否正确。改成每条融合路径与**同算法**的
    “先变换后量化” 比较之后，10 条路径全部 bit-exact。
12. **不能通过改变参考对象和容差掩盖失败**：旧 TC benchmark 对 FP64 的舍入误差
    使用混合绝对/相对容差，这是诊断而非 PDF 验收。2026-09-08 已撤回该口径，
    改为相同输入、dtype、scale 下的全张量低精度参考绝对误差。失败保留输入和记录，
    token-peak ULP 不用于证明正确舍入。

## 9. 文档任务执行状态

### 9.1 baseline kernel 优化（已完成）

1. 已用 `__shfl_xor_sync` 完成 stride=2–32，stride=1 留在两个寄存器内；
2. 已用 `half2`/`__nv_bfloat162` 向量化 global load/store；
3. d=64/128 已采用每 block 4/2 tokens；
4. 已建立 baseline/optimized 同输入同边界 A/B，6/6 bit-exact；
5. L40S FHT 实测加速 1.24x–3.42x；Systems trace 已取得，NCU 计数器仍被环境占用；
6. 本轮进一步给出 warp-per-token 版本（§4.6），d=128 在 9.1 之上再快 1.68x，
   shared memory 与 barrier 归零，且与 9.1 输出 100% 逐位一致。

### 9.2 融合量化（已完成 INT4）

1. 已固定 per-token symmetric INT4 的 scale、round-to-nearest-even、[-7,7] clamp 和
   low/high nibble packing；
2. 已实现 CPU reference quantizer 和 unfused GPU pipeline；
3. 已在 FHT 写回点融合 max-reduction、scale、quantize 和 packing；
4. FP16/BF16 × d=64/128/256 的 fused/unfused 与 GPU/CPU 均 bit-exact；
5. 已报告端到端时间、3.56x–3.88x 压缩率和量化误差；融合加速 1.26x–2.33x；
6. 本轮的 warp-per-token 融合版本把 per-token max 改成纯 warp shuffle 归约，
   相对 9.2 融合 kernel 在 d=64/128/256 上再快 2.13x–3.34x（d=512/1024 为
   1.07x–1.22x，此时已访存受限），量化协议与逐位一致性检查均不变（§6.3）。

FP8 尚未实现。它是另一套输出格式与标度协议，不应与已验收的 INT4 数据混写。

### 9.3 Tensor Core 探索（已完成）

代码在 [`tensor_core/`](../tensor_core/README.md)，算法与精度设计见 §4.7，
实测见 §6.3–§6.4 与 §7.7。落地情况：

1. **没有**用稠密 `H_d @ x` 替代 FHT。改用 `H_d = H_{d/16} ⊗ H_16`，每个 warp
   处理一个 16x16 tile，两条 `m16n16k16` mma 完成整段变换；d=512/1024 再对
   accumulator 做寄存器内跨 tile 蝶形。每 token 乘加次数为 `2d`（d=256 即 8192），
   是稠密 GEMM 的 1/8。
2. 提供 `tc_fast`（2 次 mma）与 `tc_split`（hi/lo 3 次 mma）两种精度模式，并实现了
   Tensor Core 版的融合 INT4（`d <= 256`，`d > 256` 显式回落）。
3. 题目要求的四项对比全部给出，且都在同一进程/stream/输入上测：
   - **数值误差**：见 §6.4。历史 FP64 采样和 token-peak ULP 是诊断；
     `tc_split` 不保证普遍逐位一致，PDF 合规以全张量严格绝对误差验收为准。
   - **单独 Hadamard kernel 时间**：见 §6.3 上表。d=64 时 `tc_fast` 18.38 us，
     优于最好的非 TC 实现 21.31 us（1.16x）；d>=128 落后。
   - **与量化融合后的端到端时间**：见 §6.3 下表。d=64 时
     `fused_tc_fast_int4` 18.58 us，优于 `fused_warp_int4` 22.09 us（1.19x），
     相对已验收的 9.2 融合 kernel（73.38 us）快 3.95x。
   - **相对优化后 shuffle baseline 的加速比**：这一点尤其重要。本轮特意先把非 TC
     实现做到 warp-per-token（§4.6），再拿 Tensor Core 与它比，而不是与教学
     baseline 比。若只对 baseline 比较，`tc_fast` 在 d=64 上会显示 4.0x，
     那是被弱基线放大的数字；对最好的非 TC 实现只有 1.16x。
4. **结论**：Hadamard 变换是访存受限算子，Tensor Core 的收益窗口只在 “蝶形级数
   相对访存量足够多” 的小 d 区间。d=64 上确实赢，d=128/256 因为 WMMA 需要一次
   shared memory 中转（occupancy 100% → 83%/67%）而落后，d>=256 所有实现都撞到
   HBM 上限并收敛。这个结论有 nsys launch 结构（§7.7）和 roofline 估算（§6.3
   第 3 点）两条独立证据支持。
5. 仍未做的方向：把旋转吸收进相邻 GEMM（需要与真实 attention/MLP 算子集成，
   超出本选题的独立 kernel 范围），以及用 `mma.sync` PTX 直接控制 fragment 布局
   以绕开 accumulator→operand 的 shared memory 中转。后者是 §9.4 的第一优先级。

### 9.4 可继续提升的方向

1. ~~**d=256 跨 warp 数据交换**~~：已由 §4.6 的 warp-per-token 映射解决 ——
   每线程 `d/32` 元素后 shared memory 与 barrier 归零。剩下的空间是
   **绕开 WMMA 的 shared memory 中转**：用 `mma.sync` PTX 直接按已知 fragment
   布局把第 1 次 mma 的 accumulator 重排成第 2 次的 operand，可望把 TC 路径的
   occupancy 拉回 100%，从而把 TC 的收益窗口从 d=64 扩展到 d=128。
2. ~~**量化 reduction**~~：已由 §4.6 的纯 warp shuffle 归约解决（相对 9.2
   快 2.13x–3.34x）。后续可尝试让一个 CTA 用 grid-stride 连续处理多个 token，
   进一步摊薄 launch 开销 —— 但 d>=256 已访存受限，收益预计有限。
3. **冷/热缓存双口径**：增加随机地址轮换或大于 L2 的多 buffer benchmark，分别报告
   cold-HBM 与 warm-L2 性能；取得 NCU 后用实际 DRAM/L2 bytes 建 memory-hierarchy
   Roofline。
4. **分布覆盖**：在 normal 之外加入 uniform、全零、极端 outlier 和真实模型 activation，
   分别报告 bit-exact 与量化误差，避免只优化合成 normal 数据。
5. **FP8 输出**：在不改变已验收 INT4 协议的前提下新增 E4M3/E5M2 分支，明确 scale
   粒度、饱和和 NaN/Inf 语义，再比较精度/带宽/硬件代际差异。
6. **与生产库对比**：以相同 shape、dtype、归一化、输入驻留和 Event 边界，对比
   Dao-AILab optimized kernel；避免用不同数据驻留状态做不公平比较。
7. **Tensor Core 的 INT8/FP8 通路**：本轮只用了 FP16/BF16 operand。由于 Hadamard
   矩阵元素是 `±1`，若下游本来就要 INT4/FP8 输出，可考虑先量化再在整数 Tensor
   Core 上做变换，把 §4.7 的中间舍入问题彻底绕开；代价是变换与量化的顺序改变，
   需要重新论证与 QuaRot 语义的等价性。
8. **d > 256 的 TC 融合量化**：目前 `d > 256` 时一个 token 跨多个 tile，per-token
   max 无法只靠 sub-warp 归约，接口显式回落。可用 tile 间的 accumulator 已在同一
   warp 寄存器内这一点做两级归约来支持。

## 10. 复现命令

```bash
make -j2
make smoke

# 内置 FP32 自检、性能矩阵、官方库 36 配置验收
bash scripts/run_tests.sh

# 只跑正式官方库验收
~/hadamard_env/bin/python tests/test_vs_library.py

# 9.1/9.2：baseline/optimized/unfused/fused 矩阵与图表
bash scripts/run_advanced.sh

# profiler
bash scripts/profile.sh nsys
bash scripts/profile.sh ncu

# Slurm：完整计时、Nsight、CSV 汇总和 Roofline 可视化
mkdir -p /scratch/gz2522/gz2522/tmp/hadamard/runs/slurm
sbatch slurm/profile_hadamard.slurm

# Slurm：优化与融合 A/B、Systems trace、两次 NCU 尝试
sbatch slurm/optimize_hadamard.slurm

# Slurm：在指定 GPU 分区重试真实硬件计数器
sbatch --partition=l40s_public slurm/ncu_retry.slurm

# Tensor Core 分支：构建（BF16 WMMA 需要 SM >= 80）
make -C tensor_core CUDA_ARCH=89 -j4

# Tensor Core 分支：单配置（五条变换 + 五条融合路径的正确性与 A/B）
./tensor_core/build/hadamard_tc_bench \
  --batch 4 --seq 1024 --heads 32 --head_dim 256 \
  --dtype fp16 --normalize true --warmup 20 --iters 100

# Tensor Core 分支：全矩阵扫描（FP16/BF16 x d=64..1024 + normalize=false + 尾部回落）
make -C tensor_core sweep

# Tensor Core 分支：集群一键（构建、扫描、nsys、ncu、CSV、图表）
sbatch tensor_core/slurm/tc_bench.slurm

# Tensor Core 分支：在 Hopper 节点单独采硬件计数器
sbatch tensor_core/slurm/tc_ncu_h200.slurm
```

关键证据清单：

| 产物 | 内容 |
|---|---|
| `results/library_check.csv` | 官方库 36 配置正确性 |
| `results/profile_16967596/timings.csv` | L40S baseline Event 计时 |
| `results/profile_16967596/figures/roofline.png` | 明确标注逻辑最小 I/O 的 Roofline |
| `results/profile_16967596/nsys_kernel_summary.csv` | baseline Systems kernel 元数据 |
| `results/optimization_16970010/advanced_results.csv` | 9.1/9.2 六配置 A/B、误差、压缩率 |
| `results/optimization_16970010/full_shape_checks.log` | 24 配置扩展一致性检查 |
| `results/optimization_16970010/nsys_summary/` | 四路径 Systems kernel/API/device CSV |
| `results/optimization_16970010/ncu/` | optimized/fused NCU 原始失败证据 |
| `results/ncu_retry_16975369/` | 第三块 L40S、精简四 section 的 NCU 重试证据 |
| `results/ncu_retry_16975370/` | A100 counter 被 DCGM 占用的跨架构重试证据 |
| `results/ncu_retry_16975372/` | H200 NCU 成功报告、raw CSV、硬件指标图 |
| `results/optimization_16970010/figures/advanced_performance.png` | 优化与融合四联图 |
| `tensor_core/results/tc_17083405/tc_results.csv` | 14 个唯一配置的历史时间/误差；旧 TC 融合尾部跳过 |
| `tensor_core/results/tc_17083405/sweep.log` | 逐配置原始输出（含判据与 verdict） |
| `tensor_core/results/tc_17083405/status.txt` | 历史宽松判据退出码；不代表严格 PDF 验收 |
| `tensor_core/results/tc_17083405/tc_summary.md` | 由 CSV 生成的摘要表 |
| `tensor_core/results/tc_17083405/nsys_summary/` | 十个 kernel 的 launch 形状、寄存器、shared memory |
| `tensor_core/results/tc_17083405/figures/tc_performance.png` | TC 与非 TC 的时间/误差四联图 |
| `tensor_core/results/tc_17083405/ncu/` | L40S counter 被 DCGM 占用的原始证据 |
| `tensor_core/results/tc_ncu_17083350/ncu_summary.md` | H200 上 warp/tc_fast/tc_split 的真实硬件计数器 |
| `tensor_core/results/tc_ncu_17083350/ncu/*.ncu-rep` | 三个 kernel 的 NCU 原始报告 |
| `tensor_core/results/verify_17251073/` | 第二轮 H200 三次计时、独立 FP32 参考、75 项 sanitizer |
| `tensor_core/results/verify_17251286/` | H100 真实参考库：五组各 132 配置与失败证据 |
| `tensor_core/results/ncu_17251063/` | H100 五组 × 两缓存策略 × 四目标，40 份原始 CSV |
| `tensor_core/results/verify_17251073/final_library/summary.md` | 本轮最终表格、NCU stall 分析与两张性能/Roofline 图 |

## 11. 当前验收结论

| 项目要求 | 状态 | 证据 |
|---|---|---|
| FP16，64/128/256 | 已完成 | 官方库 18/18，max_abs=0 |
| BF16，64/128/256 | 已完成 | 官方库 18/18，max_abs=0 |
| 多种 B/S/H 规模 | 已完成 | 1024/16384/131072 tokens |
| kernel 时间 ms | 已完成 | CUDA Event + CSV |
| 工程化注释与错误检查 | 已完成 | include/src/scripts 分层，CUDA_CHECK |
| nsys 采集入口 | 已完成 | L40S/2025.3 已取得 56 次 kernel timeline 与 API 汇总 |
| Roofline 与可视化 | 已完成（口径分列） | 历史 L40S 逻辑 I/O；本轮 H100 实测 DRAM all/none |
| ncu 硬件计数器 | 已完成（H200） | throughput、achieved occupancy、L2 hit、warp stall |
| 9.1 shuffle/vectorized/multi-token 优化 | 已完成 | L40S 1.24x–3.42x，6/6 bit-exact |
| 9.2 INT4 融合量化及一致性 | 已完成 | fused 1.26x–2.33x，三组 bit-exact 检查全过 |
| INT4 压缩与误差 | 已完成 | 3.56x–3.88x，MAE 0.0912–0.1084 |
| warp-per-token FHT 再优化 | 已完成 | d=128 再快 1.68x，shared/barrier 归零，与 9.1 100% 逐位一致 |
| warp-per-token 融合 INT4 | 已完成 | 相对 9.2 快 2.13x–3.34x，逐位一致性检查全过 |
| Tensor Core 加速实现 | 实现完成，非全面精度合格 | Kronecker WMMA；库验收 fast 96/132、split 121/132 |
| Tensor Core 融合量化 | 已完成（d<=256） | 与同算法 unfused 逐位一致；d>256 显式回落 |
| Tensor Core vs 非 TC 对比（进阶项） | 已完成 | 历史 L40S d=64 TC 占优；本轮 H200 核心维度 warp 变换仍更快 |
| warp/TC 的 ncu 硬件计数器 | 已完成（H200） | warp 的 barrier stall 为 0；TC 的主导 stall 是 short scoreboard / mio throttle |
| 国产平台适配 | 未开始 | 当前仅 NVIDIA CUDA / SM 8.0、8.6、8.9、9.0；Tensor Core 分支要求 SM >= 80 |

## 12. 参考资料

1. [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](https://arxiv.org/abs/2404.00456)
2. [SpinQuant: LLM quantization with learned rotations](https://arxiv.org/abs/2405.16406)
3. [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608)
4. [Dao-AILab fast-hadamard-transform 官方仓库](https://github.com/Dao-AILab/fast-hadamard-transform)
5. [NVIDIA Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/)
6. [NVIDIA ERR_NVGPUCTRPERM 处理说明](https://developer.nvidia.com/ERR_NVGPUCTRPERM)
7. [CUDA C++ Programming Guide — Warp Matrix Functions (WMMA)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-matrix-functions)
8. [NVIDIA Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
