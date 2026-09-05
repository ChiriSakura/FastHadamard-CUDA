# Hadamard 变换加速项目总结报告

> 项目：2026 夏季训练营 CUDA 方向项目·选题三
>
> 当前阶段：Week1/Week2 baseline + 9.1 kernel 优化 + 9.2 融合 INT4
>
> 最近实测：2026-09-04，NVIDIA L40S（SM 8.9）
>
> 报告状态：baseline、优化 FHT、融合量化已填写；Tensor Core 对比待后续阶段

## 1. 摘要

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
GPU-vs-CPU quantizer 三项 bit-exact 检查。Tensor Core 仍是下一阶段实验项。

### 1.1 交付状态总览

| 交付项 | 状态 | 主要证据 |
|---|---|---|
| FP16/BF16 baseline | 完成 | 官方库 36/36，逐位一致 |
| 9.1 optimized FHT | 完成 | L40S 1.24x–3.42x；核心 6/6 bit-exact |
| 9.2 fused INT4 | 完成 | L40S 1.26x–2.33x；三类一致性检查全过 |
| 扩展 shape/normalize | 完成 | 24/24 配置三项 bit-exact |
| Nsight Systems | 完成 | kernel timeline、launch 形状、register/shared-memory、API 汇总 |
| Roofline | 完成（逻辑最小 I/O） | 同节点理论 FP32/HBM roof；数据来源明确标注 |
| Nsight Compute 硬件计数器 | **完成（H200）** | optimized/fused 的 throughput、occupancy、cache、stall 已实测 |

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
6. 为 Tensor Core 实验保留清晰扩展点。

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
- `src/hadamard.cu`：模板 kernel 和 `(dtype, head_dim, normalize)` 静态分发；
- `src/main.cu`：输入生成、参数解析、CUDA Event 计时、dump 与 CSV；
- `src/advanced_bench.cu`：9.1/9.2 A/B 计时、CPU INT4 reference 和 bit-exact 检查；
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
逻辑张量流量；原因是被选中的 validation launch 之前已有 kernel 预热输入缓存。因此
这些 counter 作为 warm-cache kernel 诊断，不与 L40S 的 HBM Roofline 混算。完整
`.ncu-rep`、raw CSV、精简表和图在 `results/ncu_retry_16975372/`。

![H200 NCU 硬件指标](../results/ncu_retry_16975372/ncu_hardware_metrics.png)

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

### 8.2 开发中发现的问题与处理

1. **先锁定数学语义**：官方库默认 `scale=1.0`，项目默认正交归一化。测试显式传
   scale，同时覆盖两种语义，避免“数值都像对的但差一个 `sqrt(d)`”。
2. **修正最后一维语义**：官方库按最后一维变换；一维 dump 必须 reshape 为
   `[-1, head_dim]`，否则会把全部 token 当一行。
3. **统一累加精度和顺序**：低精度输入转 FP32、同序蝶形、末端一次舍入，使本实现
   与官方库达到 bit-exact。
4. **区分实现误差和格式误差**：大 outlier 下固定绝对阈值会小于 BF16 half-ULP。
   增加平滑相对误差并保留官方低精度对照，避免错误地为“通过阈值”修改正确 kernel。
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

## 9. 文档任务执行状态

### 9.1 baseline kernel 优化（已完成）

1. 已用 `__shfl_xor_sync` 完成 stride=2–32，stride=1 留在两个寄存器内；
2. 已用 `half2`/`__nv_bfloat162` 向量化 global load/store；
3. d=64/128 已采用每 block 4/2 tokens；
4. 已建立 baseline/optimized 同输入同边界 A/B，6/6 bit-exact；
5. L40S FHT 实测加速 1.24x–3.42x；Systems trace 已取得，NCU 计数器仍被环境占用。

### 9.2 融合量化（已完成 INT4）

1. 已固定 per-token symmetric INT4 的 scale、round-to-nearest-even、[-7,7] clamp 和
   low/high nibble packing；
2. 已实现 CPU reference quantizer 和 unfused GPU pipeline；
3. 已在 FHT 写回点融合 max-reduction、scale、quantize 和 packing；
4. FP16/BF16 × d=64/128/256 的 fused/unfused 与 GPU/CPU 均 bit-exact；
5. 已报告端到端时间、3.56x–3.88x 压缩率和量化误差；融合加速 1.26x–2.33x。

FP8 尚未实现。它是另一套输出格式与标度协议，不应与已验收的 INT4 数据混写。

### 9.3 Tensor Core 探索

Tensor Core 是实验分支，不应直接用稠密 `H_d @ x` 替代 FHT：后者会把复杂度从
`O(d log d)` 退化为 `O(d^2)`。可行方向是把 Hadamard 分解为 16x16 小块变换并用
WMMA/MMA 批量处理多个 token，或把旋转吸收到相邻 GEMM 中。验收必须同时比较：

- 数值误差；
- 单独 Hadamard kernel 时间；
- 与量化/相邻 GEMM 融合后的端到端时间；
- 相对优化后的 shuffle baseline 的加速比，而不只相对当前教学 baseline。

### 9.4 可继续提升的方向

1. **d=256 跨 warp 数据交换**：当前仍有两轮 shared-memory/barrier。可尝试 cooperative
   groups tiled partition、重新映射每线程元素数，或将 4-warp 子变换分层合并。
2. **量化 reduction**：fused d=128/256 可用 warp-level max + 更紧凑的跨 warp归约，
   并尝试让一个 CTA 连续处理多个 token，摊薄 scale 写入和 launch 开销。
3. **冷/热缓存双口径**：增加随机地址轮换或大于 L2 的多 buffer benchmark，分别报告
   cold-HBM 与 warm-L2 性能；取得 NCU 后用实际 DRAM/L2 bytes 建 memory-hierarchy
   Roofline。
4. **分布覆盖**：在 normal 之外加入 uniform、全零、极端 outlier 和真实模型 activation，
   分别报告 bit-exact 与量化误差，避免只优化合成 normal 数据。
5. **FP8 输出**：在不改变已验收 INT4 协议的前提下新增 E4M3/E5M2 分支，明确 scale
   粒度、饱和和 NaN/Inf 语义，再比较精度/带宽/硬件代际差异。
6. **与生产库对比**：以相同 shape、dtype、归一化、输入驻留和 Event 边界，对比
   Dao-AILab optimized kernel；避免用不同数据驻留状态做不公平比较。

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

## 11. 当前验收结论

| 项目要求 | 状态 | 证据 |
|---|---|---|
| FP16，64/128/256 | 已完成 | 官方库 18/18，max_abs=0 |
| BF16，64/128/256 | 已完成 | 官方库 18/18，max_abs=0 |
| 多种 B/S/H 规模 | 已完成 | 1024/16384/131072 tokens |
| kernel 时间 ms | 已完成 | CUDA Event + CSV |
| 工程化注释与错误检查 | 已完成 | include/src/scripts 分层，CUDA_CHECK |
| nsys 采集入口 | 已完成 | L40S/2025.3 已取得 56 次 kernel timeline 与 API 汇总 |
| Roofline 与可视化 | 已完成（逻辑 I/O 口径） | L40S 6 配置，52.5%–70.6% memory roof |
| ncu 硬件计数器 | 已完成（H200） | throughput、achieved occupancy、L2 hit、warp stall |
| 9.1 shuffle/vectorized/multi-token 优化 | 已完成 | L40S 1.24x–3.42x，6/6 bit-exact |
| 9.2 INT4 融合量化及一致性 | 已完成 | fused 1.26x–2.33x，三组 bit-exact 检查全过 |
| INT4 压缩与误差 | 已完成 | 3.56x–3.88x，MAE 0.0912–0.1084 |
| Tensor Core 对比 | 未开始 | 下一阶段实验分支 |
| 国产平台适配 | 未开始 | 当前仅 NVIDIA CUDA / SM 8.0、8.6、8.9、9.0 |

## 12. 参考资料

1. [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](https://arxiv.org/abs/2404.00456)
2. [SpinQuant: LLM quantization with learned rotations](https://arxiv.org/abs/2405.16406)
3. [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608)
4. [Dao-AILab fast-hadamard-transform 官方仓库](https://github.com/Dao-AILab/fast-hadamard-transform)
5. [NVIDIA Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/)
6. [NVIDIA ERR_NVGPUCTRPERM 处理说明](https://developer.nvidia.com/ERR_NVGPUCTRPERM)
