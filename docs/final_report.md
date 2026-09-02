# Hadamard 变换加速项目总结报告

> 项目：2026 夏季训练营 CUDA 方向项目·选题三
>
> 当前阶段：Week1/Week2 baseline 正确性与性能闭环
>
> 最近实测：2026-09-02，NVIDIA GeForce RTX 3060 Laptop GPU（SM 8.6）
>
> 报告状态：baseline 已填写；融合量化、Tensor Core 对比数据待后续阶段补充

## 1. 摘要

本阶段实现了一个面向 `[batch_size, seq_len, num_heads, head_dim]` 激活张量的
CUDA 快速 Walsh-Hadamard Transform（FHT）baseline。实现支持 FP16、BF16，核心
`head_dim` 为 64/128/256，并额外实例化了 32/512/1024。kernel 将输入逻辑展平为
`[total_tokens, head_dim]`，每个 block 处理一个 token，在 shared memory 中使用
FP32 完成 `log2(head_dim)` 轮蝶形计算，最后可选归一化并转换回输入类型。

以 Dao-AILab `fast_hadamard_transform` 1.1.0 为验收参考，在 2 种 dtype、3 种
`head_dim`、3 档输入规模、归一化/非归一化两种语义组成的 36 个配置上，当前实现
全部通过，`max_abs_error=0`，输出逐位一致率为 100%。RTX 3060 Laptop 上，131072
tokens 的归一化变换耗时为 0.35–1.67 ms。当前代码尚未实现融合量化与 Tensor Core，
二者作为下一阶段优化项，不能视作本阶段已交付功能。

## 2. 背景与目标

激活中的少量异常值会放大量化 scale，挤压大多数普通值的有效量化区间。正交旋转把
能量扩散到多个通道，可降低单通道动态范围，同时保持全精度网络的等价性。QuaRot
展示了旋转后端到端 4 bit 权重、激活和 KV cache 量化；SpinQuant 进一步学习旋转矩阵；
FlashAttention-3 则将 incoherent processing 与 FP8 block quantization 用于 Hopper
注意力计算。相关一手资料见文末参考文献。

本阶段目标不是宣称极限性能，而是建立可靠 baseline：

1. 支持题目要求的形状、FP16/BF16 和 64/128/256 维度；
2. 与官方 CUDA 参考库对齐变换次序、归一化和输出 dtype；
3. 建立可复现的正确性矩阵、CUDA Event 性能日志和 profiler 入口；
4. 为融合量化和 Tensor Core 实验保留清晰扩展点。

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
- `src/reference_cpu.h`：开发期 FP32 自检；
- `tests/test_vs_library.py`：正式验收矩阵；
- `scripts/run_tests.sh`：一键正确性与性能扫描；
- `scripts/profile.sh`：ncu/nsys 采集和旧版 qdstrm 导入兼容。

关键函数均注明张量语义、精度策略、同步原因和错误码。所有 CUDA Runtime 调用统一经
`CUDA_CHECK` 检查，kernel launch 后检查异步错误，不静默吞错。

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

### 5.4 当前未覆盖项

题目要求融合量化后验证“融合结果”和“先变换后量化”一致。当前 Week1/Week2 尚未
实现融合量化，因此没有对该项打勾。后续必须先确定 FP8/INT4、scale 粒度、舍入模式、
饱和范围和 INT4 packing 顺序，再建立 bit-exact 对照测试。

## 6. 性能结果与分析

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

## 8. 优化历程与开发中发现的问题

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

## 9. 下一阶段计划

### 9.1 baseline kernel 优化

1. 用 `__shfl_xor_sync` 完成 warp 内 stride，删除对应 block barrier/shared 往返；
2. 使用 `half2`/`__nv_bfloat162` 向量化 global load/store；
3. 对 d=64/128 尝试一 block 多 token，提高小任务占用与并行度；
4. 建立 `baseline / shuffle / vectorized` 同输入、同计时边界的性能回归表；
5. 权限开放后用 ncu 指标决定优化优先级，而不是只看总时间猜瓶颈。

### 9.2 融合量化

先由导师确认 FP8/INT4 和 scale 粒度。建议按以下顺序推进：

1. 实现独立 reference quantizer，固定 rounding、clamp、scale 与 packing 语义；
2. 实现 unfused `Hadamard -> quantize` GPU pipeline；
3. 在 Hadamard 写回寄存器处直接求 scale、量化并输出，避免中间低精度张量落显存；
4. 对每种 shape/dtype/分布验证 fused 与 unfused bit-exact；
5. 同时报告 kernel-only 与端到端时间，量化融合的价值应以减少中间显存流量体现。

### 9.3 Tensor Core 探索

Tensor Core 是实验分支，不应直接用稠密 `H_d @ x` 替代 FHT：后者会把复杂度从
`O(d log d)` 退化为 `O(d^2)`。可行方向是把 Hadamard 分解为 16x16 小块变换并用
WMMA/MMA 批量处理多个 token，或把旋转吸收到相邻 GEMM 中。验收必须同时比较：

- 数值误差；
- 单独 Hadamard kernel 时间；
- 与量化/相邻 GEMM 融合后的端到端时间；
- 相对优化后的 shuffle baseline 的加速比，而不只相对当前教学 baseline。

## 10. 复现命令

```bash
make -j2
make smoke

# 内置 FP32 自检、性能矩阵、官方库 36 配置验收
bash scripts/run_tests.sh

# 只跑正式官方库验收
~/hadamard_env/bin/python tests/test_vs_library.py

# profiler
bash scripts/profile.sh nsys
bash scripts/profile.sh ncu
```

## 11. 当前验收结论

| 项目要求 | 状态 | 证据 |
|---|---|---|
| FP16，64/128/256 | 已完成 | 官方库 18/18，max_abs=0 |
| BF16，64/128/256 | 已完成 | 官方库 18/18，max_abs=0 |
| 多种 B/S/H 规模 | 已完成 | 1024/16384/131072 tokens |
| kernel 时间 ms | 已完成 | CUDA Event + CSV |
| 工程化注释与错误检查 | 已完成 | include/src/scripts 分层，CUDA_CHECK |
| nsys 采集入口 | 已完成 | `.nsys-rep/.sqlite` 已生成；旧版本无 GPU kernel 明细 |
| ncu 指标 | 环境阻塞 | ERR_NVGPUCTRPERM，需管理员开放计数器 |
| 融合量化及一致性 | 未开始 | 待确认 quant/scale/packing 规格 |
| Tensor Core 对比 | 未开始 | 下一阶段实验分支 |
| 国产平台适配 | 未开始 | 当前仅 NVIDIA CUDA / SM 8.6 |

## 12. 参考资料

1. [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](https://arxiv.org/abs/2404.00456)
2. [SpinQuant: LLM quantization with learned rotations](https://arxiv.org/abs/2405.16406)
3. [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608)
4. [Dao-AILab fast-hadamard-transform 官方仓库](https://github.com/Dao-AILab/fast-hadamard-transform)
5. [NVIDIA Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/)
6. [NVIDIA ERR_NVGPUCTRPERM 处理说明](https://developer.nvidia.com/ERR_NVGPUCTRPERM)
