# FastHadamard-CUDA

2026 夏季训练营 CUDA 方向项目：Hadamard 变换加速。

当前仓库只保留选题三的实现，项目入口位于
[`03_hadamard_tc/`](03_hadamard_tc/README.md)。其中包含 FP16/BF16 CUDA baseline、
正确性测试、性能日志与最终项目报告。

## 当前状态

- NVIDIA RTX 3060 Laptop（SM 8.6）上构建、运行通过；
- 支持 FP16/BF16 和 head_dim 64/128/256；
- 与 `fast_hadamard_transform` 的 36 个配置全部 bit-exact；
- 已建立 CUDA Event 性能测试及 nsys/ncu 采集入口；
- 融合量化和 Tensor Core 对比属于下一阶段工作。

完整结论见 [`03_hadamard_tc/docs/final_report.md`](03_hadamard_tc/docs/final_report.md)。
