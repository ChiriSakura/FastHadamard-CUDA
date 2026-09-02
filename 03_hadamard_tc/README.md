# CUDA Hadamard 变换加速

本目录实现输入形状 `[batch_size, seq_len, num_heads, head_dim]` 最后一维上的
快速 Walsh-Hadamard Transform。当前 baseline 支持 FP16、BF16，核心维度为
64/128/256，内部采用 FP32 蝶形计算，并支持归一化 `Hx/sqrt(head_dim)`。

## 构建与运行

```bash
make -j2

./build/hadamard_bench \
  --batch 4 --seq 1024 --heads 32 --head_dim 128 \
  --dtype fp16 --normalize true --warmup 5 --iters 20 --check true
```

也可使用 CMake：

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j
```

## 验证

```bash
# 小规模 CUDA/CPU 冒烟
make smoke

# FP16/BF16、64/128/256、多档规模的正确性与性能扫描
bash scripts/run_tests.sh

# 只运行官方 fast_hadamard_transform 对照
~/hadamard_env/bin/python tests/test_vs_library.py
```

正式验收结果：36/36 配置通过，GPU 输出与官方库 100% bit-exact。原始记录位于：

- [`results/library_check.csv`](results/library_check.csv)：官方库正确性对照；
- [`results/results.csv`](results/results.csv)：内置 FP32 自检与 CUDA Event 时间。

## Profiler

```bash
bash scripts/profile.sh nsys
bash scripts/profile.sh ncu
```

ncu 需要 GPU performance-counter 权限。脚本不会修改系统权限，无法采集时会输出
`ERR_NVGPUCTRPERM` 提示。

## 目录结构

```text
03_hadamard_tc/
├── CMakeLists.txt
├── Makefile
├── include/                 # 公共接口与 CUDA 错误检查
├── src/                     # CUDA kernel、benchmark、CPU 开发参考
├── tests/                   # 官方库对照与 dump 检查
├── scripts/                 # 一键测试和 profiler 入口
├── results/                 # 小型 CSV 验收证据
└── docs/final_report.md     # 唯一项目总结报告
```

## 当前范围

当前完成的是 Week1/Week2 正确性 baseline。融合量化、INT4 packing、FP8、Tensor Core
实现和国产平台适配尚未完成，后续路线与实验口径见
[`docs/final_report.md`](docs/final_report.md)。
