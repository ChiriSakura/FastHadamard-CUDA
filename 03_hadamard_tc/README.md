# Hadamard 变换加速 — Week1/Week2 交付

> 2026 夏季训练营 CUDA 项目·选题三。本阶段目标：**正确、可测试、可复现**，
> 完成基础但正确的 CUDA 快速 Hadamard 变换 kernel（FP16/BF16，head_dim 64/128/256），
> 不追求极限性能、不做 Tensor Core / 融合量化（仅预留接口）。
> 完整任务要求见 `skill.md`；设计与结论见 `docs/`。
>
> **正确性基准 = 官方 `fast_hadamard_transform` 库（Dao-AILab）。**
> 在 36 个配置（2 dtype × 3 head_dim × 3 规模 × 归一化/非归一化）上，
> 本 kernel 输出与官方库 **100% bit-exact（max_abs_error = 0.0）**，
> 远优于验收阈值（FP16 < 1e-2、BF16 < 5e-2）。见 `results/library_check.csv`。

## 项目背景

低比特量化前对激活做随机旋转（Hadamard 变换）可抑制异常值、提升 FP8/INT4 量化精度
（QuaRot、SpinQuant、FlashAttention-3 的核心思想），但变换本身引入计算开销。
本项目实现快速 Hadamard 变换核，支持多种输入尺寸，后续将与量化算子融合、探索 Tensor Core。

## 快速开始

```bash
# 方式一：直接 make（本机无 cmake 时用）
make                          # 产出 build/hadamard_bench

# 方式二：标准 CMake
mkdir build && cd build && cmake .. && make -j

# 单次运行（默认就带正确性检查）
./build/hadamard_bench --batch 4 --seq 1024 --heads 32 --head_dim 128 \
                       --dtype fp16 --normalize true --warmup 5 --iters 20 --check true

# 一键批量测试（内置参考正确性 + 性能 + 官方库验收）
bash scripts/run_tests.sh

# 官方库验收测试（唯一参考实现 = fast_hadamard_transform；依赖见下节）
~/hadamard_env/bin/python tests/test_vs_library.py          # 全矩阵 36 配置
~/hadamard_env/bin/python tests/test_vs_library.py small    # 小规模冒烟
```

## 当前结果摘要（RTX 3060 Laptop, sm_86）

- **对官方库（验收基准）**：36 个配置全部通过，且全部 **100% bit-exact**
  （max_abs_error = 0.0，阈值 FP16<1e-2 / BF16<5e-2）。见 `results/library_check.csv`。
  归一化与非归一化两种模式都已验证与官方库一致。
- **对内置自检参考**（CPU FP32，仅开发期快速自检，非验收口径）：
  22 个配置中 21 个通过绝对阈值。唯一未通过项（bf16+大 outlier）为输出舍入
  物理极限（❌*），详见 `results/report.md` 与 `docs/design.md` 误差分析。
- **性能**（kernel-only，warmup=5, iters=20）：
  4x1024x32 (131072 tokens, head_dim=128) ≈ 0.75 ms；head_dim=256 ≈ 1.64 ms。
  baseline 为"一 block 一 token、一线程一元素"，访存与占用均有明显优化空间（Week3+）。
- 详细数据：`results/results.csv`、`results/report.md`、`results/library_check.csv`。
- 可提交的完整总结、最新实测性能、nsys/ncu 分析与后续优化路线见
  [`docs/final_report.md`](docs/final_report.md)。

## Profiler

```bash
bash scripts/profile.sh nsys   # 系统时间线与 CUDA API 汇总
bash scripts/profile.sh ncu    # kernel 硬件指标（需要 GPU performance-counter 权限）
```

脚本会把 profiler 产物写入 `results/`，并兼容旧版 nsys 只生成 `.qdstrm` 时使用
同版本 `QdstrmImporter` 转换。当前机器上的实际权限/版本限制及已得到的分析结论见
`docs/final_report.md` 第 7 节。

## 依赖安装（官方参考实现环境）

验收以官方 `fast_hadamard_transform` 库为基准。本机无系统级 `pip`/`sudo`，
采用**用户态 venv** 安装（不影响系统）。已在本机验证可复现：

```bash
# 1) 建 venv（系统 python 无 ensurepip，用 --without-pip + get-pip 引导）
python3 -m venv --without-pip ~/hadamard_env
curl -s -o /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py
~/hadamard_env/bin/python /tmp/get-pip.py

# 2) 装 CUDA 版 torch（与本机 nvcc 12.0 / 驱动配套）+ numpy
~/hadamard_env/bin/pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
~/hadamard_env/bin/pip install numpy ninja wheel packaging

# 3) 补 Python 头文件（无 sudo 装不了 python3-dev）：
#    apt-get download 不需 sudo，解包后把 include 目录交给编译器
cd /tmp && mkdir -p pyhdr && cd pyhdr
apt-get download libpython3.12-dev python3.12-dev
dpkg-deb -x libpython3.12-dev_*.deb . && dpkg-deb -x python3.12-dev_*.deb .
mkdir -p ~/local/include
cp -r usr/include/python3.12 usr/include/x86_64-linux-gnu ~/local/include/

# 4) 从 GitHub 源码编译安装官方库（PyPI 的 sdist 缺 csrc/，不要用）
git clone --depth 1 https://github.com/Dao-AILab/fast-hadamard-transform.git
cd fast-hadamard-transform
export CPLUS_INCLUDE_PATH=$HOME/local/include/python3.12:$HOME/local/include
export C_INCLUDE_PATH=$CPLUS_INCLUDE_PATH
TORCH_CUDA_ARCH_LIST="8.6" ~/hadamard_env/bin/pip install --no-build-isolation .
```

装好后即可运行官方库验收测试（已验证 36/36 通过）：
```bash
~/hadamard_env/bin/python tests/test_vs_library.py
```

## 目录结构（第三部分）

```
03_hadamard_tc/
├── README.md                  # 本文件：背景/构建/运行/结果
├── skill.md                   # 训练营任务书（原始要求）
├── CMakeLists.txt             # 标准 CMake 工程（训练营推荐构建方式）
├── Makefile                   # 无 cmake 环境的等价构建（本机使用）
├── include/
│   ├── hadamard.cuh           # host 接口：launch_hadamard + DataType
│   └── cuda_check.cuh         # CUDA_CHECK 错误封装（不吞错）
├── src/
│   ├── hadamard.cu            # kernel：模板 <T, HEAD_DIM, NORMALIZE> + 分发
│   ├── reference_cpu.h        # CPU FP32 参考（bench --check 内置自检用）+ 误差指标
│   └── main.cu                # hadamard_bench：参数/数据生成/计时/检查/CSV/dump
├── scripts/
│   ├── run_tests.sh           # 一键批量测试（内置自检 + 官方库验收）
│   └── make_report.py         # CSV -> Markdown 报告
├── tests/
│   ├── hadamard_reference.py  # 参考实现封装：仅官方 fast_hadamard_transform 库
│   ├── check_error.py         # 误差评估：读 dump，用官方库算标准答案 -> pass/fail
│   └── test_vs_library.py     # 官方库验收测试（全矩阵 / small 冒烟）
├── results/                   # 测试产物：CSV / report.md / dumps
└── docs/
    ├── week1_week2_plan.md    # 第一部分：工作拆解
    ├── questions_for_mentor.md# 第二部分：待确认问题清单
    ├── design.md              # 第四、五部分：参考实现与 CUDA 设计 + 误差分析
    └── acceptance_checklist.md# 第八部分：验收清单
```

## hadamard_bench 命令行（第六部分入口）

| 参数 | 说明 | 默认 |
|---|---|---|
| `--batch/--seq/--heads/--head_dim` | 形状（展平为 total_tokens × head_dim） | 4 / 1024 / 32 / 128 |
| `--dtype` | `fp16` \| `bf16` | fp16 |
| `--normalize` | `true`: y=Hx/√d（默认）；`false`: y=Hx | true |
| `--dist` | 输入分布：`normal` \| `uniform` \| `outlier` | normal |
| `--seed` | RNG 种子（可复现） | 42 |
| `--outlier_ratio/--outlier_scale` | outlier 分布参数 | 0.05 / 20 |
| `--warmup/--iters` | 预热 / 计时迭代次数 | 5 / 20 |
| `--check` | 正确性检查（对 CPU FP32 参考） | true |
| `--dump_dir` | dump 输入/输出二进制 + meta（供 Python 复核） | 关 |
| `--csv` | 结果追加到 CSV | 关 |
| `--input_bin` | 从文件加载低精度输入（跳过随机生成） | 关 |

退出码：0 = 通过（或未开检查）；1 = check 未通过；2 = 参数/配置非法。

## 正确性验证流程（第七部分）

**唯一参考实现 = 官方 `fast_hadamard_transform` 库。** 两层验证：

1. **内置自检**（`run_tests.sh` 里的 bench `--check`）：CPU FP32 参考，
   用于 kernel 开发期的快速自我一致性检查与性能计时，
   写 `results/results.csv` → 汇总 `results/report.md`。它不作为验收口径；
2. **官方库验收**（`tests/test_vs_library.py`，唯一验收基准）：
   与 `fast_hadamard_transform` 逐位对照（bench 以 `--check false` 跑、落盘，
   避免与内置自检混用）。`tests/check_error.py` 可对任意 dump 用官方库复核。

判断标准（对官方库）：FP16 max_abs_error < 1e-2、BF16 < 5e-2；
比较统一在 FP32 域、统一归一化策略（归一化传 scale=1/√d，否则用库默认 1.0）。

## 当前阶段的边界（按题目要求）

- 不做：Tensor Core、融合量化、INT4 packed、warp-shuffle 特化、极致访存优化；
- 预留：`normalize`/`dtype` 运行时参数、`--input_bin` 数据复用、
  `launch_hadamard` 分发点（后续接融合量化 / Tensor Core 版本，见 `docs/design.md` §5）。

## 官方库行为实测结论（重要）

对 `fast_hadamard_transform` 1.1.0 的实测（对照源码 + impulse/整数精确测试）：

- **默认 `scale=1.0`，即非归一化** `y = H·x`；归一化需显式传 `scale=1/sqrt(head_dim)`。
- 矩阵 = `scipy.linalg.hadamard(dim)`，**Sylvester 自然序**（与本项目蝶形同序）。
- 内部 **FP32 累加**、末端舍入回输入 dtype——与本 kernel 同策略，这是能
  **100% bit-exact** 的根本原因。
- 按**最后一维**变换；调用前务必 `reshape(-1, head_dim)`，误传 1D 会把整段数据当一行。

因为归一化与非归一化两种模式本 kernel 都与官方库 bit-exact 一致，
**无论助教/评测采用哪种约定都通过验收**；归一化口径的书面确认仍保留在
`docs/questions_for_mentor.md`。

## 已知事项 / 风险

1. 官方库依赖需用户态安装（步骤见上"依赖安装"），系统无 `pip`/`sudo`；
   本机已装好并验证。参考实现只认官方库：在无 torch 环境跑
   `check_error.py` / `test_vs_library.py` 会明确报错并提示安装，不做降级。
   不依赖 torch 的只有 bench 内置自检（`--check`，CPU FP32）与 `run_tests.sh`
   的性能/自检部分。
2. 非归一化模式（`--normalize false`）在大 head_dim 下，输出量级放大 √d 倍，
   其**绝对误差**会超过阈值——这是低精度输出舍入的物理极限，非实现问题
   （与官方库相比仍 bit-exact）。量化旋转场景请用归一化模式。
3. bench 的退出码 1 同时表示"check 未通过"与"运行期 CUDA 错误"（后者会打印
   `[CUDA_CHECK]`/`[CUDA_SYNC_CHECK]` 日志，可据此区分）。
