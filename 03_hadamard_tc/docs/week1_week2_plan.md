# Week1/Week2 工作拆解（第一部分）

目标：先正确、可测试、可复现，再谈性能。Week1 打地基，Week2 出 kernel。

## Week1：需求确认、参考实现、测试框架

| 阶段 | 任务 | 交付物 | 完成判据 |
|---|---|---|---|
| W1-D1 | 需求确认：整理问题清单，向导师/助教确认归一化约定、提交目录、量化格式等 | `docs/questions_for_mentor.md`，群内提问记录 | 关键歧义（尤其归一化）有书面答复 |
| W1-D2 | 项目目录初始化：CMake/Makefile、目录骨架、README | 可 `make` 出占位程序的工程骨架 | `make` 无错误；目录与文档一致 |
| W1-D3 | 数据生成方案：normal / uniform / outlier 三种分布，固定种子，支持二进制保存/加载 | `src/main.cu` 中 `generate_input` + `--input_bin`；dump 二进制 | 相同种子两次生成逐位一致 |
| W1-D4 | 参考实现：快速 WHT（蝶形）、FP32 内部累加、归一化可选；**最终仅保留官方 `fast_hadamard_transform` 库作为唯一参考** | `tests/hadamard_reference.py`（官方库封装）；`src/reference_cpu.h`（C++/FP32，仅 bench 自检） | 与官方库 36 配置 bit-exact；库自检（impulse/对合/保范数）通过 |
| W1-D5 | 误差评估脚本：max_abs / mean_abs / rmse / pass-fail，FP16<1e-2、BF16<5e-2 | `tests/check_error.py`；bench 内置 `--check` | 对已知错误输入能正确报 FAIL |
| W1-D6 | 基础测试矩阵：dtype × head_dim × 规模，先小后大 | `scripts/run_tests.sh` | 18 核心配置 + 4 分布扩展配置全部可一键运行 |
| W1-D7 | 汇总与复盘：跑一遍小配置，确认流程闭环 | `results/` 下的 CSV/报告雏形 | `bash scripts/run_tests.sh small` 通过 |

## Week2：CUDA baseline kernel

| 阶段 | 任务 | 交付物 | 完成判据 |
|---|---|---|---|
| W2-D1 | kernel v1：一模板化 `hadamard_kernel<T, HEAD_DIM, NORMALIZE>`，一 block 一 token，shared memory + 蝶形 | `src/hadamard.cu` | 编译通过，无 CUDA 错误 |
| W2-D2 | host 接口与分发：`launch_hadamard(...)`，dtype × head_dim × normalize 分发，错误码约定 | `include/hadamard.cuh` | 非法配置返回明确错误码而非崩溃 |
| W2-D3 | 计时与主程序：CUDA Event、warmup、avg/min/max；命令行参数 | `build/hadamard_bench` | `--warmup/--iters` 生效，时间稳定 |
| W2-D4 | 正确性验证闭环：GPU vs CPU FP32 参考，统一归一化，FP32 域比较 | bench `--check` + dump 交叉验证 | 18 核心配置全部 PASS |
| W2-D5 | 误差归因：区分"实现误差"与"输出格式舍入极限"，补充相对误差指标 | `max_rel_error` 指标 + `results/report.md` 注释 | 实测相对误差 == 格式理论极限（证明 bit-exact） |
| W2-D6 | 批量测试与报告：一键扫描 → CSV → Markdown 报告 | `scripts/run_tests.sh` + `scripts/make_report.py` | `results/report.md` 自动生成 |
| W2-D7 | 收尾：文档、验收清单、给 Week3 的扩展接口说明 | `docs/design.md`、`docs/acceptance_checklist.md`、README | 验收清单逐项可核对 |

## 当前实际进度（2026-08-30）

Week1/Week2 全部完成。**参考实现仅保留官方 `fast_hadamard_transform` 库**：
对官方库的 36 配置验收（2 dtype × 3 head_dim × 3 规模 × 归一化/非归一化）
**全部通过且 100% bit-exact（max_abs=0.0）**，远优于阈值（FP16<1e-2、BF16<5e-2）。

内置自检参考（CPU FP32）的 22 配置中 21 个通过绝对阈值；唯一未通过的
（bf16+大 outlier）已证明是输出格式舍入的物理极限（相对误差 3.89e-3 == 2^-8 理论上界），
非实现问题。详见 `results/report.md`、`results/library_check.csv` 与 `docs/design.md`。
