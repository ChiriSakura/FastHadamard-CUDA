# Week1/Week2 验收清单（第八部分）

对照检查（全部基于本机实测，RTX 3060 Laptop / CUDA 12.0 / sm_86）：

| # | 验收项 | 状态 | 证据/复现方式 |
|---|---|---|---|
| 1 | 支持输入形状 `[batch, seq, heads, head_dim]`，任意组合展平为 `[total_tokens, head_dim]` | ✅ | sweep 覆盖 1x128x8 ~ 4x1024x32 |
| 2 | 支持 dtype：FP16、BF16 | ✅ | 每个 head_dim 两种 dtype 均测 |
| 3 | 支持 head_dim：64 / 128 / 256（另预留 32/512/1024） | ✅ | 64/128/256 全过；32/512/1024 已实例化可分发 |
| 4 | **FP16 vs 官方库**：max_abs_error < 1e-2 | ✅ | 18 个 FP16 配置 **全部 0.0（100% bit-exact）**，`results/library_check.csv` |
| 5 | **BF16 vs 官方库**：max_abs_error < 5e-2 | ✅ | 18 个 BF16 配置 **全部 0.0（100% bit-exact）**，`results/library_check.csv` |
| 5b | 对内置/独立参考（normal/uniform 分布） | ✅ | FP16 最大 7.8e-3、BF16 最大 1.56e-2，均达标；大 outlier 例外见下 |
| 6 | 能输出性能日志（kernel 时间 ms） | ✅ | CUDA Event，avg/min/max，写入 `results/results.csv` |
| 7 | 能一键运行测试 | ✅ | `bash scripts/run_tests.sh`（small/all 两档，含官方库验收） |
| 8 | 有 README | ✅ | 根目录 `README.md`（构建/运行/依赖/结果） |
| 9 | 预留融合量化与 Tensor Core 扩展接口 | ✅ | `normalize`/`dtype` 运行时参数；`--input_bin`；`launch_hadamard` 分发点（见 `docs/design.md` §5） |
| 10 | 结果可复现 | ✅ | 固定 `--seed`（默认 42），相同种子逐位一致 |

**核心结论**：以官方 `fast_hadamard_transform` 为基准的 36 配置验收
（2 dtype × 3 head_dim × 3 规模 × 归一化/非归一化）**全部通过且 100% bit-exact**，
归一化口径两种都与官方库一致，无论评测采用哪种约定均达标。

## 关于第 5 项的唯一例外（必须说明）

配置 `bf16 + head_dim=128 + outlier(scale=20)`：max_abs_error = 6.2e-2 > 5e-2。

- 已证明**非实现错误**：该配置 `max_rel_error = 3.89e-3`，恰等于 BF16 的理论舍入上界 2^-8；
  且该 token 的 GPU 输出与 Python 独立重算**逐位一致**。
- 根因：大 outlier 把输出量级抬到 [16,32)，BF16 的 half-ulp = 0.0625 本身 > 5e-2，
  任何正确实现都无法用绝对误差通过。
- 处理：报告以 ❌* 标注并在 `results/report.md` 附说明；已向助教提出"outlier 场景
  改用相对误差验收"的确认问题（`docs/questions_for_mentor.md` P1-8）。

## 复现命令

```bash
make                                   # 或 mkdir build && cd build && cmake .. && make -j
bash scripts/run_tests.sh              # 全量：内置自检正确性+性能；末尾自动跑官方库验收
~/hadamard_env/bin/python tests/test_vs_library.py        # 官方库验收（36 配置）
~/hadamard_env/bin/python tests/test_vs_library.py small  # 官方库验收（小规模冒烟）
```
