# 仓库整理与归档

本次整理不改 kernel 数学或默认分发策略；将此前未提交的优化、验收、报告与清理
一起交付。原始计时和通过/失败判据保持不变。

整理后 `tensor_core/results` 从约 210 MiB 降至 12 MiB，主 `results` 从约 4.3 MiB
降至 993 KiB。14 项专项 CPU 测试、Slurm 语法和 diff 空白检查通过；仓库文档的
本地链接均可解析。图表重建后四份核心汇总 CSV 的 SHA256 与整理前完全一致。

## 当前目录职责

| 路径 | 内容 |
|---|---|
| `src/`、`include/` | 主项目 baseline、优化 FHT、warp 和融合 INT4 |
| `tensor_core/src/`、`include/` | WMMA / PTX / 精度诊断实验分支 |
| `tensor_core/scripts/` | 验收、计时、分析；`roofline_plot.py` 统一绘图实现 |
| `slurm/`、`tensor_core/slurm/` | 作业模板；一次性恢复脚本放在 `tensor_core/slurm/archive/` |
| `tests/` | CPU 工具测试与 GPU 参考验收入口 |
| `docs/` | 总结、历次审计、补测和归档记录 |
| `results/`、`tensor_core/results/` | 可审阅的实测证据，不携带原始大文件 |

[Tensor Core 结果索引](../tensor_core/results/README.md) 提供最终报告与原始 CSV 入口；
[Slurm 入口说明](../tensor_core/slurm/README.md) 区分通用脚本和历史恢复脚本。

## 清理范围与恢复

共归档 1903 个文件/链接，约 200.4 MiB，逐项记录在
[cleanup_manifest.csv](cleanup_manifest.csv)。移动前后校验 SHA256；原文件均保留，
没有不可恢复地删除实验数据。移除的空目录只用于整理已清空的结果/缓存路径。

归档根目录：

```text
/scratch/gz2522/gz2522/tmp/hadamard/runs/repo-cleanup-wSEn5pfh
```

归档文件位于 `files/<manifest 中的 path>`，保留原仓库相对路径。清理计划与实际
执行脚本也在归档根目录。原失败样本的五个符号链接指向前轮已迁移的 scratch
目录，本次连同链接一起归档；清单中链接 SHA256 是链接目标字符串的摘要。

| 类型 | 数量 | 保留在仓库的替代证据 |
|---|---:|---|
| 逐配置控制台日志 | 1320 | 每组完整 validation.csv / status.json / 负例检查 |
| 原始失败样本与链接 | 365 | 误差与输入模式记录、失败配置索引 |
| profiler / 张量 / 源码快照等二进制 | 84 | 原始计数器 CSV、环境、关键日志、图表 |
| H200 NCU 重复副本 | 61 | 唯一原始目录 `hopper_gaps_17291971/ncu` |
| 空日志 | 34 | 作业状态及其他非空日志 |
| Python 缓存 | 31 | 源码 |
| 已被最终报告替代的中间汇总 | 8 | `verify_17251073/final` 与 `final_library` |

恢复示例（在仓库根目录执行；避免覆盖现有文件）：

```bash
cp -an /scratch/gz2522/gz2522/tmp/hadamard/runs/repo-cleanup-wSEn5pfh/files/tensor_core/results/verify_17251073/progress \
  tensor_core/results/verify_17251073/
```

scratch 受集群保留/清理策略约束，不等于永久备份；如需长期保存原始 profiler
报告与失败张量，应另行转存。Git 中保留的汇总与脚本可用于审阅和重新采集。

## 去重后的图表重建

Hopper 的两条分析入口共用 `roofline_plot.py`；NCU 原始文件不再在后续计时目录
复制一份，使用 `summarize_hopper.py --profile-dir ...` 指向原始采集。具体命令见
[结果索引](../tensor_core/results/README.md#重建最新图表)。
