# 实验分析材料

2026-10-24/25 两次完整运行（S&P 500、CSI 300，模型 deepseek-flash 关闭思考，论文配置 20 轮）的追踪数据、分析结果和分析脚本。完整文字分析见 Claude 上的报告「AlphaPROBE 美股因子挖掘运行分析报告」。

## traces/ — Langfuse 导出的全部观测记录

| 文件 | 内容 |
| --- | --- |
| `alphaprobe-sp500-20261024-235057.json` | sp500 完整 20 轮（960 条） |
| `alphaprobe-csi300-20261025-080424.json` | csi300 第 1 轮（续跑前的会话） |
| `alphaprobe-csi300-20261025-091926.json` | csi300 第 2–20 轮（从检查点续跑后的会话） |

每条记录含 `input` / `output` / `metadata`：`generate-offspring` 是每次大模型调用的完整提示词与回复，`evaluate-offspring` 是每个候选的 IC、ICIR 与入库结果，`retrieve-parents` 是每轮选中的父因子。Langfuse 公钥字段已替换为 `<redacted>`。

## results/ — 计算结果

| 文件 | 内容 |
| --- | --- |
| `sp500_factor_metrics.json`、`csi300_factor_metrics.json` | 最终 50 个入库因子在训练/验证/测试期的 IC、ICIR、RankIC，量纲、谱系深度、父因子、解释文字，以及测试期两两相关矩阵 |
| `sp500_factor_table.md`、`csi300_factor_table.md` | 上面结果整理成的逐因子表 |
| `sp500_brain_style_backtest.json` | 仿 WorldQuant BRAIN 口径（delay 1、市场中性、8% 截断）的日频多空回测：夏普、换手率、Fitness |
| `paper_appendix_vs_csi300.json` | 论文附录 Table 4 的 28 个因子与本次 csi300 的 50 个因子，在同一份数据（`cn_data_2024`）、同样三个区间上的 IC/ICIR/RankIC，以及每个论文因子与本次因子池的最高测试期相关 |

## scripts/ — 生成上述结果的脚本

在仓库根目录用 `.venv\Scripts\python.exe` 运行，例如：

```
python analysis/scripts/analyze_health.py analysis/traces/alphaprobe-sp500-20261024-235057.json
python analysis/scripts/factor_metrics.py <运行的 log 目录> out.json csi300 C:/Users/<你>/.qlib/qlib_data/cn_data cuda:0
```

| 脚本 | 作用 |
| --- | --- |
| `dump_session.py` | 用 langfuse-cli 导出一个 session 的全部观测记录（需要 `.env` 里的 Langfuse 密钥） |
| `analyze_health.py` | 每轮的候选去向、解析失败、错误类型 |
| `analyze_explore.py` | 拒绝原因、父因子复用、重复结构、特征与主题变化 |
| `analyze_mods.py` | 子因子相对父因子的改动类型（只改窗口 / 包一层 / 部分改动 / 新思路） |
| `analyze_pool.py`、`dimcheck.py` | 从训练日志读取最终因子池并检查量纲 |
| `factor_metrics.py` | 从 `checkpoint.pt` 读取因子池，在三个区间计算逐因子指标（需要 qlib 数据和检查点） |
| `signif.py` | 训练期显著性（按 20 日重叠换算的 t 统计量）与样本外方向保持 |
| `wq_sim.py` | 仿 BRAIN 的日频多空回测 |
| `paper_vs_mine.py` | 不依赖 qlib、直接读 `.bin` 文件，在 CPU 上评估论文附录因子和本次因子池：`python -I analysis/scripts/paper_vs_mine.py <仓库根目录> <qlib 数据目录> out.json` |

运行日志在仓库根目录的 `logs/`，每次运行的因子池与指标在 `data/knowledge_logs/`（`checkpoint.pt` 超过 GitHub 单文件 100 MB 上限，不在仓库里）。
