# Goodreads Poetry：矩阵分解与统一评估

使用 UCSD Goodreads Book Graph 的 Poetry 子集，比较 Basic MF、FunkSVD、显式 ALS、Masked NMF 与 SVD++；Bias-aware residual ALS 是额外诊断模型。当前已提交的 `team_model_comparison.json` 为 **complete**，包含全部五个计划模型及诊断模型，前端聚合产物已发布。此前“等待 SVD++ 结果”的说明已过时。

[项目计划](docs/project-plan.md) · [实现说明](docs/implementation-guide.md) · [数据交接](docs/team-handoff.md) · [当前比较 JSON](results/team_model_comparison.json) · [网站产物](../../public/artifacts/goodbooks/metrics.json)

## Frozen data contract / 数据协议

`config.json` 与 `canonical_manifest.json` 固定数据版本 `goodreads-poetry-v1`、seed `20260830`、过滤、用户规模、时间切分及文件校验值。所有模型读取同一 bundle，不各自过滤或划分。

当前 bundle：5,129 用户、5,551 物品、261,709 交互；train / validation / test 分别 191,436 / 26,044 / 44,229。显式评分测试为 17,168 条，排序评估 4,765 用户。

字段：`user_idx, item_idx, rating, is_read, is_reviewed, event_time, split`。`rating>0` 是显式评分；隐式反馈为 `is_read OR is_reviewed OR rating>0`。按用户时间划分，并使用显式评分边界确保可评估；缺失矩阵位置不是零分。

## Current results / 已冻结测试结果

| 模型 | RMSE | MAE | Recall@10 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| Basic MF | 1.149409 | 0.882003 | 0.036178 | 0.023947 |
| FunkSVD | 0.861341 | 0.668944 | 0.007369 | 0.004920 |
| ALS | 1.119088 | 0.828761 | 0.005637 | 0.002728 |
| NMF + L2 | 0.984062 | 0.719640 | 0.001193 | 0.000835 |
| SVD++ | 0.860568 | 0.666782 | 0.007855 | 0.005447 |
| Bias-aware ALS | 0.861570 | 0.675045 | 0.007762 | 0.005978 |

表格来自当前汇总 JSON，均使用相同数据人群和候选协议。Basic MF 排序指标最好，SVD++ 的评分误差最低；评分预测与推荐排序不是同一个目标。前端按已观察的 NDCG@10 展示 Basic MF，这属于离线结果展示，不是再用测试集调参。测试结果不用于继续选择超参数。

## Models and evaluation / 实现边界

Basic MF、FunkSVD 与 SVD++ 训练观测评分；ALS、NMF、Bias-aware ALS 使用稀疏显式矩阵。FunkSVD 加全局／用户／物品偏置；SVD++ 隐式历史仅来自 train。SGD 以验证 RMSE early stopping 并恢复最佳 epoch。

统一评分指标在计算时裁剪到 `[1,5]`，排序使用原始分数。每个合格测试用户对完整训练候选库排序，排除其训练已看物品；相关性为 `rating>=4 OR (rating==0 AND is_read)`，同分按 `item_idx` 升序，报告 K=5/10/20 的 Precision、Recall、NDCG。无采样负例。共享实现位于 `goodbooks_mf/evaluation.py`。

## Reproduction / 环境与流程

从仓库根目录使用根 Python 环境：

```bash
uv sync --locked
# 取得许可并准备数据后，核验冻结 bundle：
uv run --locked python research/goodbooks-mf/verify_data.py
uv run --locked pytest research/goodbooks-mf/tests -q
```

阅读已提交结果或运行前端不需要私有 bundle。重现实验才需要原始数据或团队 canonical bundle；原始下载与预处理入口分别为 `download_data.py`、`prepare_data.py`，以 `verify_data.py` 成功为前提。

| 阶段 | 命令入口 | 约束 |
| --- | --- | --- |
| 验证候选搜索 | `run_all_experiments.py` | 只反序列化 train/validation；候选由 `experiment_config.json` 固定 |
| 仅搜索 SVD++ | `run_all_experiments.py --models svdpp --output /path/to/new-selection.json` | 同一验证协议，独立输出 |
| 冻结 ALS / NMF / Bias-aware ALS 测试 | `run_frozen_test.py`, `run_unified_evaluation.py` | 已完成；旧冻结结果不可覆盖，不用于新调参 |
| SVD++ 最终测试 | `run_svdpp_evaluation.py` | 已完成；恢复最佳验证 checkpoint 后一次性测试 |
| 汇总 | `build_team_summary.py` | 核对所有模型的版本、seed、协议、人群与状态 |
| 前端发布 | `publish_frontend_artifacts.py` | bundle 核验 + complete 汇总后写聚合 JSON |

例如，在已核验且冻结结果匹配的环境中重建展示产物：

```bash
uv run --locked python research/goodbooks-mf/build_team_summary.py
uv run --locked python research/goodbooks-mf/publish_frontend_artifacts.py
```

`run_experiment.py` 是早期 Basic MF / FunkSVD 的冻结测试复现入口，不是新候选验证入口。冻结测试脚本拒绝覆盖已有配置／结果；不要为了重跑删除冻结证据。若改数据或调新模型，须建立新的版本和输出，保持测试集不参与选型。

## Storage / 存储与许可

`data/` 中的原始数据、Parquet 和 CSR 不提交。`results/` 保存冻结配置、验证选择和最终聚合结果；`public/artifacts/goodbooks/metrics.json` 保存前端所需计数、字段、协议和模型汇总，不导出用户／物品 ID、交互或逐用户榜单。

UCSD 数据仅供学术使用，不重新分发或商业使用；下载权限和完整流程见交接文档。历史指南若仍描述 pending 状态，以当前汇总 JSON 和本 README 为准。
