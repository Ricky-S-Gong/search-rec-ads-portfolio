# Goodreads MF 团队接手说明：Ricky、Yutao 与 Codex

本文是基于 Ziqi 已完成工作的执行手册。Ricky、Yutao 以及各自的 Codex 应先完整阅读，再修改代码。

## 1. 当前基础

Ziqi 已经冻结：

- 数据清洗与去重规则；
- k-core 和用户抽样规则；
- 连续 ID mapping；
- per-user temporal split；
- validation/test train-catalog 约束；
- Parquet 和 CSR artifact schema；
- SHA-256 manifest；
- Basic MF 和 FunkSVD；
- RMSE/MAE 正式结果。

对应 PR：

```text
https://github.com/Ricky-S-Gong/search-rec-ads-portfolio/pull/25
```

## 2. 不可修改的数据契约

数据版本：

```text
goodreads-poetry-v1
```

固定配置：

```json
{
  "version": "goodreads-poetry-v1",
  "seed": 20260830,
  "min_user_interactions": 20,
  "min_user_ratings": 10,
  "min_item_interactions": 10,
  "min_item_ratings": 3,
  "max_users": 5900,
  "train_fraction": 0.8,
  "validation_fraction": 0.1
}
```

任何成员都不能：

- 重新随机抽样用户；
- 重新执行 k-core；
- 重新 split；
- 更改 seed；
- 把 `rating == 0` 当成评分；
- 将 validation/test 数据加入训练；
- 在模型脚本中自行删除用户或图书；
- 将 Goodreads 数据提交到公开仓库。

如果确实需要新数据规则，必须生成新版本，例如 `goodreads-poetry-v2`，不能静默覆盖 v1。

## 3. 获得共享数据

Ziqi 本地生成的压缩包：

```text
goodreads-poetry-v1.tar.gz
size: 5.4 MB
sha256: f2543ec052d616b07983d3e29a1f46b71071dd52994d76df89e8d251f05265ed
```

该压缩包必须通过课程允许的受限共享存储提供，只对项目成员开放。不要放到公开 GitHub、公开 Drive 或网站 artifact。

下载以后，在仓库根目录解压到：

```text
research/goodbooks-mf/data/processed/goodreads-poetry-v1/
```

期望目录：

```text
goodreads-poetry-v1/
├── interactions.parquet
├── train.parquet
├── validation.parquet
├── test.parquet
├── user_mapping.parquet
├── item_mapping.parquet
├── train_explicit.npz
├── train_implicit.npz
└── manifest.json
```

## 4. 每个人开始开发前必须执行

```bash
git checkout main
git pull --ff-only
uv sync --locked
uv run python research/goodbooks-mf/verify_data.py
```

成功输出必须包括：

```json
{
  "version": "goodreads-poetry-v1",
  "counts": {
    "interactions": 261709,
    "items": 5551,
    "test": 44229,
    "train": 191436,
    "users": 5129,
    "validation": 26044
  }
}
```

如果 checksum mismatch：停止开发，不要重新预处理。重新从团队共享位置下载正确 bundle。

验证脚本还会将 bundle 内的 manifest 与仓库提交的 `canonical_manifest.json` 完整比较。因此自行重跑 preprocessing 得到的另一套自洽数据也不能冒充团队冻结版本。

## 5. 文件 SHA-256

`verify_data.py` 会自动验证以下值：

| File | SHA-256 |
|---|---|
| interactions.parquet | `df6d97c2b376e74ffae21c78483d2864bcdaa18b692ee15bf838161bc4cb89bd` |
| train.parquet | `7f07d6add39027e19f7d93dacbe2be4c31bb036a6b500bacf76b80c623ab57cd` |
| validation.parquet | `40ab726437afff933583dc40ffe8aa9a9a6e1af6da786a3b8691c9bd1216a6da` |
| test.parquet | `5f5384a34890043c33d7570ceda8b0f65a1cd2d2152ff12351020aa1f045ac38` |
| user_mapping.parquet | `44a99125f04a2a1dc1c26a0c12fe9365978d685f6d68e3fceea72d4871c09aac` |
| item_mapping.parquet | `61af10d7c43d394a1c1815d5ca92959f082fd8e3e716c8db665d408e6eba5832` |
| train_explicit.npz | `6c13b33285b4692a60caad3733a96c2a4670bf0d15a50f50a2cba9cc2659e0d7` |
| train_implicit.npz | `cfa9fafc1c87d5c2f5e17cf3f082bbf92f82dd051c6c5f54528ee3334aa54eb3` |

## 6. 如何读取数据

### 读取长表

```python
from pathlib import Path
import pandas as pd

DATA_DIR = Path("research/goodbooks-mf/data/processed/goodreads-poetry-v1")

train = pd.read_parquet(DATA_DIR / "train.parquet")
validation = pd.read_parquet(DATA_DIR / "validation.parquet")
test = pd.read_parquet(DATA_DIR / "test.parquet")

train_ratings = train[train["rating"] > 0]
validation_ratings = validation[validation["rating"] > 0]
test_ratings = test[test["rating"] > 0]
```

### 读取 CSR

```python
from scipy import sparse

explicit_matrix = sparse.load_npz(DATA_DIR / "train_explicit.npz")
implicit_matrix = sparse.load_npz(DATA_DIR / "train_implicit.npz")

assert explicit_matrix.shape == (5129, 5551)
assert explicit_matrix.nnz == 117397
assert implicit_matrix.shape == (5129, 5551)
assert implicit_matrix.nnz == 119942
```

### 读取 mapping

```python
users = pd.read_parquet(DATA_DIR / "user_mapping.parquet")
items = pd.read_parquet(DATA_DIR / "item_mapping.parquet")
```

## 7. 字段语义

| Field | Meaning | Use |
|---|---|---|
| user_idx | 0–5128 连续下标 | 所有模型 |
| item_idx | 0–5550 连续下标 | 所有模型 |
| rating | 0 或 1–5 | `>0` 才是显式评分 |
| is_read | 用户是否读过 | SVD++ implicit set |
| is_reviewed | 用户是否评论 | SVD++ implicit set |
| event_time | UTC timestamp | split 已冻结，不应重切 |
| split | train/validation/test | 训练与评估边界 |

统一隐式反馈：

```python
implicit = frame[frame["is_read"] | frame["is_reviewed"] | frame["rating"].gt(0)]
```

## 8. 统一模型 API

团队模型应尽量符合：

```python
model.fit(train, validation=None)
model.predict(user_idx, item_idx)
model.recommend(
    user_idx,
    candidate_item_idxs,
    seen_item_idxs=(),
    k=10,
)
```

约定：

- `predict` 返回 raw score。
- RMSE/MAE 评估时把评分预测 clip 到 `[1, 5]`。
- Top-K 排序使用 raw score，不能先 clip。
- recommend 必须排除训练期已见 item。
- 分数完全相同时以 `item_idx` 升序作为最终 deterministic tie-break。

已有参考：

```text
goodbooks_mf/models.py::BasicMF
goodbooks_mf/models.py::FunkSVD
```

## 9. Ricky 的任务：统一 evaluation

### 9.1 需要交付的函数

建议位置：

```text
research/goodbooks-mf/goodbooks_mf/evaluation.py
```

需要实现：

```python
rmse(actual, predicted)
mae(actual, predicted)
precision_at_k(recommended, relevant, k)
recall_at_k(recommended, relevant, k)
ndcg_at_k(recommended, relevant, k)
evaluate_ratings(model, test)
evaluate_ranking(model, candidate_set, k_values=(5, 10, 20))
```

### 9.2 固定 relevance

使用项目计划约定：

```text
relevant = rating >= 4 OR (rating == 0 AND is_read == True)
```

建议在主结果中同时记录 relevant 定义，防止结果脱离语义。

### 9.3 固定候选集

需要由 Ricky 一次性生成并冻结：

```text
evaluation_candidates.parquet
```

建议字段：

```text
user_idx, item_idx, label, source
```

其中：

- positive：该用户 test relevant item；
- negative：从 train catalog 中采样的未见且非 test-positive item；
- 所有模型使用完全相同的 user/item pairs；
- seed 必须是 `20260830` 或在统一配置中单独冻结；
- 候选集中排除训练期已见 item；
- 每个用户使用相同数量和相同 ID 的 negatives；
- candidate artifact 也要加入 checksum manifest。

如果计算允许，优先使用 full train catalog minus seen；如果必须 sampled negatives，应在报告中明确这会使指标依赖采样策略。

### 9.4 指标细节

Precision@K：

```text
Top-K 中 relevant item 数 / K
```

Recall@K：

```text
Top-K 中 relevant item 数 / 该用户所有 candidate relevant item 数
```

NDCG@K：使用 binary relevance 时：

```text
DCG = Σ rel_rank / log2(rank + 1)
NDCG = DCG / IDCG
```

没有 relevant candidate 的用户不应强行记为 0；应从该指标分母排除，并报告 evaluated user count。

同样地，正式数据有 1 位用户在 cold-item removal 后没有 validation 显式评分。rating evaluation 必须按实际有效记录计算，并报告 evaluated rating/user count，不能假设每个 split 中每位用户都有显式评分。

### 9.5 Ricky 的 SVD++

训练只能使用 train split。

用户隐式历史：

```text
N(u) = train 中 is_read OR is_reviewed OR rating > 0 的 items
```

预测：

```text
μ + b_u + b_i + q_iᵀ[p_u + |N(u)|^-1/2 Σ y_j]
```

不能使用 validation/test interactions 构造 `N(u)`，否则产生 leakage。

## 10. Yutao 的任务：ALS

建议位置：

```text
research/goodbooks-mf/goodbooks_mf/als.py
```

输入使用：

```text
train_explicit.npz
```

只对 observed ratings 优化。不要把 CSR 中缺失位置当成 rating 0。

ALS 更新：

```text
p_u = (Q_Iᵀ Q_I + λI)^-1 Q_Iᵀ r_u
q_i = (P_Uᵀ P_U + λI)^-1 P_Uᵀ r_i
```

实现要求：

- 使用 `np.linalg.solve`，不要显式计算 matrix inverse；
- user update 使用 CSR row access；
- item update 使用 CSC 或 transpose CSR；
- validation 选 `n_factors`、`reg_lambda`、iterations；
- 最佳配置冻结后才运行 test；
- 输出训练时间和峰值内存或至少矩阵尺寸。

推荐小网格：

```text
n_factors: [20, 40, 80]
reg_lambda: [0.01, 0.02, 0.05, 0.1]
iterations: [5, 10, 20]
```

可先 smoke，再缩小正式网格。

## 11. Yutao 的任务：masked NMF

建议位置：

```text
research/goodbooks-mf/goodbooks_mf/nmf.py
```

核心约束：

```text
P >= 0
Q >= 0
```

重要：普通 sklearn NMF 会将传入矩阵中的 0 当成观测值 0。对本项目来说 CSR 里的 0 是 missing，不是低评分。因此必须实现 masked loss 或只遍历 observed entries。

目标函数：

```text
Σ_(u,i in Ω) (r_ui - p_u·q_i)² + regularization
```

验收测试必须证明：向矩阵加入大量未观测位置不会改变 observed-only loss 的定义。

## 12. Yutao 的任务：统一 experiment runner

建议位置：

```text
research/goodbooks-mf/run_all_experiments.py
research/goodbooks-mf/experiment_config.json
```

runner 顺序：

1. `verify_bundle()`；
2. 读取 frozen train/validation/test；
3. 在 train 上训练每个 candidate config；
4. 用 validation 选择最佳 config；
5. 冻结选择结果；
6. 最佳 config 只运行一次 test；
7. 调用 Ricky 的共享 rating/ranking evaluation；
8. 写统一 results rows；
9. 输出模型对比表和图表。

统一结果字段建议：

```text
dataset_version
model
seed
n_factors
learning_rate
reg_lambda
iterations_or_epochs
best_validation_metric
rmse
mae
precision_at_5
precision_at_10
precision_at_20
recall_at_5
recall_at_10
recall_at_20
ndcg_at_5
ndcg_at_10
ndcg_at_20
evaluated_rating_count
evaluated_ranking_users
training_seconds
inference_seconds
```

## 13. 不允许的数据泄漏

以下都属于 leakage：

- 用 test RMSE 选超参数；
- 看过 test NDCG 后调整 candidate policy；
- 使用 validation/test 建 SVD++ 用户隐式历史；
- 用全数据计算模型 bias 或 global mean；
- 在所有 split 上重新做 item popularity 并用于 tie-break；
- 根据 test 用户表现删除“难用户”；
- 重新 split 直到结果更好。

允许使用：

- train 训练模型；
- validation 选超参数和 early stopping；
- test 在所有决策冻结后做最终一次评估。

## 14. 测试要求

每个新行为先写 failing test，再写实现。

Ricky 至少需要测试：

- RMSE/MAE 手算样例；
- Precision/Recall/NDCG 手算样例；
- 无 relevant user 的处理；
- seen-item exclusion；
- candidate set deterministic；
- 所有模型使用相同 candidates；
- SVD++ 不读取 future history。

Yutao 至少需要测试：

- ALS 一个 update 的 shape 和 finite values；
- `np.linalg.solve` 的结果降低 observed loss；
- NMF factors 始终非负；
- NMF 未观测值不进入 loss；
- runner 只在 validation 选参；
- test 只在配置冻结后调用；
- results schema 完整。

完成后运行：

```bash
uv run pytest
```

如果本地 pytest launcher 有环境问题，可以执行：

```bash
.venv/bin/python research/run_python_tests.py
```

但 PR 的 GitHub CI 必须成功。

## 15. 推荐协作顺序

1. 合并 Ziqi 的 PR #25。
2. 三个人下载私有 bundle 并通过 checksum。
3. Ricky 先冻结 evaluation API、relevance 和 candidate artifact。
4. Yutao 同时实现 ALS/NMF，但暂时只用 RMSE/MAE 调试。
5. Ricky 实现 SVD++。
6. 三个负责人将模型接到相同 evaluation API。
7. Yutao 用统一 runner 重跑所有最佳 validation configs。
8. 全员审查结果 schema、数据版本和候选集 hash。
9. 最后制作结果表、图和面试讲解。

## 16. 给 Ricky 的 Codex 任务说明

可直接把下面内容作为任务背景：

```text
在 research/goodbooks-mf 中继续 Goodreads Poetry MF 项目。先阅读
TEAM_HANDOFF_RICKY_YUTAO.md、ZIQI_IMPLEMENTATION_GUIDE.md、config.json、
manifest.json 和现有 tests。不要修改 preprocessing、split、ID mapping、
seed 或 goodreads-poetry-v1 数据。

你的任务是：
1. 先用测试定义统一 RMSE、MAE、Precision@K、Recall@K、NDCG@K。
2. 固定 test users、relevance 规则和所有模型共享的 candidate set。
3. candidate artifact 必须确定性生成并带 checksum。
4. 实现 SVD++，隐式历史只能来自 train split。
5. 接入现有 BasicMF/FunkSVD API。
6. 输出统一 results row，不要根据 test 调参。
7. 运行仓库全套测试并更新文档。

遇到不明确的 evaluation 规则时先在 PR/项目计划中确认，不能自行修改数据。
```

## 17. 给 Yutao 的 Codex 任务说明

```text
在 research/goodbooks-mf 中继续 Goodreads Poetry MF 项目。先阅读
TEAM_HANDOFF_RICKY_YUTAO.md、ZIQI_IMPLEMENTATION_GUIDE.md、config.json、
manifest.json 和现有 tests。不要修改 preprocessing、split、ID mapping、
seed 或 goodreads-poetry-v1 数据。

你的任务是：
1. 先用测试实现 sparse explicit-feedback ALS。
2. 实现 masked NMF，未观测 CSR 位置绝不能作为 rating 0 进入 loss。
3. 建立统一 experiment runner 和 config schema。
4. 所有超参数只使用 train/validation 选择，test 只运行最终冻结配置。
5. 接入 Ricky 的共享 evaluation functions 和 candidate set。
6. 输出统一结果表、训练时间和图表所需数据。
7. 运行仓库全套测试并更新文档。

若共享 evaluation 尚未合并，可以先实现模型和 validation RMSE/MAE，
但不要复制一套临时 Top-K 指标成为最终接口。
```

## 18. 完成定义

团队项目只有在以下条件都满足时才算完成：

- 三个人的 `verify_data.py` 全部成功；
- 所有模型使用相同 dataset version 和 split；
- 所有模型使用相同 test users 和 candidates；
- 五个模型都输出 RMSE、MAE、Precision@K、Recall@K、NDCG@K；
- validation 选择和最终 test 评估分离；
- GitHub CI 成功；
- 公开仓库不含 Goodreads 原始或处理后记录；
- 结果表记录 dataset version、seed、参数和训练时间。
