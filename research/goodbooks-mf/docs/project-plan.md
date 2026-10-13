# Matrix Factorization Project Plan

## 1. Project Overview

使用 Goodreads Book Graph 的 Poetry 子集，实现链接章节中需要落地的五个模型：Basic MF、FunkSVD、ALS、NMF 和 SVD++。Time-aware MF 在原文中只有概念介绍，没有实现细节，因此不纳入项目。

所有模型共享同一份预处理数据、split、evaluation functions 和实验配置。

## 2. Algorithms to Implement

| Algorithm | Core Idea | Feedback | Difficulty |
|---|---|---|---|
| Basic MF | 用 SGD 学习 user/item latent factors | Explicit | Medium |
| FunkSVD | Basic MF 加 global/user/item bias | Explicit | Medium |
| ALS | 交替最小二乘更新 user/item factors | Explicit | Medium–Hard |
| NMF | 对 latent factors 加非负约束 | Explicit | Medium |
| SVD++ | 在 FunkSVD 中加入用户历史隐式交互 | Explicit + Implicit | Hard |

## 3. Team Responsibility Overview

| Member | Main Responsibility | Algorithms | Other Responsibility |
|---|---|---|---|
| Ricky | 高级 MF | SVD++ | 维护共享 evaluation functions；评估 SVD++ |
| Ziqi | 数据处理与 SGD 基础模型 | Basic MF、FunkSVD | 下载、清洗、过滤、split；评估自己的模型 |
| Yutao | 线性代数模型与实验汇总 | ALS、NMF | 统一实验 runner；评估自己的模型并汇总结果 |

## 4. Individual Task Lists

### Ricky

- 实现统一的 RMSE、MAE、Precision@K、Recall@K 和 NDCG@K functions。
- 固定 Top-K 候选集和负采样规则，供三人共同调用。
- 实现 SVD++ 的 bias、implicit factors 和 SGD training loop。
- 调参 latent dimension、learning rate 和 regularization。
- 使用共享 evaluation 评估 SVD++，输出统一 results row。

### Ziqi

- 下载并流式读取 Goodreads Poetry books/interactions 数据。
- 清洗重复记录、非法评分、缺失 ID 和时间字段。
- 完成 k-core filtering、连续 ID 映射及统一 split。
- 输出标准 Parquet 和 CSR sparse matrices。
- 实现 Basic MF 的 SGD、正则化和 early stopping。
- 在 Basic MF 上加入 bias，实现 FunkSVD。
- 使用共享 evaluation 评估两个模型，输出统一 results rows。

### Yutao

- 实现 sparse explicit-feedback ALS。
- 实现 masked NMF，确保未观测值不被当作评分 0。
- 建立统一 experiment runner 和配置读取逻辑。
- 使用共享 evaluation 评估 ALS 和 NMF。
- 汇总所有模型的指标、训练时间和结果图表。

## 5. Shared Data Pipeline

### Subset choice

选择官方 **Poetry subset**：原始规模约 36,514 本书、273 万条交互，是官方 genre subsets 中较小的一组，同时包含评分和 `is_read/is_reviewed`，适合本地实验。

推荐缩减流程：

1. 只保留 `rating` 合法且 user/book ID 完整的记录。
2. `(user_id, book_id)` 重复时保留信息最完整、时间最新的一条。
3. 迭代过滤，直到稳定：
   - user 至少 20 条交互、5 条显式评分；
   - book 至少 10 条交互、3 条显式评分。
4. 如果仍过大，用固定 seed 采样最多 8,000 名合格用户，再执行一次过滤。
5. 目标规模：5,000–8,000 users、5,000–10,000 books、150,000–300,000 interactions。
6. 按每位用户的交互时间做 80%/10%/10% train/validation/test split。
7. validation/test 中的 user 和 book 必须在 train 中出现。

标准字段：

```text
user_idx, item_idx, rating, is_read, is_reviewed, event_time, split
```

- `rating > 0`：用于 explicit training 和 RMSE/MAE。
- `is_read/is_reviewed/rating > 0`：作为 SVD++ 的隐式交互集合。
- 所有人直接读取同一份 processed data，禁止自行重新过滤或 split。

## 6. Evaluation Plan

| Metric | Purpose | Notes |
|---|---|---|
| RMSE | 评分预测误差 | 只对 test 中有效评分计算 |
| MAE | 平均绝对评分误差 | 与 RMSE 使用相同记录 |
| Precision@K | Top-K 准确性 | K = 5、10、20 |
| Recall@K | 相关图书召回率 | 排除 train 已见物品 |
| NDCG@K | 推荐排序质量 | 重点报告 NDCG@10 |

统一规则：

- 所有模型都报告五项指标，方便横向比较。
- Top-K relevant item 定义为 `rating >= 4`，或无评分但 `is_read=True`。
- 所有模型使用同一批 test users、candidate items 和 sampled negatives。
- Ricky 维护公共指标代码；每个人负责调用它评估自己的模型，而不是重复实现公式。

## 7. Repository Structure

```text
project/
├── data/               # Ziqi
├── preprocessing/      # Ziqi
├── models/             # 各算法负责人
├── evaluation/         # Ricky
├── configs/            # Yutao
├── experiments/        # Yutao
├── results/            # Yutao
├── tests/               # 全员
└── README.md
```

## 8. Integration Plan

1. Ziqi 先冻结 processed data 和 split，其他模型依赖该输出。
2. 三人统一 `fit()`、`predict()` 和 `recommend()` API。
3. Ricky 冻结 evaluation functions 和 candidate set。
4. 每个人运行并检查自己模型的五项统一指标。
5. Yutao 用统一 runner 重跑所有最佳 validation 配置。
6. 最终输出模型对比表，以及评分误差、Top-K 指标和训练时间三类图表。
