# Ziqi 的 Goodreads Poetry Matrix Factorization 实施说明

本文记录 Ziqi 负责部分的完整实施过程。目标是让没有参与开发的人也能理解项目为什么这样设计、怎样复现，以及面试时如何解释技术选择。

## 1. 完成状态审计

### 已完成

- 下载官方 Goodreads Poetry books 与 interactions 数据。
- 流式读取压缩 JSON，避免解压出巨大的中间文本文件。
- 清理缺失 ID、非法评分与无效时间。
- 对 `(user_id, book_id)` 做确定性去重。
- 进行迭代 k-core filtering，直到用户和图书约束同时稳定。
- 使用固定 seed 抽样，并再次执行 k-core。
- 将字符串 ID 映射为连续整数 ID。
- 完成按用户时间顺序的 train/validation/test split。
- 移除 validation/test 中训练阶段未出现的图书。
- 输出 Parquet、显式评分 CSR 和隐式交互 CSR。
- 生成带 SHA-256 的 manifest，并提供跨机器验证命令。
- 实现 Basic MF：SGD、L2 regularization、validation early stopping。
- 实现带 global/user/item bias 的 FunkSVD。
- 在正式 Poetry v1 数据上运行两个模型，保存 RMSE、MAE、最佳 epoch 和训练时间。
- 将 GoodBooks 测试接入仓库统一测试入口；本地 40 项测试和 GitHub CI 均通过。

### 等待团队集成

- 统一 evaluation functions 和全训练目录候选协议已经实现；私有 bundle
  到位并验证后，需要重跑 Basic MF/FunkSVD 以生成正式 Top-K 指标。
- 处理后的数据不能上传到公开 GitHub。需要课程允许的受限共享空间，才能完成三人访问配置。

因此，Ziqi 可以独立完成的工程和模型工作已经完成；正式 Top-K 结果与
私有数据分发是尚未闭环的外部依赖。

## 2. 问题定义

项目使用 [UCSD Goodreads Book Graph](https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html) 的 Poetry 子集。原始子集约有 36,514 本书和 2,734,350 条交互。

项目需要同时支持两种反馈：

- 显式反馈：`rating` 为 1–5，用于 Basic MF、FunkSVD、ALS、NMF、SVD++ 的评分训练和 RMSE/MAE。
- 隐式反馈：用户读过、评论过或评分过一本书，用于 SVD++ 的历史集合。

统一定义如下：

```text
explicit = rating > 0
implicit = is_read OR is_reviewed OR rating > 0
```

`rating == 0` 表示没有评分，不是用户打了 0 分。把它当成负反馈会系统性破坏模型。

## 3. 完整数据流程

```text
官方 .json.gz
    ↓ 流式读取
SQLite staging
    ↓ 清洗与全局确定性去重
迭代 k-core
    ↓ 固定 seed 用户抽样
第二次迭代 k-core
    ↓ 连续 ID 映射
按用户时间切分
    ↓ 去除 validation/test cold items
重新连续编码
    ↓
Parquet + CSR + manifest
    ↓ checksum verification
所有模型共享同一输入
```

核心原则是：数据只处理和切分一次。其他成员不得在自己的模型脚本里重新过滤、抽样或 split。

## 4. 环境复现

仓库使用 Python 3.11–3.13，并通过 `uv.lock` 固定依赖版本。Parquet 由 PyArrow 读写。

```bash
git clone https://github.com/Ricky-S-Gong/search-rec-ads-portfolio.git
cd search-rec-ads-portfolio
uv sync --locked
```

为什么要提交 lock file：

- 三台机器安装相同版本的 NumPy、Pandas、SciPy、PyArrow 和 pytest。
- 避免库升级造成随机数、Parquet schema 或数值计算差异。
- CI 与本地使用同一依赖图。

## 5. Step 1：下载并验证官方源文件

执行：

```bash
uv run python research/goodbooks-mf/download_data.py
```

脚本下载：

```text
goodreads_interactions_poetry.json.gz
goodreads_books_poetry.json.gz
```

下载完成后生成 `source_manifest.json`，记录源文件 SHA-256。数据位于：

```text
research/goodbooks-mf/data/raw/
```

该目录已被 Git 忽略。Goodreads 数据仅限学术用途，且提供方要求不要再分发或用于商业用途。

## 6. Step 2：流式读取与 SQLite staging

实现位置：

```text
goodbooks_mf/data.py::iter_json_gzip
goodbooks_mf/data.py::stage_to_sqlite
```

不直接执行 `pd.read_json` 的原因：原始文件有约 273 万条交互，一次性构造大量 Python 字典和 DataFrame 会造成较高峰值内存。

当前做法：

1. 使用 `gzip.open(..., "rt")` 逐行读取。
2. 每行解析一个 JSON object。
3. 清洗后按 batch 写入 SQLite。
4. 使用 `(user_id, item_id)` 作为 primary key，在磁盘上完成全局去重。

SQLite staging 每次重建数据表，避免源文件发生变化后保留旧记录。

## 7. Step 3：数据清洗

实现位置：

```text
goodbooks_mf/data.py::_clean_record
goodbooks_mf/data.py::normalize_interactions
```

每条记录执行以下检查：

1. `user_id` 必须存在且非空。
2. `book_id` 必须存在且非空，并统一改名为 `item_id`。
3. `rating` 必须可以转换为整数且位于 `[0, 5]`。
4. 时间依次尝试：`date_updated`、`read_at`、`date_added`、`event_time`。
5. 时间必须成功解析，并统一转换为 UTC。
6. `is_read` 与 `is_reviewed` 统一转换为 boolean。

最终时间范围为：

```text
2007-02-04 04:43:16 UTC
到 2017-11-05 18:45:50 UTC
```

## 8. Step 4：确定性去重

虽然官方 genre 文件通常已经接近去重状态，pipeline 仍显式处理重复 `(user_id, item_id)`。

优先级为：

1. 信息更完整的记录优先；
2. 完整度相同时，更新时间更晚的记录优先。

完整度分数为：

```text
(rating > 0) + is_read + is_reviewed
```

SQLite 使用 UPSERT 实现相同规则，因此无论输入顺序怎样，最终保留规则都固定。正式产物中的重复数为 0。

## 9. Step 5：迭代 k-core filtering

配置位于 `config.json`：

```json
{
  "min_user_interactions": 20,
  "min_user_ratings": 10,
  "min_item_interactions": 10,
  "min_item_ratings": 3
}
```

不能只过滤一次。删除低活跃图书以后，一些用户可能不再满足阈值；删除用户以后，一些图书也可能不再满足阈值。因此算法必须交替过滤用户与图书，直到行数不再变化。

伪代码：

```text
repeat:
    remove users below interaction/rating thresholds
    remove items below interaction/rating thresholds
until number of rows no longer changes
```

面试时应强调：这是 fixed-point k-core，不是单次 `groupby().filter()`。

## 10. Step 6：固定 seed 抽样

全局 k-core 后，如果用户仍然过多，则从排序后的用户列表中使用固定 RNG 抽样。

```text
seed = 20260830
max_users = 5900
```

抽样以后再次执行 k-core，因为用户抽样会降低图书支持度。

最终得到：

| 指标 | 数量 |
|---|---:|
| Users | 5,129 |
| Books | 5,551 |
| Interactions | 261,709 |

这个规模满足项目计划约 5k–8k users、150k–300k interactions 的目标，同时允许五个模型在普通 CPU 上实验。

## 11. Step 7：连续 ID mapping

原始 Goodreads ID 是字符串，不适合直接用作数组下标。

处理方法：

1. 对原始 ID 排序。
2. 用户映射到 `[0, n_users)`。
3. 图书映射到 `[0, n_items)`。
4. 输出双向可追踪的 mapping tables。

文件：

```text
user_mapping.parquet: user_id, user_idx
item_mapping.parquet: item_id, item_idx
```

排序后编码使 mapping 与数据读取顺序无关，增强可复现性。

## 12. Step 8：按用户时间切分

实现位置：

```text
goodbooks_mf/split.py::per_user_temporal_split
```

每位用户按：

```text
event_time, item_idx
```

稳定排序。切分比例为：

```text
train = 80%
validation = 10%
test = 10%
```

切分边界使用显式评分的位置，而不是简单按所有 interaction 行号切。边界生成时会为每个用户的三部分分配显式评分；随后执行 cold-item removal 时，正式数据中有 1 位用户的 validation 显式评分被移除。统一 rating evaluation 应按实际含显式评分的记录和用户报告 evaluated count。

时间切分比随机切分更符合真实推荐场景：模型只能使用过去预测未来，降低 temporal leakage。

切分以后：

- validation/test 中只保留 train catalog 已出现的 item。
- 正式数据中 validation cold items = 0。
- 正式数据中 test cold items = 0。
- 5,129 位用户中，5,128 位在 validation 有显式评分；所有用户在 train 和 test 都有显式评分。
- 过滤后重新编码，保证整数 index 无空洞。

正式 split：

| Split | 所有交互 | 显式评分 |
|---|---:|---:|
| Train | 191,436 | 117,397 |
| Validation | 26,044 | 14,601 |
| Test | 44,229 | 17,168 |

## 13. Step 9：共享数据格式

标准长表 schema：

| 字段 | 类型 | 含义 |
|---|---|---|
| `user_idx` | int32 | 连续用户下标 |
| `item_idx` | int32 | 连续图书下标 |
| `rating` | int64 | 0 表示无评分；1–5 为显式评分 |
| `is_read` | bool | 是否读过 |
| `is_reviewed` | bool | 是否写过评论 |
| `event_time` | UTC datetime | 统一事件时间 |
| `split` | string | train/validation/test |

产物：

```text
interactions.parquet
train.parquet
validation.parquet
test.parquet
user_mapping.parquet
item_mapping.parquet
train_explicit.npz
train_implicit.npz
manifest.json
```

CSR 矩阵：

| Matrix | Shape | Non-zero | Values |
|---|---:|---:|---|
| train_explicit | 5129 × 5551 | 117,397 | rating 1–5 |
| train_implicit | 5129 × 5551 | 119,942 | binary 1 |

## 14. Step 10：manifest 与跨机器验证

bundle 内的 `manifest.json` 保存：

- dataset version；
- seed；
- 所有过滤和 split 配置；
- users/items/interactions/split counts；
- 每个 Parquet/NPZ 文件的 SHA-256。

仓库另外提交只含元数据和 hash 的 `canonical_manifest.json`。验证不仅检查本地文件与本地 manifest，还要求本地 manifest 与仓库 canonical manifest 完全一致，防止某台机器重新生成了一套自洽但不同的数据。

验证命令：

```bash
uv run python research/goodbooks-mf/verify_data.py
```

任何文件丢失或字节发生变化都会抛出 checksum mismatch。这样可以证明三个人实际使用的是同一份数据，而不只是“配置看起来一样”。

## 15. Step 11：Basic Matrix Factorization

实现位置：

```text
goodbooks_mf/models.py::BasicMF
```

模型：

```text
prediction(u, i) = p_u · q_i
```

目标函数：

```text
Σ(r_ui - p_u·q_i)² + λ(||p_u||² + ||q_i||²)
```

只遍历 observed rating triplets，不扫描完整 dense user-item matrix。

SGD 更新：

```text
error = rating - prediction
p_u ← p_u + lr × (error × q_i - λ × p_u)
q_i ← q_i + lr × (error × old_p_u - λ × q_i)
```

更新 item vector 时使用更新前的 user vector，避免同一步梯度被新的 user vector 污染。

## 16. Step 12：FunkSVD

实现位置：

```text
goodbooks_mf/models.py::FunkSVD
```

预测公式：

```text
prediction(u, i) = global_mean + user_bias[u] + item_bias[i] + p_u · q_i
```

偏置的含义：

- `global_mean`：全体训练评分均值；
- `user_bias`：某个用户习惯打高分还是低分；
- `item_bias`：某本书整体更受欢迎还是更不受欢迎；
- latent dot product：偏置以外的个性化偏好。

Goodreads 评分整体偏高，且不同用户量表使用方式不同，因此 bias 对这个数据尤其重要。

## 17. Step 13：regularization 与 early stopping

正式参数：

```text
n_factors = 40
learning_rate = 0.01
reg_lambda = 0.02
max_epochs = 100
patience = 10
seed = 20260830
```

每个 epoch 后计算 validation RMSE。如果连续 `patience` 个 epoch 没有足够改善，则停止训练，并恢复 validation RMSE 最低时的参数快照。

重要点：最终测试使用 best validation epoch，而不是最后一个训练 epoch。

## 18. Step 14：正式实验结果

SVD++ 由独立的 `goodbooks_mf/svdpp.py` 实现，不修改本节冻结的 Basic
MF/FunkSVD。它只使用 train 构造隐式历史，并通过共享实验 runner 使用
validation RMSE 选择八个固定候选。私有 bundle 验证通过后，运行：

```bash
uv run python research/goodbooks-mf/run_all_experiments.py \
  --models svdpp \
  --output research/goodbooks-mf/results/ricky_validation_selection.json
uv run python research/goodbooks-mf/run_svdpp_evaluation.py
```

第二条命令冻结选中配置后才读取 test，并拒绝覆盖已经存在的正式产物。
SVD++ 的最终数字必须从生成的 JSON 引用，不能手工填写。

运行：

```bash
uv run python research/goodbooks-mf/run_experiment.py
```

结果：

| Model | RMSE | MAE | Best epoch | Epochs trained | Time |
|---|---:|---:|---:|---:|---:|
| Basic MF | 1.149409 | 0.882003 | 32 | 43 | 42.79s |
| FunkSVD | 0.861341 | 0.668944 | 9 | 20 | 24.52s |

解释：

- FunkSVD RMSE 比 Basic MF 低约 0.288。
- FunkSVD MAE 比 Basic MF 低约 0.213。
- bias 吸收了大量系统性评分偏差，因此模型更快达到最佳 validation epoch。
- 评分预测会在计算 RMSE/MAE 时 clip 到 `[1, 5]`；Top-K 排序应使用未 clip 的 raw score，避免大量并列。

## 19. Step 15：测试策略

测试覆盖：

- 非法评分、缺失 ID 和时间清洗；
- 重复记录选择规则；
- k-core 是否迭代到稳定；
- ID mapping 是否确定；
- staging 是否会残留旧源记录；
- temporal split 的时间顺序和显式评分覆盖；
- cold item removal；
- Basic MF 固定 seed 的确定性；
- FunkSVD bias 与 best epoch restoration；
- recommendation 排除已见 item；
- Parquet/CSR bundle 与 checksum；
- 修改 artifact 后验证必须失败；
- 从小型 gzip 到最终 bundle 的端到端集成测试；
- 实验结果是否包含数据规模、超参数和训练时间。

验证结果：

```text
40 passed, 0 failed
GitHub Actions: success
```

## 20. 面试讲解框架

可以用下面的两分钟版本：

> 我负责 Goodreads Poetry 矩阵分解项目的数据基础设施、Basic MF 和 FunkSVD。原始数据有约 273 万条交互，我没有一次性加载，而是流式解析 gzip，并用 SQLite primary key 做全局确定性去重。之后执行同时约束交互数和显式评分数的迭代 k-core，用固定 seed 抽样并再次 k-core，得到约 5,100 用户、5,500 图书和 26 万交互。我按每个用户的时间线切分，并确保 validation/test item 在训练 catalog 中出现。为了让三个人严格使用同一数据，我输出 Parquet、CSR 和包含所有 SHA-256 的 manifest，训练前强制验证。模型方面，我实现了只遍历 observed triplets 的 Basic MF 和带 global/user/item bias 的 FunkSVD，并使用 L2 regularization 与 validation early stopping。FunkSVD 的测试 RMSE 从 Basic MF 的 1.149 降到 0.861，说明评分尺度偏差是 Goodreads 的重要信号。

## 21. 常见面试问题

### 为什么不用普通 SVD 直接分解矩阵？

评分矩阵极度稀疏，缺失值不是 0。普通 SVD 需要完整矩阵或填充缺失值，会引入错误假设。FunkSVD 只优化观测评分。

### 为什么 rating 0 不能训练？

数据中 0 表示没有显式评分，并不表示用户非常不喜欢这本书。

### 为什么要做 k-core？

减少只有少量历史的用户和图书，提升参数可学习性、降低方差，并控制计算规模。

### 为什么 k-core 要迭代？

删除一侧节点会改变另一侧的支持度。只有迭代到 fixed point 才能保证所有剩余节点同时满足约束。

### 为什么抽样后还要再次 k-core？

抽掉用户后，一些图书会失去足够交互和评分支持。

### 为什么采用时间切分而不是随机切分？

推荐系统实际使用过去预测未来。随机切分可能把未来兴趣泄漏到训练集。

### 为什么去掉 cold validation/test items？

纯 MF 无法为训练中未出现的 item 学习 latent vector。保留它们会把冷启动和模型质量混在一起。

### 为什么要加入 bias？

用户打分习惯和图书整体受欢迎程度造成系统性偏差。latent factors 不应浪费容量重复学习这些一阶效应。

### 为什么 Top-K 排序不能用 clip 后的分数？

大量预测可能被压到 5，造成排序并列。raw score 保留相对顺序；clip 只用于评分范围与误差指标。

### RMSE 和 MAE 有什么区别？

RMSE 对大误差惩罚更重；MAE 更容易解释为平均差多少颗星。一起报告能更全面描述误差。

### 如何确保三个人数据一致？

锁定 config 和 seed 还不够；每个产物必须通过 manifest SHA-256 验证。相同 hash 才能证明字节一致。

### MF 如何处理新用户或新图书？

本项目的纯协同过滤不能直接解决 cold start。可使用 popularity fallback、内容特征、作者/genre embedding 或混合推荐。

### 当前实验的最大限制是什么？

只使用 Poetry 子集，数据来自 2017 年；Top-K 指标尚需统一 candidate set；genre 子图也不代表完整 Goodreads 行为。

## 22. 尚未完成的团队事项

1. Ricky 冻结共享 candidate set 和 Top-K evaluation functions。
2. 使用同一 test users 与 candidates 评估五个模型。
3. 在结果中加入 Precision@5/10/20、Recall@5/10/20、NDCG@5/10/20。
4. Yutao 的统一 runner 使用 validation 选参，并且测试集只在参数冻结后运行一次。
5. 将 5.4 MB processed bundle 放到经过课程许可的受限共享位置。

这些事项不应通过修改 Ziqi 的 preprocessing 来完成；`goodreads-poetry-v1` 应保持 immutable。
