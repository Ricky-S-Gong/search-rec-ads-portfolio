# KuaiRand-Pure 数据处理与交接

这份说明记录 Pure smoke-test 数据已完成的处理、给训练代码的输入
规则，以及 Ziqi 接下来需要交付的两个接口。

## 已完成

原始数据位于共享 Drive：

```text
kuairand-Two-Tower/raw/KuaiRand-Pure/data/
```

处理后的数据位于：

```text
kuairand-Two-Tower/datasets/kuairand-pure-v1/
```

| 文件 | 实际内容 | 用途 |
| --- | --- | --- |
| `events.parquet` | 2,622,668 条曝光/观看记录 | 选择训练、验证和测试样本 |
| `user_histories.parquet` | 476,724 条正样本的严格历史 | User Tower 的序列输入 |
| `item_features.parquet` | 7,583 个视频的静态属性 | Item Tower 的特征输入 |
| `manifest.json` | 数据版本、字段和 split 数量 | 快速核对数据版本 |

### `events.parquet`

字段为：

```text
request_id, user_id, item_id, request_time_ms, tab, long_view, is_random, split
```

- `item_id` 来自官方 `video_id`。
- 官方 `time_ms` 已是 Unix 毫秒时间戳。
- `split` 已按时间固定：`warmup`（4 月 14 日及之前）、`train`
  （4 月 15–28 日）、`validation`（4 月 29 日–5 月 3 日）、`test`
  （5 月 4 日起）。
- `is_random=True` 的记录只用于未来诊断，不用于训练。

### `user_histories.parquet`

字段为：

```text
request_id, user_id, history_item_ids
```

每行通过 `request_id` 连接回 `events.parquet`。`history_item_ids` 是该
用户在当前正样本之前最多 50 个非随机 `long_view` 视频 ID。当前请求与
同一毫秒发生的请求都不会进入历史，避免标签泄漏。历史可以为空；训练
代码应以 padding/mask 支持空历史，或在第一版明确过滤它们。

### `item_features.parquet`

字段来自 `video_features_basic_pure.csv`，其中 `video_id` 已改名为
`item_id`，包含作者、视频类型、上传方式、可见状态、时长、画面尺寸、
音乐和标签等静态属性。所有 `events.item_id` 都已检查可在此文件找到。

没有使用 `video_features_statistic_pure.csv`。它的播放、点赞、评论等
聚合指标可能使用了当前或未来期间的反馈，直接输入模型会造成时间泄漏。
`user_features_pure.csv` 也还未使用；第一版 User Tower 先只使用历史序列。

## 给训练代码的最小规则

1. 取 `events.parquet` 中 `split == "train"`、`long_view == True` 且
   `is_random == False` 的行作为训练正样本。
2. 用 `request_id` 左连接 `user_histories.parquet`；用 `item_id` 左连接
   `item_features.parquet`。
3. 验证使用同样规则下的 `split == "validation"` 行。
4. 负样本采样、loss、optimizer、Recall@K 由训练模块负责；不得把 test
   行用于词表统计、归一化统计或模型选择。

## 接下来由 Ziqi 交付

### 1. 基础 Item Tower

**目标：** 输入一批 `item_id` 和 `item_features`，输出固定维度的视频向量。

- 第一版只用 `item_features.parquet` 的静态字段。
- 类别字段使用 embedding；数值字段（时长、宽、高）标准化后拼接。
- 输出维度与 User Tower 约定一致，例如 64 维。
- 代码建议位置：`research/kuairand-twotower/kuairand_twotower/item_tower.py`。

### 2. 训练批次读取器

**目标：** 从三张 Parquet 表读取并连接出训练/验证 batch。

- 按上述最小规则过滤正样本。
- `history_item_ids` 补齐到固定长度 50，并同时返回 mask。
- 返回目标 `item_id`、目标视频特征、历史视频 ID、历史 mask 与 split。
- 代码建议位置：`research/kuairand-twotower/kuairand_twotower/dataset.py`。

### 3. 与 Yutao 的接口确认

在开始编码前确认三个数字：`embedding_dim`、`max_history_length`（当前
数据为 50）和 batch 的字段名。Yutao 可以现在并行完成 User Tower、训练
循环和评估骨架；这两个文件完成后即可接入端到端训练。

## 已交付的模型接口

`kuairand_twotower.dataset.build_dataloader` 读取三张 Parquet 表并返回
`(loader, schema)`。它只接受 `train` 或 `validation` split，并筛选
`long_view=True`、`is_random=False` 的正样本。

每个 batch 都有以下张量：

```text
target_categorical  [B, 8]
target_numeric      [B, 4]
history_categorical [B, 50, 8]
history_numeric     [B, 50, 4]
history_mask        [B, 50]  # bool，True 表示真实历史
```

8 个类别特征依次为 `item_id`、`author_id`、`video_type`、`upload_type`、
`visible_status`、`music_id`、`music_type`、`tag`。4 个数值特征依次为
时长、宽、高和上传日期。类别 ID 的 `0` 专门保留给 padding/unknown。

`kuairand_twotower.item_tower.ItemTower` 接收 `schema` 的类别基数，输出
L2-normalized 的 64 维向量；目标视频和历史视频使用同一个调用方式：

```python
from kuairand_twotower.dataset import build_dataloader
from kuairand_twotower.item_tower import ItemTower

loader, schema = build_dataloader(
    "/content/drive/MyDrive/kuairand-Two-Tower/datasets/kuairand-pure-v1/events.parquet",
    "/content/drive/MyDrive/kuairand-Two-Tower/datasets/kuairand-pure-v1/user_histories.parquet",
    "/content/drive/MyDrive/kuairand-Two-Tower/datasets/kuairand-pure-v1/item_features.parquet",
    split="train",
    batch_size=32,
    shuffle=True,
)
item_tower = ItemTower(schema.categorical_cardinalities, schema.numeric_feature_count)
batch = next(iter(loader))
target_vectors = item_tower(batch["target_categorical"], batch["target_numeric"])
history_vectors = item_tower(batch["history_categorical"], batch["history_numeric"])
print(target_vectors.shape, history_vectors.shape, batch["history_mask"].shape)
```

预期输出形状是 `torch.Size([32, 64])`、`torch.Size([32, 50, 64])` 和
`torch.Size([32, 50])`。Yutao 将 `history_vectors` 与 `history_mask` 传给
User Tower，并自行处理负采样、loss、训练和 W&B。

## 复现命令

```bash
PYTHONPATH=research/kuairand-twotower python research/kuairand-twotower/prepare_kuairand_data.py \
  --raw-dir /content/drive/MyDrive/kuairand-Two-Tower/raw/KuaiRand-Pure/data \
  --output-dir /content/drive/MyDrive/kuairand-Two-Tower/datasets/kuairand-pure-v1

PYTHONPATH=research/kuairand-twotower python research/kuairand-twotower/build_user_histories.py \
  --events /content/drive/MyDrive/kuairand-Two-Tower/datasets/kuairand-pure-v1/events.parquet \
  --output /content/drive/MyDrive/kuairand-Two-Tower/datasets/kuairand-pure-v1/user_histories.parquet

PYTHONPATH=research/kuairand-twotower python research/kuairand-twotower/prepare_item_features.py \
  --basic /content/drive/MyDrive/kuairand-Two-Tower/raw/KuaiRand-Pure/data/video_features_basic_pure.csv \
  --events /content/drive/MyDrive/kuairand-Two-Tower/datasets/kuairand-pure-v1/events.parquet \
  --output /content/drive/MyDrive/kuairand-Two-Tower/datasets/kuairand-pure-v1/item_features.parquet
```
