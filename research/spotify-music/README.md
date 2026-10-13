# Spotify：内容相似度推荐

CPU 实验和可交互曲目探索，使用 Kaggle `vatsalmavani/spotify-dataset` 的音频属性。无需 Spotify API key；没有用户行为数据，因此不是个性化偏好训练。

## Current artifact / 当前产物

`spotify-experiment-v1`：170,653 条去重曲目，seed 42；前端示例 5,000 条。脚本生成的产物位于 [public/artifacts/spotify](../../public/artifacts/spotify/)，包含 `metrics.json`、`demo-tracks.json`、`feature-space.svg`、`feature-trends.svg`。

| 方法 | 实验定义 |
| --- | --- |
| Popularity | 无种子曲目或偏好信号时的回退 |
| Euclidean KNN | StandardScaler 后的欧氏距离 |
| Exact cosine | MinMaxScaler 后的精确余弦相似度 |
| Cluster cosine | 先缩小到聚类候选，再计算余弦相似度 |

音频属性包括 acousticness、danceability、energy、instrumentalness、liveness、loudness、speechiness、tempo、valence。前端提供可调权重探索；离线实测按脚本中的冻结方法进行，不把每次页面交互当作新实验。

评估使用 40 个查询，报告检索耗时、聚类对精确结果的 Recall@10、覆盖、多样性和特征邻域指标。这里的 Recall 是对精确检索结果的还原率，不是用户喜欢歌曲的命中率；不能与 MovieLens / KuaiRand 的 Recall 直接比较。

## Run / 复现

从仓库根目录使用根 Python 环境：

```bash
uv sync --locked
uv run --locked python research/spotify-music/run_experiment.py --data research/spotify-music/data/data.csv
uv run --locked pytest research/spotify-music/tests -q
```

`download_data.py` 调用外部 `kaggle` CLI；需另行安装并配置其下载权限，`uv sync` 不会安装 Kaggle CLI。也可以自行取得授权数据后放入忽略的 `data/` 目录。已有 CSV 时无需下载：

```bash
uv run --locked python research/spotify-music/download_data.py
```

避免覆盖网站的已发布结果，可指定新输出目录：

```bash
uv run --locked python research/spotify-music/run_experiment.py --data research/spotify-music/data/data.csv --output /tmp/spotify-experiment --demo-size 5000
```

## Boundaries / 解释边界

原始数据不提交或重新分发。前端使用生成的精简曲目与特征产物，没有用户行为或凭证。内容相似、类型一致性、检索近似程度和延迟都是代理指标，不能证明用户满意度或线上推荐收益。数据来源与展示属性以当前指标 JSON 和仓库第三方说明为准。
