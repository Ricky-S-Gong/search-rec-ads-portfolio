# MovieLens：协同过滤与热门基线

使用官方 MovieLens 1M 显式评分，在 CPU 比较 Bayesian popularity、bias-aware User-CF 与 Item-CF。当前发布版本是 `movielens-cf-v4`，保留 v3 mean-centered 模型作诊断对照。

[当前指标](../../public/artifacts/movielens/metrics.json) · [数据字典与分布](../../public/artifacts/movielens/profile.json) · [候选搜索与比较](../../public/artifacts/movielens/comparisons.json)

## Run / 复现

从仓库根目录使用根 Python 环境：

```bash
uv sync --locked
uv run --locked python research/movielens-cf/download_data.py
uv run --locked python research/movielens-cf/run_experiment.py
uv run --locked pytest research/movielens-cf/tests -q
```

已存在原始数据时不需要重新下载。快速检查使用独立输出，不覆盖网站产物：

```bash
uv run --locked python research/movielens-cf/run_experiment.py --smoke --output /tmp/movielens-smoke
```

## Protocol / 数据与评估

seed 42，按用户时间进行 80/10/10 切分，在验证集选择参数，随后用 train+validation 重拟合并评估 test。每个用户对拟合候选库中的全部未看电影排序，不采样负例；相关目标是测试评分 4 或 5 星的电影。

HitRate@10 是至少命中一部相关电影的用户比例；Recall@10 是每位用户命中的相关电影数／该用户全部相关测试电影数，再按用户平均。NDCG 衡量命中位置。三种方法均记录用户 bootstrap 95% 置信区间，另有覆盖、长尾、新颖性、活跃度切片、回退比例、并列诊断和成本。这与 KuaiRand 的单目标按请求评估不同。

## Models and selection / 模型与选型

Bias baseline 为 `global_mean + user_bias + item_bias`，User-CF / Item-CF 计算评分残差的用户／物品余弦相似度，再把邻居残差加回偏置预测。排序使用未裁剪分数；并列依次看 Bayesian popularity、绝对相似度证据、movie ID。裁剪到 1–5 只用于评分指标和展示。

完整搜索为 `k={20,40,80} × min_support={5,10,20} × shrinkage={10,25,50}`，先固定 `min_neighbors=2`，再在领先三个候选上比较 `{2,5}`。验证选择按 NDCG、Recall、HitRate、完全并列榜单比例、候选覆盖顺序进行。热门平滑先验的 `prior_weight` 从 `{10,25,50,100}` 在验证集选择。所有候选与淘汰依据进入比较产物。

## Evidence limits and storage / 限制与存储

v4 的改进方向受到已观察的 v3 测试并列问题启发，因此是 post-hoc 模型迭代，不是完全未接触测试结果的确认性实验。具体参数仍由 train/validation 决定。`rating>=5` 只是只读敏感性分析，不替换主相关性定义。

原始 MovieLens 文件放在忽略的 `data/`；GroupLens 不允许无授权重新分发。网站仅读取 `public/artifacts/movielens/` 的派生 JSON/SVG，实验默认输出会更新该目录，试验新方案应使用独立 `--output`。完整 CPU 结果已发布，不代表线上服务已经部署。
