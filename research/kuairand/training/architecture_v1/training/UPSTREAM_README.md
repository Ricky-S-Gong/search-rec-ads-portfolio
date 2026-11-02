# KuaiRand-1K：独立 logQ 对照实验

从 training_features 的冻结版本复制，仅改变训练目标中的负采样频率修正。旧源码、旧 run 和 best.pt 保留。正式训练由用户在新 Colab 手动启动，从头初始化；不要加载旧目标的 checkpoint。

## 修正与比较

Q(i) = 训练正样本中视频 i 的出现次数 / 训练查询总数。使用已经校验为 long_view=1 且 is_random=0 的 train 分片；不使用 validation/test 反馈、全量曝光或视频总目录频率。Q 是整轮随机打乱采样的经验边际概率；批内去重屏蔽和不放回抽样使它不是每个位置的精确条件概率。

每一列视频的训练 logit = cosine(user,item) / temperature - log(Q(item))。随后沿用同目标视频重复位置屏蔽和对角线交叉熵。非正/非有限或尺寸不正确的概率报错。推理和验证只计算原始 cosine，不额外加减 logQ。

与旧 10 轮实验保持相同：seed=42、Attention 4 heads、Deeper、Residual、BatchNorm、dropout=0.1、用户 131 维输入、物品 64 维输入、输出 64 维；Adam lr=0.001、batch=256、temperature=0.1、最多 10 epochs、连续 3 轮未提升停止。每轮完整验证并保留最佳 checkpoint，按 Recall@100 优先、NDCG@100 打破并列。

全量候选、upload_next_day 资格规则和验证分母不变；未覆盖目标按 miss 计。目标覆盖率约 75.738%。测试集保持封存。新旧 loss 不是相同目标，不能比较 loss 绝对值来判断质量。

## 运行与恢复

配置：configs/attention-logq-epochs10-es3.yaml。新 run 使用独立空目录，W&B 标签含 logq。新版 latest.pt 可在同一配置、数据、源码、运行时和目录下恢复；新增 counts_sha256 绑定，Q 从校验数据重建，频率缓冲不重复写入模型 checkpoint。旧目标的 checkpoint 不允许恢复到新目标。

```sh
uv sync --frozen --python 3.11.16
uv run python run.py preflight --data DATA --config configs/attention-logq-epochs10-es3.yaml --output CHECK_DIR
uv run python run.py train --data DATA --config configs/attention-logq-epochs10-es3.yaml --output NEW_RUN_DIR
uv run python run.py train --data DATA --config configs/attention-logq-epochs10-es3.yaml --output SAME_RUN_DIR --resume
uv run python run.py export --data DATA --config configs/attention-logq-epochs10-es3.yaml --output SAME_RUN_DIR
```

检查：`uv run python -m unittest test_training test_logq`。合成测试覆盖独立公式/梯度、重复目标、非法概率、训练频率来源、精确恢复、旧目标拒绝、early stopping、特征参与、AMP 重试。真实数据另做 CPU 特征与 loss 验收，零次优化器更新。GPU 验收和完整质量比较尚待用户训练；不保证 logQ 提升。

静态内容属性沿用固定内容假设；历史点赞可用时间仍用播放结束代理。历史左补 -1，mask 排除补齐，原始视频 ID=0 合法。下载器不下载封存测试目标。源码包不含数据、模型、向量或 API key。
