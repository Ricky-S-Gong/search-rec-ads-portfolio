# Pure 双塔集成 v2：训练前准备

代码固定到 Ziqi PR #36 的 commit 9359168f8d55b85f456c0b01d1cbbbfa3b1c37f6。
原始上游 item_tower.py、dataset.py、data.py、DATA_HANDOFF.md 和校验值保存在
research/kuairand/kuairand/vendor/ziqi/。上游文件未修改。

数据位于共享 Drive：
kuairand-Two-Tower/datasets/kuairand-pure-v1/。
四个文件的当前大小和修改时间与已核验的快照一致；下载时仍逐个验证 SHA-256。

## 已完成的接口对齐

- 使用 Ziqi 的实际 ItemTower，8 个类别特征＋4 个数值特征，输出 64 维。
- 上游 loader 只返回编码特征，没有原始用户/视频 ID 和请求上下文，也没有可恢复的数据游标。
  因此消费者批次适配器保留原始 ID、time_ms、tab、history_item_ids 和 history_mask，
  并继续使用确定性全局排列与 checkpoint 游标。未直接将上游 DataLoader 替换进训练循环。
- 上游 loader 在全候选视频上拟合词表和数值统计。消费者改为只在 train 引用的视频上拟合，
  将参数和全候选的变换结果持久写入 manifest；验证、测试不会参与拟合，未知类别映射为 0。
- 原始视频 ID 0 与 padding 严格区分：历史 padding 原始 ID 为 -1；编码后保留第 0 行为 padding。
- 历史视频由同一个 ItemTower 编码，用户塔对有效历史做 masked mean，与用户 ID、场景和时间拼接。
  空历史保留，池化向量为零，不丢弃样本。
- 主训练采用 train 非随机 long_view 正样本；validation 使用独立正样本，每个 epoch 记录
  validation/inbatch_loss。这不是 Recall@K，召回评估仍交给 Ricky。

## 数据与协议验证

124,780 条训练正样本，30,780 条验证正样本，7,583 个视频。
全部 476,724 条非随机正样本历史重建比较通过：只含同一用户严格更早毫秒的最多 50 次观看，
同毫秒目标不进入历史，重复观看保留。历史可包括 warmup 和请求之前的行为。

UTC 时间切分逐条与上游 data.py 对齐，标签沿用官方 long_view，随机曝光排除。
只有来源文件校验值、上游 commit、固定 UTC 边界和历史完整核验全部通过，消费者 manifest
才标记 time_protocol_verified=true；没有手动绕过旧 manifest 的阻塞开关。

## 如何开始

使用 v2 Colab 笔记本和 SAR-colab-integrated-v2.zip/sha256，旧 v1 是尚未接物品塔的快照。
所有单元手动运行：挂载个人 Drive → 校验源码并重建锁定环境 → 授权下载固定数据 →
prepare-pure → check-data → 读取 W&B Secret → GPU preflight → 手动 train。

为新笔记本开启 WANDB_API_KEY 的 Secret 访问。预检查须显示 ready_for_training=true。
GPU preflight 验证实际 CUDA/AMP 前向、loss 和有限梯度，不更新参数，不创建训练 run。
本地 CPU 检查不等同于当前 Colab GPU 检查；本轮没有运行真实训练实验。

切换 Colab CPU/GPU runtime 会丢失 /content 缓存，需要重新运行初始化和数据准备。
训练输出固定在个人 Drive，断开后按原配置、源码和数据快照使用 --resume-latest。
开始训练后 10–15 分钟查看 W&B loss、warning、吞吐和 checkpoint。

## 函数说明

- prepare_pure.prepare_pure：检查源文件、UTC split、全部严格历史，准备 train/validation/candidates 与持久特征映射。
- item_tower.fit_item_encoding：只从 train 视频拟合类别词表与数值变换，冻结后变换所有候选。
- item_tower.ItemAdapter.encode_items：将原始 ID 映射到冻结的元数据，并调用共享的上游 ItemTower。
- UserTower.encode_users：对带 mask 的历史向量池化，拼接用户与上下文，再输出归一化向量。
- ParquetBatches.batches：补齐历史、返回原始目标 ID，保留全局排列和下一批游标。
- training.train：Adam/AMP、重复目标屏蔽、验证 loss、W&B、完整 checkpoint 和恢复。
- prepare_pure.check_data：遍历训练/验证记录；preflight 额外检查实际设备的双塔前后向。

真实阶段验收仍需手动训练产物与 Ricky 的检索评估；Pure 不能自动替代团队原定 1K 验收。
