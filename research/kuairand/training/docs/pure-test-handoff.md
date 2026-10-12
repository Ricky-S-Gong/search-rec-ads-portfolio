# Pure 留出测试集交接

本包不包含训练或验证请求，不启动训练，也不报告推荐效果。

- test_events.parquet：71,438 条非随机测试曝光，包含原始 long_view=0/1。
- test_requests.parquet：22,083 条 long_view=1 评估请求；字段 request_id、user_id、item_id（正确目标）、time_ms（Unix 毫秒 UTC）、tab、history_item_ids，以及原始标签/split。
- item_features.parquet：原始静态视频属性。
- candidates.parquet：7,583 个按 ID 排序的候选视频及兼容特征。
- frozen_features.json：从训练数据拟合的用户词表与视频编码，用于检查编码一致性，不是可加载的训练 manifest。
- evaluation_manifest.json：数量、范围、源文件哈希、产物哈希和协议。
- export_test.py：本包的可重复生成脚本，历史包在原项目的 research/kuairand 下运行；当前源码入口为 research/kuairand/training/export_test.py。

## 分割和历史协议

按行为时间分割，测试窗为 2022-05-04 00:00 UTC 至数据末尾。test request_id 与 train/validation 无交集；用户和视频允许重叠。不是用户/视频冷启动分割。
逐条重建并核对全部 22,083 条测试历史：同一用户，严格早于当前毫秒，最多 50 条非随机长观看，顺序与发布历史一致，同毫秒全部排除。
采用滚动历史：较早的验证/测试长观看可以作为后续请求的已观测历史；不更新模型/特征统计。若需要固定在训练末尾的历史，请另设协议，不能混用指标。
884 条请求来自训练词表外用户，使用模型未知用户索引 0，应单独报告该子组；该未知用户 embedding 未在当前 train 中受训练。
历史缺位用 raw item ID=-1，mask=False；有效视频 ID=0 必须保留，不能当 padding。

## 评估建议与接口

读取 test_requests，每行是一条请求，item_id 是单个相关目标，历史按左侧 padding 到 50。
用同一 checkpoint 的 User Tower（含共享 Item Tower 历史编码）生成归一化用户向量。
与已经导出的归一化视频向量做点积，按分数降序取 Top100；按真实 item_id 对齐，不能假定向量行号。
计算 Recall@100（目标是否命中）和 NDCG@100（单目标命中时 1/log2(rank+1)，否则 0），按请求平均；同一请求集比较训练期热门 baseline。
本包默认全 7,583 项候选，不过滤历史已看视频，以免把重复观看的真实目标删掉；每个请求的视频历史可用性尚未审计，因此结果是全目录离线诊断。若加入资格过滤，需要同步更新目标资格和分母并报告排除数量。
不要用 test 调参；参数选择继续用 validation。

## 对应训练产物（尚在 Yutao 个人 Drive）

W&B run: https://wandb.ai/SAR-Lab/kuairand-two-tower/runs/b2ed5728b9a2
checkpoint: SAR/runs/yutao/w1-pure-integrated-v2-001/checkpoints/step-000000488-fd99a30d.pt
checkpoint SHA256: e3ee79a232f322d8099238b85f74a8a6507fb9c8bfac350c282b0837934d62e0
视频向量目录: SAR/releases/stage1/yutao-vectors-001/ （items-00000.npz、items-00001.npz 和 export manifest）

加载模型请使用实际 Colab 那次训练对应的配置、源码和训练 manifest；实际 manifest 哈希为 0e10290ad2eac565ad5386ddf51eaaeeb181426b436ac1ac6c56bb9db27ab0ba。
本包使用同源文件在本地重新准备，因此本地 preparation manifest 哈希可能因路径/元数据不同而异；不要用它替换实际 checkpoint 绑定的 manifest。
只加载可信团队 checkpoint。训练的 test 曝光/目标未参与优化，测试视频的静态候选属性可以用于推理。
