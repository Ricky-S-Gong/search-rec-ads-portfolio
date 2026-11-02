# KuaiRand-1K：Ziqi 架构实验 Implementation Plan

日期：2026-10-31。范围：先确定并重训 B0，建立共享 R0，再完成 Ziqi 的方向一架构实验。本文是实施计划，不代表已完成新训练或代码改造。

## 1. 核验结论与实施起点

本计划结合最新团队分工 PDF、GitHub 当前代码与报告，以及 Drive 实际 1K 训练包制定。你的即时范围是六个唯一配置：B0、R0、A1、A2、A3、A5；A4 复用 B0。后续 H5–H8 optimizer/scheduler 仍属于你的分工，但等 B1 冻结后再做。

### 1.1 必须用实际 1K 源码

GitHub `research/kuairand/README.md` 和 1K 报告均明确：仓库 `training/` 是早期 Pure 实现；1K 实验使用独立 Drive release。不要直接用 Pure `run.py` 复现 1K，也不要从 Pure Basic 的无历史输入定义 B0。

| 依据 | 本次实际核验 | 对实施的影响 |
| --- | --- | --- |
| GitHub | 核验时 main SHA：`1f328e3224205ba7a655b8edef38a30a38796656` | 锁定这次调研所依据的版本 |
| 原 1K logQ 源码包 | `SAR-KuaiRand-1K-training-logq-v1.zip`，SHA256=`d06ab5a84b17bc232223edd54c3cb926076cf2a26fda49d2eb9150a48690537b` | 读取其中 `configs/baseline.yaml`，确认旧 shallow 定义 |
| 可复用 ablation 包 | `SAR-KuaiRand-1K-ablation-colab-v3.zip`，SHA256=`c6ba01c80a8c377f022d87659a6e33a18a81febb325c124ed6d570a781a19459` | 基于其中 `kt1k/`、runner 和汇总脚本做小范围修改 |
| 两包 baseline 配置 | 两份 `configs/baseline.yaml` 逐字比较一致 | B0 有实际源码依据，不是重新猜一个 MLP |
| 现有 notebook | 读取 Yutao 已执行的 manual-training-features-v2 notebook | 找回 baseline 的历史输出，但 notebook 当前源 cell 已改成 deep，不能直接点 Run All 当 B0 |

### 1.2 Yutao 在 deep 之前的 shallow 是什么

原 baseline 使用内容与行为特征，用户输入 131D、物品输入 64D；历史物品通过共享 Item Tower 和既有行为反馈处理后做 masked mean pooling，再与用户上下文、行为统计和偏好特征拼接。保持这些输入，只改变后续实验指定的架构因素。

- User tower：`131 → 128 → 64 → 64`，对应 `user_hidden=[128,64]`。
- Item tower：`64 → 64 → 64`，对应 `item_hidden=[64]`。
- 输出均为 L2-normalized 64D；MLP hidden layer 后为 ReLU，末端为输出投影。
- `history_encoder=mean`、`batch_norm=false`、`residual_blocks=0`、`dropout=0`。
- `behavior_content_features=true`，最多 50 条严格历史。

这里的“shallow”指两塔 MLP。原模型仍包含 embedding、历史反馈 projection/gate 和偏好构造；保留这些公共模块才能保持输入一致。不能把 B0 描述成完全没有历史处理、只含 Linear 的裸 MLP。配置中的 `attention_heads=4` 只是备用字段，mean 分支不会实例化 Attention。

### 1.3 找回的历史成绩如何使用

GitHub 报告此前只保留了 baseline 汇总。本次从已执行 notebook 的输出中额外找回 run `6eff0cad3e82`，记录为训练完成 3 epochs、best epoch=3、Recall@100=`0.0007918269`（`0.07918269%`）、NDCG@100=`0.0001582770`。这增强了历史线索，但本次未重新加载其 checkpoint，也未重算排名。notebook 当前代码与历史输出还存在版本混杂，不能把该输出当成新实验验收证据。

正式 B0 按本轮共同规则从头重训：最多 20 epochs、patience=3。旧 3-epoch 成绩仅作为 historical reference，不填进新的匹配消融结果表。

## 2. 正式 B0：哪些继承，哪些为本轮统一修改

| 配置项 | 正式值 | 来源与说明 |
| --- | --- | --- |
| 架构、特征 | 上述原 shallow + mean；behavior/content 开启 | 继承原 baseline |
| Optimizer | Adam，betas=(0.9,0.999) | 原 engine 实际直接构建 Adam；不能只看 YAML 是否写 optimizer |
| Learning rate / weight decay | 0.001 / 0 | 继承原 baseline |
| Batch size / temperature | 256 / 0.1 | 所有架构组固定 |
| BN / Residual / dropout | false / 0 / 0 | 所有架构组固定 |
| Sampling | 自然训练 batch + in-batch negatives；No Correction | 显式设 sampling_correction=none，alpha=0，smoothing=0 |
| 训练预算 | epochs≤20，early_stopping_patience=3 | 本轮团队规则；原 baseline 是 3 epochs |
| LR schedule | constant | 架构组不加入 warm-up、cosine 或其它调参 |
| Seeds | 先 42；最终 42、2026、3407 | 六个唯一配置全部补齐；不只复跑赢家 |
| 设备 | CUDA + AMP + deterministic | 正式实验尽量统一 GPU 型号与运行环境 |
| 选择 checkpoint | validation Recall@100 优先，再 NDCG@100，再更早 epoch | 严格比较前两项可自然保留完全同分的早期 epoch |
| 排名评价 | full catalog FP32；K=100；upload_next_day | raw item ID 升序打破同分；允许重复推荐；未覆盖目标按 miss 计 |
| 评价分块 | query_batch_size=128；item_batch_size=16384 | 继承现有配置；这是计算分块，不是训练 batch |

注意：No Correction 仍使用 in-batch negatives；它不等于没有负样本。B0 不使用 logQ 或 item-balanced resampling。

## 3. 方向一实验矩阵与归因

全部固定第 2 节配置。Deep 定义为 user_hidden=[256,128,64]、item_hidden=[128,64]。本轮深度对照同时改变层数和宽度，应称“两塔 MLP 配置/容量变化”，不能声称独立识别纯层数效应。

| ID | 两塔 | 历史 encoder | 匹配对照 | 能回答的问题 |
| --- | --- | --- | --- | --- |
| B0 | Shallow | Mean | 全局简单参考 | 原浅层方案在统一预算下的表现 |
| R0 | Deep | Attention | A5；A2 | Deep 下换 encoder；Attention 下换 MLP 配置 |
| A1（P1） | Deep | LSTM | R0；A5 | LSTM vs Attention；LSTM vs Mean |
| A2（P2） | Shallow | Attention | B0；R0 | Shallow 下 Attention vs Mean；同 encoder 的 MLP 配置差异 |
| A3（P2） | Shallow | LSTM | A2；B0；A1 | Shallow encoder 差异；LSTM 下的 MLP 配置差异 |
| A4 | Shallow | Mean | reused B0 | 同配置不重复训练 |
| A5（P3，必做） | Deep | Mean | B0；R0；A1 | Mean 下的 MLP 配置差异；Deep encoder 对照 |

原 Attention 分支是 4-head context-query attention，带其既有 position embedding；LSTM 是单层、hidden size=64，按有效长度移除左 padding，以最后 hidden state 聚合。直接复用，不换成 Transformer，不额外堆叠网络。

Attention 与 LSTM 的比较识别的是现有 encoder 整体实现差异，包括其位置/顺序处理；不能进一步断言收益只来自 attention 权重。报告各配置 trainable 参数量、耗时和显存，说明容量差异。本轮不额外增加参数量严格匹配实验。

## 4. 实施前必须修正的现有包问题

以下已由源码静态检查发现，均需在正式控制组前处理。

| 文件/模块 | 当前状态 | 计划改动 | 验收标准 |
| --- | --- | --- | --- |
| configs/ablation_control.yaml | Deep Attention + BN + Residual + dropout=.1 + logQ | 新建 b0.yaml、r0.yaml；替换本轮架构矩阵，不继承旧 control | 六行 resolved config 除指定架构项和 run metadata 外一致 |
| configs/ablation_plan.yaml | 旧矩阵含 residual 组、旧 H 组；部分架构行仍继承 logQ/dropout | 使用 B0/R0/A1/A2/A3/A5 的明确 ID，A4 仅 alias | 自动生成 6×3=18 个唯一正式任务，无旧 control 混入 |
| run_ablation.py | 自动额外插入旧 control；默认依次 subprocess；capture_output 会延后显示日志 | 改为显式实验清单，实时 tee 日志；按 config×seed 去重、恢复 | 没有隐含第七配置；运行中可见 step；已有完成结果不覆盖 |
| runner preflight | 主要检查 passed，未充分绑定本轮 resolved config | 核对源码、配置、数据与设备绑定 | 配置变化后的旧 preflight 不能误放行 |
| kt1k/engine.py | 保存 latest/best；尚无每 epoch 独立权重及 Artifact 上传 | 集成 Annabel helper：epoch-NNN.pt、Artifact、索引 | 任一 epoch 可定位、下载；latest 用于恢复，best 用于评价 |
| engine / evaluation | 有训练 step loss 和 ranking 指标，无共同口径 validation loss | 集成固定 batch/query 方案的 validation loss；累计 epoch train loss | 每 epoch 两种 loss 与 Recall/NDCG 完整，loss 非最后一个 batch 冒充平均 |
| summarize_ablation.py / report | 旧 ID、旧 control/finalist 逻辑 | 对齐新 ID/control_id、三 seeds 和 A4 reuse | 18 行原始结果，6 行均值/标准差；失败任务单列 |

Annabel helper 若尚未交付，我可以先实现你必须使用的最小接口，并给双方同一版本，避免各写一套互不兼容的记录逻辑。正式 B0/R0 之后若公共代码改变了训练或评价行为，应重新冻结版本并重跑受影响控制组。

## 5. 分阶段执行与每一步的出口

### 阶段 0：代码与数据冻结——我完成准备，你完成 GPU 授权

我整理实际 1K 源码、环境锁、数据/feature manifest、六份配置和实验清单；不把 Pure 模块拷来替换实际执行逻辑。保留旧 release，新增本轮工作目录。

只准备 train/validation 及其共同特征，训练入口不读取 sealed test。历史报告给出的预期规模：train positives=1,335,793，validation queries=555,677，catalog=4,371,868；最终以本轮已核验 manifest 为准。行为特征须读取 enrichment overlay，不能只下载 base dataset 后静默落回 108D 用户输入。

你在 Colab/所用 GPU 平台选择 runtime、授权 Drive、通过 Secrets 提供 W&B 凭证并确认有项目写权限；凭证不写入 notebook、Git 或报告。当前环境没有已连接的远程 GPU 会话，正式训练预计由你启动。

出口：六行配置检查通过；源码与数据 hash 有记录；resolved model 输入确认为 user=131、item=64、输出=64。

### 阶段 1：短程验收——我写检查，你在目标 GPU 运行

先运行已有测试，再补本轮新增行为的最小检查：

1. B0/R0/A1/A2/A3/A5 实际模块检查：B0 无 Attention/LSTM/BN/Residual；所有行 dropout=0、No Correction。
2. 历史 mask：0、1、50 条历史；无效 padding 不改变输出；空历史无 NaN；LSTM 保持有效历史顺序。
3. 检查 enriched 特征接入、两塔梯度、normalized 64D 输出和 finite loss；重复 item 非对角不能被当负例。
4. 小样本完成训练→validation→epoch checkpoint→Artifact→best/latest 全链路；小样本结果只标 smoke。
5. 人为中断再恢复，比较相同设备上的未中断路径：batch cursor、optimizer、scaler、RNG、更新次数与指标一致；不保证跨不同 GPU 位级一致。
6. 先对六个配置各运行真实数据零更新 GPU preflight，再做 100–200 updates 的短程探测，记录训练吞吐和显存。短程探测使用独立输出目录，正式模型仍从头训练。

出口：恢复和上传已在目标 GPU 验证；无静默降级、错分支、NaN 或资源不可承受的问题。若只是 evaluator 的分块 OOM，统一调整分块并重做 preflight；不得私自降低某一组训练 batch=256。

### 阶段 2：B0 与 R0 seed42——先建立共同控制

顺序跑 B0 seed42，再跑 R0 seed42，均最多 20 epochs、patience=3、每 epoch 完整 validation。原始 3-epoch baseline 和旧 BN/Residual/logQ 成绩不参与新控制组计算。

每个控制组都需有 resolved config、hash、实际模型结构、逐 epoch 指标、best/latest/epoch 权重、恢复证据和 W&B 链接。B0 的初步 pipeline 跑通后可开发其它行；R0 通过后即可把共享包交 Annabel，不必等你的架构实验全部结束。

出口：B0/R0 的新结果可复核；Annabel 用相同 R0、源码与协议开展 S1–S3。R0 三 seeds 由你统一归档并共享，不让双方各跑一份控制组。

### 阶段 3：完成剩余架构 seed42——先有可汇报结果

按团队优先级排队：A1 → A2 → A3 → A5。A5 是 P3 但必做，不能因 R0 优于 B0 就跳过它。实验均独立随机初始化，不从前一组权重继续训练。

出口：6 个唯一配置各 1 seed 的结果表、loss/metric 曲线、matched delta、耗时/显存与当前状态。单 seed 标 exploratory，不据此声称稳定提升。

### 阶段 4：补齐三 seeds——完成你的架构交付

对每个唯一配置补 seeds=2026、3407，共新增 12 次；加上 seed42 合计 18 次完整架构训练。A4 引用 B0，不额外计数。比较使用相同 seed 的 paired delta，并报告三 seeds 的均值、样本标准差和各 seed 原始值；只有 3 seeds，不夸大统计显著性。

结合 Annabel 的采样结果确定后续 B1。若最佳 deep encoder 与最佳采样组合从未共同训练过，要新增组合验证，不能直接把两个赢家拼成已验证 B1。若 shallow 胜过 deep，报告应如实保留该结论；按团队既定规则选 deep B1 时，明确它仅用于后续 deep 内调参。

出口：你的六个配置均为 completed，或给出明确 blocked 证据；为 B1 提供架构结果，不开展新的超参搜索。

## 6. 并行、恢复与资源预算

单个 GPU 默认一个正式训练进程；现有 runner 是顺序队列，多线程不能自动变成多 GPU 训练。可在两个独立 GPU runtime 分配互不重复的配置；Annabel 的采样任务也可在她的 runtime 并行运行。每个任务独占 run 目录与 W&B run ID，只共享只读数据和冻结源码。

建议目录：`runs/1k-architecture-v1/{config_id}/seed-{seed}/`。任务状态为 pending/running/completed/failed/blocked/reused；启动前核对 manifest，恢复时使用 latest，不从 best 恢复成新的独立 seed。需用占用锁或原子任务认领，防止两个 runtime 同时写同一任务。

正式每轮约 ceil(1,335,793/256)=5,218 次更新，上限 104,360 次/训练。18 次最多 360 个训练 epochs，early stopping 可减少实际数量；全目录验证通常也是主要成本，不能只根据 train step 时间估算总耗时。

先用一个完整 epoch 测得各 encoder 的 train_seconds、validation_seconds 和 peak memory，再估算剩余队列。若第 e 轮后提前停止，预算约为各轮 train+validation 时间之和，再加数据加载与权重上传；本计划不虚报固定小时数。每 epoch 独立 checkpoint 与排名文件会占空间，先测单份大小并预留 18×20 份上限；不要未经同步就删除既有产物。

## 7. 我能做什么，你需要做什么

| 工作 | 我负责 | 你负责 |
| --- | --- | --- |
| 确定 B0 | 查源码、核验历史证据、解释定义；本轮已完成静态核验 | 理解其含义，与 Annabel 使用同一共同控制 |
| 实验代码 | 修改配置、runner、记录/恢复接口、汇总脚本，复用 kt1k | 在目标 GPU 验收实际运行环境 |
| 正确性检查 | 写并运行可用本地测试、分析 mask/shape/梯度问题 | 运行 GPU/真实数据 preflight 和恢复检查 |
| 运行包 | 提供可执行 notebook、明确命令、队列与恢复说明 | 选择 GPU、授权 Drive、设置 Secrets、启动队列 |
| 正式训练 | 有可访问的已授权 runtime 时可协助操作；分析回传日志和异常 | 当前条件下负责启动、检查 session、算力/存储与断线恢复 |
| 监控 | 实现日志和状态输出；在你提供的日志基础上定位错误 | 启动时、首个 epoch 后、日常数次、异常时检查运行状态 |
| 实验分析 | 合并结果、matched delta、三 seeds 汇总、曲线与结论草稿 | 复核是否符合团队协议，沟通结果并解释实验 |
| Annabel 交接 | 准备同一源码/配置/数据协议和控制组索引 | 把共同包交给 Annabel；沟通其 helper 接口与状态 |

不要求你手动盯每个 batch。我会把需要看的信息集中到运行状态、loss/metrics、GPU/显存、checkpoint 和同步状态。当前对话不会在结束后持续监控远程 session；你发现异常或积累新日志后，我可以继续诊断与修复。

### 你运行时的具体监控清单

| 检查时机 | 看什么 | 何时行动 |
| --- | --- | --- |
| 启动后最初几分钟 | config_id/seed、No Correction、131/64D 输入、GPU、run 路径 | 分支或输入错立即停，修复后新目录重跑 |
| 训练中 | step 是否增长、loss 是否 finite、AMP retries 是否持续、显存 | NaN/Inf 或重复 overflow 先存日志并停查；不临时给单行改 lr |
| 验证中 | phase=validation、query/item 进度、GPU/日志仍活跃 | 没有 train step 增长可能只是验证；结合首轮耗时与 evaluator heartbeat 判断卡住 |
| 每 epoch 后 | Recall/NDCG、coverage、best_epoch、patience、epoch Artifact | coverage/分母变化必须查协议；曲线无需每轮都上升 |
| 断线/恢复后 | latest.pt、cursor、run ID、manifest binding、重复 epoch | 配置或源码绑定不一致先查原因，不能删 hash 绕过 |
| 一次任务结束 | stop_reason、best/epoch 文件、Artifact 同步、结果表行 | 训练结束但上传未完成仍需补同步，不标交付完成 |

## 8. 运行接口与交付清单

实施后沿用实际 1K CLI，在源码 `training/` 下运行。下列命令使用计划中新建配置文件，当前尚未生成，不应直接复制到旧包执行：

```bash
python run.py preflight --data "$DATA_DIR" --config configs/b0.yaml --output "$CHECK_DIR"
python run.py train --data "$DATA_DIR" --config configs/b0.yaml --output "$RUN_DIR"
# 相同配置、源码、数据、输出目录下恢复：
python run.py train --data "$DATA_DIR" --config configs/b0.yaml --output "$RUN_DIR" --resume
```

我后续应交付：

1. 一份冻结的实际 1K 运行包与环境锁；一份 notebook，包含授权、检查、短程、正式队列、恢复、汇总步骤。
2. 六个唯一架构配置、任务 manifest、A4→B0 alias、control_id 与配置差异校验。
3. 共用记录接口：逐 epoch train/validation loss、Recall/NDCG、actual LR、GPU/时间、参数量、checkpoint/Artifact 索引。
4. 针对新增功能的测试与本地测试记录；GPU 和真实数据验收明确由运行证据补齐。
5. `architecture_raw_results.csv`（每 config×seed 一行）、`architecture_summary.csv`、匹配差值、曲线和解释报告。
6. 给 Annabel 的 R0 交接说明：源码 hash、数据/feature/split 协议、配置、三 seeds 控制记录、成功命令与恢复路径。

结果表至少包含：config_id、control_id、seed、MLP/encoder、sampling、source/config/data hash、best_epoch、completed_epochs、stop_reason、epoch train_loss、fixed validation_loss、Recall@100、NDCG@100、trainable_parameters、train/eval/total_seconds、peak_gpu_memory、run_url、checkpoint_uri、status。指标原始值按 0–1 保存，展示百分数需明确 ×100；delta 同时给原始尺度与百分点，避免把 0.00079 当 0.079。

最终 test 沿用团队约定，由 Annabel 的统一 test 入口在配置、checkpoint 和比较方案全部锁定后执行；本阶段不拿 test 选架构。最终合并的是结果与 checkpoint 索引，不是模型权重。

## 9. 来源与核验边界

- [GitHub 项目说明](https://github.com/Ricky-S-Gong/search-rec-ads-portfolio/blob/1f328e3224205ba7a655b8edef38a30a38796656/research/kuairand/README.md)
- [GitHub 1K 结果与来源说明](https://github.com/Ricky-S-Gong/search-rec-ads-portfolio/blob/1f328e3224205ba7a655b8edef38a30a38796656/research/kuairand/reports/1k/2026-10-17-logq-vs-popularity-vs-two-tower.md)
- [原 1K 源码包](https://drive.google.com/file/d/1K7Esa-qNEZbo6Mcr58PcEiaDg0LBK4hW/view)
- [现有 ablation v3 源码包](https://drive.google.com/file/d/1XWp6t-Ctu5l9hg3TU9G6RuYVRvE08yTy/view)
- [已执行 manual-training-features-v2 notebook](https://colab.research.google.com/drive/1bm1DrrtJFrB-0Ba57UaFvsN-H1-sLZiH)
- 最新团队计划：`KuaiRand_1K_Ablation_Implementation_Plan (1)(1).pdf`，本次读取完整六页；本计划保留其 ID、分工、优先级和三 seeds 要求。

W&B 提供的 CoreWeave 页面本次无法读取；上述历史指标来自 GitHub 报告与 notebook 输出，未据此宣称远程 run 的当前状态。未下载完整训练数据、未运行新 GPU 训练、未改共享 Drive/GitHub，也未提交实验结果。本轮完成的是源码核验与实施设计。
