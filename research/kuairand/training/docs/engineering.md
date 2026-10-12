# Yutao Week 1：实现与交接

用户塔、双塔组合/损失、训练与恢复、向量导出、CLI 和 Colab Notebook 已实现。正确性测试使用 CPU 合成数据，不是推荐效果实验。Colab A100 锁定环境及用户塔 CUDA/AMP 单步检查已通过；真实 1K 数据和 Ziqi 物品塔的集成尚待验证。

## 与 Ziqi 对接的 manifest

此实现只消费已发布的正样本 Parquet，不下载或清洗原始日志、不构建历史、不决定候选资格。`batch_source.py` 是训练消费者适配层；若 Ziqi 提供不同格式，修改适配层，共同维护唯一清洗流程。

```json
{
  "schema_version": 1,
  "dataset_version": "kuairand-1k-v1",
  "feature_version": "basic-v1",
  "positive_only": true,
  "label": "long_view",
  "duration_unit": "seconds",
  "time_protocol_verified": true,
  "train_time_ranges_ms": [[0, 1]],
  "user_ids": [1, 2],
  "train_shards": [{"path": "train.parquet", "rows": 2, "sha256": "REPLACE"}],
  "candidates": {"path": "candidates.parquet", "rows": 2, "sha256": "REPLACE"}
}
```

示例时间是占位符；真实时间区间必须由发布者按 4/15–4/28 的训练日期和核验后的时区提供，采用半开毫秒区间。`time_protocol_verified` 是发布者对时间戳、时区和反馈协议的声明，不是自动推导。user_ids 仅由训练窗口确定。路径相对 manifest，SHA-256 与行数必须真实。

训练表必须含下列无 null 字段：

| 字段 | 输入约定 |
|---|---|
| user_id / item_id | 当前版本内非负整数 ID |
| time_ms | int64 Unix 毫秒，处于训练窗口 |
| tab | 整数；越界值走未知场景 |
| category | 整数；缺失使用双方约定的编码 |
| duration | 有限浮点，秒；缺失使用约定哨兵 |
| long_view / is_random | 分别全为 1 / 0；随机曝光另留诊断集 |

候选表含 item_id/category/duration，按非负 item_id 全局升序且唯一。候选资格、数据清洗和真实重复观看的保留由数据发布方负责；不要因 user/item 相同直接去重。

第一周 baseline 无历史，训练器传 `[B,0]` mask。`UserTower` 接受历史接口但不读取日志。未来历史只能由 Ziqi 的组装器提供。

`model.item_factory` 指向 `module:function`。工厂输入 `(config, manifest)`，返回带 `encode_items(features)` 的 `torch.nn.Module`。features 包括 item_id/category/duration，每个为 `[N]`；输出 `[N,64]`。默认指向 Ziqi 尚未提供的模块。tests 中的物品塔只用于测试，不能作为真实训练结果发布。

## 公共函数说明

| 函数 | 输入 → 输出 / 行为 |
|---|---|
| load_config(path) | YAML → 验证后的字典和 canonical SHA-256 |
| check_env() | 无输入 → Python、包、CUDA 与 GPU 信息 |
| init_wandb(config,hash) | 配置 → 独立 setup run；读取个人凭据 |
| UserTower.encode_users(features,history,mask) | user_id/time_ms/tab `[B]` → `[B,64]` 单位向量；未知用户专用 embedding |
| TwoTower.encode_items(features) | 物品字段 → 维度和数值检查后的归一化向量 |
| batch_loss(users,items,ids,temperature) | 对齐向量 → 正对角线 CE；重复 item ID 的非对角项不作为负样本 |
| read_manifest(path) | manifest → 版本、路径根、SHA；验证声明、文件校验和行数 |
| ParquetBatches.batches(epoch,start) | 轮次/下一批游标 → 确定性全局打乱的 batch |
| train(config,manifest,output,resume) | 配置/数据/持久目录 → checkpoint、metrics.jsonl、run_manifest.json |
| save_checkpoint(directory,state) | 完整状态 → 校验成功的 .pt 与 .complete.json |
| load_checkpoint(path,binding) | 可信私有 checkpoint → 校验后状态；损坏/版本错误明确失败 |
| latest_checkpoint(directory) | 目录 → 最新完整且校验通过的文件 |
| export_items(config,manifest,checkpoint,output,start,end) | `[start,end)` ID 范围 → 对齐 ID 的 NPZ 分片及 manifest |

用户上下文采用固定 UTC 偏移，配置默认 0 是待核验起点。最后一批仅一条样本时 BatchNorm 用 running stats，批内负样本损失为零，不丢弃该记录。归一化向量是匹配表示，不是点击概率。

## 手动运行

```bash
cd research/kuairand/training
uv sync --locked
bash cmd/check_env.sh
uv run --locked python run.py check-config
uv run --locked python run.py init-wandb
bash cmd/train.sh --manifest /absolute/manifest.json --output /persistent/runs/yutao/run-001
bash cmd/train.sh --manifest /absolute/manifest.json --output /persistent/runs/yutao/run-001 --resume-latest
uv run --locked python run.py export --manifest /absolute/manifest.json --checkpoint /absolute/model.pt --output /persistent/releases/vectors-001 --start-id 0 --end-id 100
```

init-wandb 和 train 会创建/接续远程 run，必须由成员手动运行。train 默认 CUDA、在线 W&B，参数由 YAML 读取。导出示例 100 只是接口示意，真实 ID 区间来自候选清单。新 run/导出目录需为空；恢复使用原目录和原配置。修改配置作为新 run。

W&B 使用 SAR-Lab/kuairand-two-tower。API Key 放 .env 或 Colab Secrets，不上传环境变量或凭据文件。个人 Colab 账号与 W&B 学校账号可以不同。

## 恢复与性能边界

checkpoint 保存双塔、Adam、AMP scaler、Python/NumPy/PyTorch CPU/CUDA RNG、轮次、下一批游标、global step、消费数和 W&B run ID。版本绑定包括配置、manifest、代码、Python/PyTorch 和设备类型。先完整保存，再校验并发布完成标记。只加载团队自己产生的可信 checkpoint，PyTorch pickle 不适合不可信文件。

W&B resume=must 恢复同一 run；train_step 作为曲线横轴，记录恢复事件。非正常退出后，最后一次 checkpoint 之后的更新会丢失并重放，这是正常恢复语义。run_manifest 消费数表示已恢复模型轨迹的逻辑样本数；原始 metrics.jsonl 保留重放记录，不能用日志行数当成唯一事件覆盖证据。

缓存按 row-group 数量限制，实际内存由发布的 row-group 大小决定。全局 permutation 约 8 字节/训练事件。这是偏重正确性的首版读取器，正式训练前将输入缓存至 /content，测量等待占比，与 Ziqi 对接预取与高效读取。checkpoint 暂不自动清理，成员须监控空间并保留冻结模型和可恢复点。

## Colab 入口与验收

使用 research/kuairand/training/notebooks/colab.ipynb。手动选择 GPU、挂载 Drive，固定代码版本，安装 uv 0.12.18，再创建 Python 3.11.16 的锁定 venv。Notebook 的挂载代码使用托管 kernel，训练脚本使用 venv Python。先看 check-env 的 cuda_available，再启动实验。所有 Notebook 单元均未执行。

代码可从固定 Git commit 拉取，或上传交付 ZIP 至 /content/SAR-colab-source.zip。ZIP 不包含凭据、数据、模型或本地环境。2026-10-08 已确认个人 Gmail 获得共享文件夹编辑权限，文件夹仍为空；输出放自己的 MyDrive 私有 run 目录。

CPU 检查已覆盖未知用户/场景、空历史、singleton、手算重复目标损失、确定性 batch、恢复前后模型/Adam/RNG/loss 一致、checkpoint 损坏与版本错误、导出 ID 与向量一致、随机曝光/未来事件拒绝。

待验收：Colab CUDA/AMP 恢复容差、真实 1K 数据/物品塔、Drive 持久写入、吞吐，以及 Ricky 的检索/评估/Stage 1 报告。不能把合成检查当成业务效果。

2026-10-08 实际 Colab 验证：Python 3.11.16、PyTorch 2.14.1+cu130、A100-SXM4-40GB；7 项 CPU 测试和用户塔前向/AMP 反向/Adam 更新/checkpoint roundtrip 通过。完整训练器的 CUDA 恢复尚未验收。验证运行时已删除；随后在 CPU 运行时使用 Colab Secret 完成 W&B 创建与回读；检查 run f0pad7ji，setup/connection_ok=1，状态 finished。详情见 colab-environment-validation.json。Notebook 使用 run_command 流式输出子进程日志，避免静默输出。

W&B 线上验证通过： https://wandb.ai/SAR-Lab/kuairand-two-tower/runs/f0pad7ji 。现有学校 W&B 账号的 API Key 可由个人 Colab 通过 Secrets 使用。只记录 setup/connection_ok，未运行训练实验或伪造模型指标。
