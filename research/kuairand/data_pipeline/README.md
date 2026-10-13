# KuaiRand data pipeline / 数据准备

此模块负责官方日志到初始 Parquet 交接文件；它使用仓库根目录 Python 环境。当前正式实验是已发布到 Drive 的 Pure 内容属性版本，Colab 直接读取现有文件，无需重复下载或重新预处理。

[当前项目](../README.md) · [训练入口](../training/README.md) · [评估与审计](../evaluation/README.md)

## Two stages / 初始准备与正式审计

| 阶段 | 入口 | 产物与边界 |
| --- | --- | --- |
| 初始数据交接 | 本目录四个脚本 | 原日志标准化、基础时间切分、请求时间之前的历史、原始物品属性；这是数据来源，不等于最终验收协议 |
| 正式 Pure 协议 | `training/run.py prepare-week1` | 官方归档与来源 hash 审计、完全相同行去重、按反馈完成时间重建历史、冻结训练／验证／候选 |
| 内容属性版本 | `training/run.py prepare-content` | 保持已审计请求／候选文件不变，训练范围拟合 7 类别与 4 数值编码；保存固定内容假设与时长核验 |

初始 `build_user_histories.py` 的“严格早于请求”只按请求时间检查；正式实验还要求反馈已完成。不要用初始历史文件替换当前已审计数据。`DATA_HANDOFF.md` 是早期 Pure 交接记录，历史数量与接口说明不代表最新版训练包。

## Existing Drive layout / 当前共享存储

```text
kuairand-Two-Tower/
├── raw/                         # 已有官方数据
├── datasets/
│   ├── kuairand-pure-v1/         # 初始来源文件
│   ├── week1-pure-completion-v1/ # 已审计 ID-only 请求与候选
│   └── week1-pure-content-v1/    # 当前训练属性包
├── runs/week1-pure/<run-label>/
└── releases/week1-pure/<run-label>/
```

当前包包含 `manifest.json`、`audit.json`、`train.parquet`、`validation.parquet`、`candidates.parquet`。数据和历史版本不可覆盖，校验值以对应 manifest 为准。用户、物品词表和填补／标准化参数只在训练范围拟合；测试保持封存。

## Initial preparation / 新数据的初始准备

以下从仓库根目录运行，路径替换为实际已解压的目录与新的输出目录：

```bash
uv sync --locked
uv run --locked python research/kuairand/data_pipeline/prepare_kuairand_data.py --raw-dir /path/to/KuaiRand-1K/data --output-dir /path/to/new-source
uv run --locked python research/kuairand/data_pipeline/build_user_histories.py --events /path/to/new-source/events.parquet --output /path/to/new-source/user_histories.parquet
uv run --locked python research/kuairand/data_pipeline/prepare_item_features.py --basic /path/to/KuaiRand-1K/data/video_features_basic_1k.csv --events /path/to/new-source/events.parquet --output /path/to/new-source/item_features.parquet
```

Pure 使用相同脚本，但传入 Pure 原始目录和 `video_features_basic_pure.csv`。这些脚本自身目录包含 `kuairand_data`，通常不需要手动设 `PYTHONPATH`。

仅在尚无官方 1K 归档时使用下载器；已存在文件会校验 MD5，不重复下载：

```bash
uv run --locked python research/kuairand/data_pipeline/download_data.py --destination /path/to/raw/KuaiRand-1K.tar.gz
```

下载器专用于 1K，不会自动解压或发布正式训练 release。现有 Pure 的 `prepare-week1` 包含固定来源 hash 与候选约束，不能直接拿它给 1K 生成正式包；1K 的版本协议和 Drive 发布是下一周任务。

## Formal Pure publication / 正式 Pure 版本

从训练目录运行，输入必须是已核验的团队 Pure 来源与原始归档：

```bash
cd research/kuairand/training
uv run --locked python run.py prepare-week1 --config configs/pure-basic.yaml --source /path/to/verified-pure-source --archive /path/to/KuaiRand-Pure.tar.gz --output /path/to/new-completion-data
uv run --locked python run.py prepare-content --config configs/pure-content-basic.yaml --manifest /path/to/completion-data/manifest.json --source /path/to/verified-pure-source --output /path/to/new-content-data
```

原始 `is_rand` 被标准化为 `is_random`，`video_id` 为 `item_id`。随机展示记录保留在初始数据中，但当前正式训练／验证只使用常规曝光长观看。发布属性缺少历史版本，采用显式固定内容假设；时长与请求记录核对，可见状态和整月统计被排除。

原始／处理后交互和模型文件不提交到 Git；未来基线与随机曝光协议需单独设计、发布版本，不能把旧 release 改写为新实验。
