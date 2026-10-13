# KuaiRand training / 双塔训练

当前入口是 [week1_content_retraining.ipynb](notebooks/week1_content_retraining.ipynb)。它复用既有模型、训练器、checkpoint、向量导出及评估 CLI，在 Colab GPU 执行最新 Pure 内容属性实验。运行结果见 [项目概览](../README.md) 和 [中文报告](../reports/week1/README.md)。

## Inputs and architecture / 输入与架构

| 塔 | 输入与编码 | MLP |
| --- | --- | --- |
| Basic 用户塔 | 用户 ID embedding 32 + tab embedding 8 + UTC 日／周周期 4 | `44→128→64→64` |
| Enhanced 用户塔 | 上述 44 维 + 最多 50 条已可用正反馈的 64 维 masked mean | `108→128→64→64` |
| 两组物品塔 | item_id、author_id、video_type、upload_type、music_id、music_type、首个 tag，各 8 维；video_duration、server_width、server_height、相对上传天数，共 4 维 | `60→64→64` |

类别词表与数值填补／标准化只在训练物品拟合，验证与全候选库复用冻结参数。未知类别索引 0，历史 padding ID -1，空历史池化为零。发布属性采用固定内容假设；动态可见状态、整月统计与无时间版本的用户统计排除。

用户 MLP 在前两次 Linear 后使用 ReLU 与 Dropout，末尾 Linear 后转 FP32 并 L2 归一化；物品 MLP 为 Linear、ReLU、Linear、L2。当前 dropout=0，batch normalization 关闭，最终向量均为 64 维。Enhanced 的历史和目标在同一个模型内共享物品塔；Basic 与 Enhanced 各自独立训练。

训练只使用常规曝光的 `long_view=1`。批内交叉熵以正确配对的对角列为目标，非对角同视频位置被屏蔽；其他采样视频不是确认的负反馈。配置：seed 42、batch 256、1 epoch、Adam lr 0.001、betas (0.9,0.999)、weight decay 0、temperature 0.1、确定性 CUDA AMP。

## Colab / 正常运行

1. 打开内容属性 notebook，连接 CUDA GPU，在 My Drive 添加团队文件夹 `kuairand-Two-Tower` 的快捷方式，并挂载 Drive。
2. 在 Colab Secrets 中配置 `WANDB_API_KEY`，允许本 notebook 访问。密钥不写进文件或输出。
3. 核对 `REF` / `COMMIT` 与 `RUN_LABEL`。新的实验必须使用新的 run 名称；默认示例名称已被已完成实验占用。重现已有 run 时固定其实际 commit `562682341f1b7a9cc49e518fc43b948052ecf871`，不要用更新后的代码接续旧目录。
4. 按顺序执行环境、数据和 GPU/W&B preflight；两组 `ready_for_training=true` 后执行训练，再导出、评估、恢复验收和 stage release。出现版本失败先查绑定，不覆盖旧结果。

Notebook 直接读 `datasets/week1-pure-content-v1/`；依赖和代码安装到 `/content`，数据不会从 Zenodo 重新下载。可选本地缓存只复制已有 Drive 文件。新产物写入 `runs/week1-pure/<run-label>/` 和 `releases/week1-pure/<run-label>/`。

[已执行 notebook](https://colab.research.google.com/drive/1rQwo_Fe_HufE3q7VvA1RTNmPiDBHQ_Yv) · [Basic W&B](https://wandb.ai/SAR-Lab/kuairand-two-tower/runs/087cea6d74ef) · [Enhanced W&B](https://wandb.ai/SAR-Lab/kuairand-two-tower/runs/3e2c0f26ec9b)

## Locked environment / 本地检查

在仓库根目录执行；uv 必须是 0.12.1，训练 Python 是 3.11.15：

```bash
uv sync --locked --directory research/kuairand/training --python 3.11.15
uv run --locked --directory research/kuairand/training python run.py check-env
uv run --locked --directory research/kuairand/training python run.py check-config --config configs/pure-content-enhanced.yaml
uv run --locked --directory research/kuairand/training pytest tests -q
uv run --locked --directory research/kuairand/training ruff check . ../evaluation --exclude vendor --config pyproject.toml
```

`uv --directory` 会改变执行目录，配置参数须使用该环境下的相对路径；下方训练示例从 `training/` 运行：

```bash
cd research/kuairand/training
uv run --locked python run.py preflight --config configs/pure-content-enhanced.yaml --manifest /path/to/content/manifest.json
uv run --locked python run.py train --config configs/pure-content-enhanced.yaml --manifest /path/to/content/manifest.json --output /path/to/new-run
# 中断后，仅在原始版本和同一输出目录恢复：
uv run --locked python run.py train --config configs/pure-content-enhanced.yaml --manifest /path/to/content/manifest.json --output /path/to/original-run --resume-latest
```

真实训练默认要求 CUDA，不在本地 CPU 检查或 CI 自动执行。非 Colab 凭证通过环境变量或仓库根目录的忽略文件 `.env` 提供。

## Code and historical runs / 代码与历史

| 文件／目录 | 职责 |
| --- | --- |
| `run.py` | check、prepare、train、export、evaluate、verify-resume、stage-report 共用 CLI |
| `recommender/user_tower.py`, `item_tower.py`, `model.py` | 用户塔、冻结特征映射、物品塔接入、匹配损失 |
| `recommender/training.py`, `checkpoint.py`, `batch_source.py` | Adam/AMP、批次游标、完整状态保存和 Parquet 输入 |
| `recommender/export.py` | 带 checkpoint 绑定与 shard 校验值的候选向量 |
| `configs/pure-content-{basic,enhanced}.yaml` | 当前内容属性两组配置 |
| `notebooks/week1_baseline.ipynb`, `configs/pure-{basic,enhanced}.yaml` | 历史 ID-only 比较 |
| `recommender/vendor/data_pipeline_snapshot/` | 固定上游物品塔与 provenance；不修改 |

早期 ZIP 训练与测试交接见 [pure-handoff](docs/pure-handoff.md)、[pure-test-handoff](docs/pure-test-handoff.md) 和 [engineering](docs/engineering.md)。这些记录不能与当前协议混成同一结果表；旧 checkpoint 用原源码、配置、manifest 和运行环境恢复。正式 Week 1 结果使用验证集，不读取历史交接中的测试集。
