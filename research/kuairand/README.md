# KuaiRand：双塔候选召回

给定用户在请求时的上下文，从符合时间资格的视频中召回 Top-100。当前完成的是 **KuaiRand-Pure Week 1 内容属性实验**；模型尚未超过热门基线，1K 预处理和新的基线／评估协议属于下一周任务。

## Navigation / 模块职责

| 目录 | 职责 | 指南 |
| --- | --- | --- |
| `data_pipeline/` | 初始日志标准化、时间切分、请求历史和原始物品属性交接 | [数据流水线](data_pipeline/README.md) |
| `training/` | 两塔、训练、W&B、checkpoint、导出；统一 CLI 与 Colab | [训练](training/README.md) |
| `evaluation/` | 完成时间协议、精确 Top-K、指标、恢复验收与 release | [评估](evaluation/README.md) |
| `reports/week1/` | 最新中文技术报告、可编辑 LaTeX、PDF 和聚合证据 | [报告](reports/week1/README.md) |

`training/recommender/vendor/data_pipeline_snapshot/` 是固定版本的上游物品塔，不直接修改其文件或 provenance hash。统一执行入口为 `training/run.py`，没有第二套训练系统。

## Latest measured experiment / 最新实测

Release：`week1-pure-content-v1-001`；执行源码：`562682341f1b7a9cc49e518fc43b948052ecf871`。两组各消费 124,255 条训练正反馈、486 次更新，使用同一份 30,589 条验证请求和 7,583 个候选。

| 模型 | 用户输入 | 物品输入 | Recall@100 | NDCG@100 |
| --- | --- | --- | ---: | ---: |
| 热门 | 不使用用户特征 | 训练期正常推荐产生的长观看次数 | 27.078% | 0.069607 |
| Basic | 用户 ID、场景、周期时间：44 维 | 7 个类别 + 4 个数值：60 维 | 2.504% | 0.007095 |
| Enhanced | 相同基础输入 + 平均历史：108 维 | 相同物品特征；历史复用本模型的物品塔 | 2.262% | 0.006145 |

物品塔 `60→64→64`；用户塔 `44/108→128→64→64`，均输出归一化 64 维向量。两组分别训练，架构一致不表示权重相同。默认 seed 42、Adam lr 0.001、batch 256、1 epoch、temperature 0.1、CUDA AMP。

监督使用官方 `long_view=1`，仅保留 `is_random=0` 的常规推荐曝光。随机曝光指平台为实验随机选视频展示，不是打乱 batch。损失为批内交叉熵，屏蔽同目标视频的非对角重复位置；其余视频是采样比较对象，不是确定的负反馈。

结果来自常规曝光验证，受旧推荐策略的选择偏差影响，不能解释为无偏偏好评价或线上效果上限。下一阶段需设计随机曝光下的计数／长观看率基线及独立时间窗口评估。发布属性采用固定内容假设，只有时长与请求日志独立核对；可见状态、整月统计和无历史版本的用户统计被排除。

## Run / 运行入口

在仓库根目录：

```bash
uv sync --locked --directory research/kuairand/training --python 3.11.15
uv run --locked --directory research/kuairand/training python run.py --help
uv run --locked --directory research/kuairand/training pytest tests -q
```

GPU 入口：[week1_content_retraining.ipynb](training/notebooks/week1_content_retraining.ipynb)。它从挂载的 Drive 直接读已准备的数据，不重复下载原始数据。旧的 [week1_baseline.ipynb](training/notebooks/week1_baseline.ipynb) 用于 ID-only 历史对照，不是当前内容属性训练入口。

[已执行 Colab notebook](https://colab.research.google.com/drive/1rQwo_Fe_HufE3q7VvA1RTNmPiDBHQ_Yv) · [完整 Drive release](https://drive.google.com/drive/folders/18qFm162pVnjsusK1s0mux6dTWIbtdKZw) · [最新报告](reports/week1/week1.pdf)

## Storage and reproducibility / 存储与版本

Git 保存聚合指标与版本证据；Drive 保存数据、模型、向量与逐请求预测；W&B 保存 run 日志。Drive 路径以项目／协议／run 命名，旧产物保留。训练、恢复和评估检查源码、配置、数据 manifest、Python/PyTorch 与设备绑定；不能修改旧 manifest 来绕过版本失败。

`stage_report.json` 的 `complete=true` 仅表示三组评估和 CUDA 恢复证据齐全，不代表效果胜出。恢复检查是同一 runtime 内的受控中断恢复，尚未验证重建 Colab runtime 后的恢复。正式测试集保持封存。

旧目录迁移和本机 checkout 说明见 [workspace guide](../../docs/workspace.md)；[原项目计划](docs/project-plan.md) 保留作规划依据，以当前报告和 README 判断已完成状态。
