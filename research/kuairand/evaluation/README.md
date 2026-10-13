# KuaiRand evaluation / 精确召回与阶段验收

当前评估版本是 Pure `completion-time-v1` 加 `train-publication-content-v1` 特征包。统一 CLI 在 `../training/run.py`，使用训练目录的锁定环境。当前真实结果见 [项目概览](../README.md) 和 [报告证据](../reports/week1/evidence/content-v1/stage_report.json)。

## Protocol / 冻结协议

- UTC：4 月 15 日前为历史预热，4 月 15–28 日训练，4 月 29–5 月 3 日验证，5 月 4 日起测试封存。原始 `date` 不作为切分依据。
- 一条验证请求是一条去重后的常规曝光长观看记录；输入包含该次用户、场景、时间及 Enhanced 的已可用历史，唯一正目标是该条记录实际长观看的视频。共 30,589 条，按请求平均，不按用户等权平均。
- 正历史最多 50 条；只纳入 `request_time + play_time < current_request_time` 的已完成反馈。包含此前已可用的验证反馈，但模型参数不更新。播放时长是反馈完成时间代理。
- 完整候选库 7,583 个视频；上传日期只有天粒度，从次日 UTC 开始合格，缺失资格信息排除。允许重复推荐已看视频；所有模型使用相同请求和候选规则。
- 双塔按单位向量点积排序；热门按训练窗口常规曝光长观看计数排序，同分按原始 item ID 升序。热门分数不依赖用户；本次候选全已合格，所有请求得到同一热门榜。
- 分块精确 Top-100，不保存整个请求×候选矩阵；不足 100 时返回实际数量。未覆盖目标仍留在分母并记为未命中。

单目标 Recall@100 是命中请求数／总请求数，等于 HitRate；100 是榜单长度，不是分母。NDCG 只把实际目标记为相关性 1，理想情况是排第 1，IDCG=1；命中排名 r 时得 `1/log2(r+1)`，否则 0。其他候选在指标中的相关性 0 不等于已确认的负反馈。

常规曝光由原推荐算法选择，随机曝光由平台随机选视频展示；当前验证依赖常规曝光，不能作为无偏线上效果。随机曝光计数／长观看率基线、低曝光处理和独立随机曝光评估仍需设计，当前热门基线不是效果上限。

## Commands / 执行

先 `cd research/kuairand/training`；以下路径占位符替换为同一个实际 run 的冻结产物：

```bash
uv run --locked python run.py evaluate --config configs/pure-content-basic.yaml --manifest /path/to/content/manifest.json --output /path/to/new-popularity --label popularity
uv run --locked python run.py evaluate --config configs/pure-content-enhanced.yaml --manifest /path/to/content/manifest.json --checkpoint /path/to/enhanced/checkpoint.pt --vectors /path/to/enhanced/vectors --output /path/to/new-enhanced-evaluation --label enhanced
uv run --locked python run.py verify-resume --config configs/pure-content-enhanced.yaml --manifest /path/to/content/manifest.json --checkpoint /path/to/enhanced/checkpoint.pt --output /path/to/new-recovery-check
uv run --locked python run.py stage-report --config configs/pure-content-basic.yaml --results /path/to/popularity/evaluation.json /path/to/basic/evaluation.json /path/to/enhanced/evaluation.json --resume-check /path/to/recovery/resume_verification.json --output /path/to/new-release
```

Basic 评估使用对应 basic 配置、checkpoint、vectors 和 `--label basic`。配置、数据、训练代码、Python/PyTorch、设备及向量绑定不一致时明确失败，不修改旧元信息绕过检查。

## Evidence and limits / 产物与限制

`evaluation.json`：指标、目标覆盖、条件指标、未知用户、空历史、不足 K 和计时；`predictions.parquet`：私有逐请求榜单；`release_manifest.json` / `stage_report.json` / `stage_report.md`：版本绑定与阶段验收。

计时区分 GPU 用户编码和 CPU 精确排序，包含特征组装与必要的设备传输，排除初始加载和一个预热 batch；这是批量离线成本，不是线上延迟。计时预热与数据历史预热是不同概念。

`verify-resume` 在第 25 步中断 Enhanced，比较最终参数、优化器、AMP、RNG 和游标是否与连续训练一致。当前通过的是同一 runtime 内的受控恢复。`complete=true` 要求三组相同协议评估、两组完整 CUDA 训练及恢复证据，不保证模型超过热门，也不证明新 runtime 恢复。

当前数据包：Drive `datasets/week1-pure-content-v1/`。其请求、切分和候选文件继承已审计的 `week1-pure-completion-v1`；改变的是训练拟合的物品属性编码。历史 ID-only 与早期 8 类别模型保留作追溯，不进入当前报告正文。Git 仅存聚合证据，逐请求数据留在 Drive。
