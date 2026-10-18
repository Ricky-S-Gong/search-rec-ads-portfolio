# KuaiRand-1K：logQ vs 热门基线 vs 基础双塔 / Experiment results

2026-10-17 · Validation only / 仅验证集 · run `c90679f9d390`

logQ 实验已完成：最多 20 epochs，在第 17 轮因连续 3 轮未提升而提前停止；保留第 14 轮最佳 checkpoint。Best checkpoint: epoch 14; stopped after epoch 17 with patience 3.

| 模型 / Model | 训练轮数 / Completed epochs | 最佳轮次 / Selected epoch | Recall@100 | NDCG@100 |
| --- | ---: | ---: | ---: | ---: |
| 热门基线 / Popularity | 0 | 未取得 / unavailable | 0.160165% | 0.000332050 |
| 基础双塔 / Mean-history two-tower | 3 | 未取得 / unavailable | 0.079183% | 0.000158277 |
| 当前 logQ 双塔 / Attention + Residual + BN + logQ | 17 | 14 | 0.513968% | 0.001230623 |

当前 logQ 相对热门基线 Recall/NDCG 分别约 **3.21× / 3.71×**，相对已记录基础双塔约 **6.49× / 7.78×**。架构、训练与模型选择预算不同，不能把全部差距归因于 logQ。These are observed validation comparisons, not an isolated causal ablation or held-out test result.

## 评估口径 / Protocol

- Dataset: KuaiRand-1K (`kuairand-1k-content-v1`); positive train queries **1,335,793**, validation queries **555,677**, full metadata catalog **4,371,868** videos.
- 每条查询一个非随机曝光的 long_view 正目标。Recall@100 是逐事件平均命中率，单目标 NDCG@100 命中时为 `1/log2(rank+1)`，未命中为 0。
- Exact full-catalog FP32 ranking, ascending raw video ID tie break, repeats allowed; query-time candidate rule `upload_next_day`.
- 候选覆盖 **420,861/555,677 = 75.7384%**；未覆盖目标保留在分母中按 miss 计，不是缩小候选池后的数字。
- 热门热度只统计训练集 long_view=1、is_random=0 的事件次数；无神经网络训练，不用验证或测试反馈。Counts and ranking were independently checked with DuckDB and 5,000 brute-force query checks.
- 封存测试集尚未评估；本次只发布聚合结果，不上传数据、逐请求预测、模型权重、向量或密钥。No test performance, vector export or online deployment is claimed.

## 模型设置 / Models

**基础双塔 / Mean-history baseline:** 已有历史特征。历史视频经 Item Tower 编码并结合历史行为反馈，masked mean pooling 后与用户上下文、行为统计及偏好 concat。用户输入 131 维，物品输入 64 维，两塔输出 normalized 64 维；用户 MLP `131→128→64→64`，物品 `64→64→64`。Residual/BatchNorm 未开启，dropout=0，训练 3 轮。

**当前 / Current logQ:** Deeper + 4-head context-query Attention + Residual + BatchNorm + Dropout=0.1。用户 MLP `131→256→128→64→64`，物品 `64→128→64→64`；每个隐藏层后一个 residual block。最多 50 条严格历史，输入同为用户 131 / 物品 64 维，输出 normalized 64 维。Adam lr=0.001、batch=256、temperature=0.1、seed=42、CUDA AMP。每轮完整验证，以 Recall 优先、NDCG 打破并列选择 checkpoint。

**logQ:** `training_logit = cosine / temperature - log(Q(video))`。Q 仅由训练正目标频率拟合，批内重复目标的非对角位置屏蔽；验证和推理使用原始 cosine。Training-only correction; no logQ adjustment to evaluation scoring.

## 证据完整性 / Evidence limits

热门指标为本地实际评估并独立核对；logQ 指标由已执行 Colab 的 17 轮完整聚合日志读取，W&B 同步完成。本报告没有重新加载 logQ checkpoint 或重算其逐请求排名。

**基础双塔数值来自此前保存的 3 轮汇总记录，本次没有找回其原始排名、checkpoint 或 W&B run ID，未重新运行；具体最佳轮次未知。** 该记录为 Recall `0.0007918269066382089`、NDCG `0.00015827701259809888`。以下 JSON 明确标记为 historical recorded aggregate，避免伪装成新验证结果。

The historical mean-history baseline aggregate has weaker provenance than the current run and independently verified popularity metrics. Architecture and epoch-budget differences also limit causal interpretation.

## 查找入口 / Run and source

- [W&B：c90679f9d390](https://wandb.ai/SAR-Lab/kuairand-two-tower/runs/c90679f9d390)
- [已执行 Colab / Executed notebook](https://colab.research.google.com/drive/1B1Tn4fLzZ983m7gssJfoVbFhHaAZUGrr)
- [1K 冻结源码包 / Frozen training source](https://drive.google.com/file/d/1K7Esa-qNEZbo6Mcr58PcEiaDg0LBK4hW/view)
- Run folder: `1k-attention-logq-epochs20-es3-run001`; best artifact: `best.pt` (epoch 14).
- Source ZIP SHA256: `d06ab5a84b17bc232223edd54c3cb926076cf2a26fda49d2eb9150a48690537b`.
- Base data manifest SHA256: `16e28ba2598869986c387fc432c412b804ade49898012d696398190a5e353787`.
- 这些是独立 1K Drive release 的结果，仓库既有 Pure `training/` 不是本次实际执行代码。Pure 与 1K 指标不能直接比较。

## 逐轮指标 / Epoch history

| Epoch | Recall@100 | NDCG@100 | Hits | New best |
| ---: | ---: | ---: | ---: | :---: |
| 1 | 0.097179% | 0.000207599 | 540 | Yes |
| 2 | 0.244746% | 0.000572771 | 1360 | Yes |
| 3 | 0.353263% | 0.000855535 | 1963 | Yes |
| 4 | 0.420208% | 0.001012300 | 2335 | Yes |
| 5 | 0.446302% | 0.001059790 | 2480 | Yes |
| 6 | 0.443423% | 0.001100449 | 2464 | No |
| 7 | 0.481395% | 0.001154930 | 2675 | Yes |
| 8 | 0.481935% | 0.001131379 | 2678 | Yes |
| 9 | 0.475816% | 0.001174255 | 2644 | No |
| 10 | 0.478695% | 0.001115563 | 2660 | No |
| 11 | 0.499571% | 0.001223640 | 2776 | Yes |
| 12 | 0.477796% | 0.001161031 | 2655 | No |
| 13 | 0.476896% | 0.001176593 | 2650 | No |
| 14 | 0.513968% | 0.001230623 | 2856 | Yes |
| 15 | 0.459440% | 0.001122234 | 2553 | No |
| 16 | 0.470777% | 0.001137494 | 2616 | No |
| 17 | 0.456200% | 0.001110239 | 2535 | No |

## 可复核的聚合证据与配置 / Aggregate evidence and configs

<details>
<summary>展开完整 JSON：三组结果、来源、17 轮日志、配置与版本 / Full JSON</summary>

```json
{
  "comparison": {
    "dataset": "KuaiRand-1K",
    "split": "validation",
    "date": "2026-10-17",
    "queries": 555677,
    "catalog_size": 4371868,
    "candidate_rule": "upload_next_day",
    "target_coverage": 0.7573842358060529,
    "aggregation": "event-micro; one long_view positive per query; uncovered targets remain misses; repeats allowed",
    "test_evaluated": false,
    "model_rows": [
      {
        "model_id": "popularity-train-long-view",
        "label": "热门基线 / Popularity",
        "epochs_completed": 0,
        "selected_epoch": null,
        "recall_at_100": 0.001601649879336377,
        "ndcg_at_100": 0.0003320504074438232,
        "evidence_status": "independently verified aggregate",
        "recall_percent": 0.1601649879336377
      },
      {
        "model_id": "two-tower-mean-baseline",
        "label": "基础双塔 / Mean-history two-tower",
        "epochs_completed": 3,
        "selected_epoch": null,
        "recall_at_100": 0.0007918269066382089,
        "ndcg_at_100": 0.00015827701259809888,
        "evidence_status": "historical recorded aggregate; no raw baseline ranks recovered",
        "recall_percent": 0.07918269066382089
      },
      {
        "model_id": "attention-residual-bn-logq",
        "label": "当前 logQ 双塔 / Attention + Residual + BN + logQ",
        "epochs_completed": 17,
        "selected_epoch": 14,
        "recall_at_100": 0.005139676466724374,
        "ndcg_at_100": 0.0012306225651929504,
        "evidence_status": "17-epoch Colab log; selected epoch 14; W&B run c90679f9d390",
        "recall_percent": 0.5139676466724373
      }
    ],
    "logq_fold_vs_reference": {
      "popularity-train-long-view": {
        "recall_at_100": 3.2089887640449435,
        "ndcg_at_100": 3.7061317727825673
      },
      "two-tower-mean-baseline": {
        "recall_at_100": 6.49090909090909,
        "ndcg_at_100": 7.775118730082298
      }
    },
    "interpretation": "Different architectures and training budgets; observed validation comparison, not isolated logQ causal ablation or test performance."
  },
  "evidence": {
    "two-tower-baseline-recorded-metrics.json": {
      "model_id": "two-tower-mean-baseline",
      "dataset": "kuairand-1k-content-v1",
      "split": "validation",
      "training_epochs": 3,
      "best_epoch": null,
      "recall_at_100": 0.0007918269066382089,
      "ndcg_at_100": 0.00015827701259809888,
      "queries": 555677,
      "catalog_size": 4371868,
      "candidate_rule": "upload_next_day",
      "source": "historical-baseline-record.md",
      "verification_status": "Previously recorded aggregate; original baseline checkpoint, ranks and W&B run ID were not recovered for this publication. Not rerun.",
      "test_evaluated": false
    },
    "popularity-validation-metrics.json": {
      "baseline": "train_long_view_popularity",
      "dataset": "kuairand-1k-content-v1",
      "manifest_sha256": "16e28ba2598869986c387fc432c412b804ade49898012d696398190a5e353787",
      "source_sha256": "64517cb22eedb679c017071b0d3e631f67e03fc20d0dfa7aa7cd858af090e08c",
      "train_queries": 1335793,
      "queries": 555677,
      "catalog_size": 4371868,
      "hits": 890,
      "recall_at_100": 0.001601649879336377,
      "ndcg_at_100": 0.0003320504074438232,
      "covered_targets": 420861,
      "target_coverage": 0.7573842358060529,
      "candidate_rule": "upload_next_day",
      "eligibility_states": 5,
      "score": "Count each nonrandom long_view=1 training event once; no warmup/validation/test feedback",
      "ranking": "Exact full catalog; count descending / raw item ID ascending; query-time eligibility; repeats allowed; uncovered targets miss",
      "aggregation": "Mean over validation events, one positive target per query; not macro averaged over users",
      "test_evaluated": false,
      "parameter_updates": 0,
      "evaluation_seconds": 1.7417866250034422,
      "inputs_sha256": {
        "manifest.json": "16e28ba2598869986c387fc432c412b804ade49898012d696398190a5e353787",
        "candidates.parquet": "a496d50cee702bb456f00b10438bb4182acdb97fad7dfac9a35b581aab2ebc36",
        "train-000.parquet": "e0667ecb583e23d29e028c7a648a0c682e3d1b4aa273baf6626bbe9609190f67",
        "train-001.parquet": "6db7a4d25c7d936214943f7b929d0c13440f3b68fc447c2f706e88f4a2435f6c",
        "train-002.parquet": "85e03cfe25891f83818babc84d99b9a6e3acd1778442ec94e2f1207570deb489",
        "train-003.parquet": "6a90b84e58777703333803fad133c44cb27e192f8c16202d3e83ce217e52a59f",
        "train-004.parquet": "9c688d8963624cdf45134ef73c3ab1164b3909b4b842f59db2dab3ee0e46c343",
        "train-005.parquet": "092f844590df640fbf07c266bbbdec8b1a9bf8a973ca68641f39b9fdbf2babec",
        "validation-000.parquet": "875bdcfd49be64590085fb8773215ea695a8df5acb477443395ff9cb4eb23285",
        "validation-001.parquet": "08357d235c9bba38f4df4c45037cf389e559baa0ad13b19d3c208752551739d0",
        "validation-002.parquet": "820e57be0b7bcbadc5908c4e8ae68e8395d494b70b51925981ab54607840afe6"
      }
    },
    "provenance.json": {
      "date": "2026-10-17",
      "dataset": "KuaiRand-1K",
      "manifest_sha256": "16e28ba2598869986c387fc432c412b804ade49898012d696398190a5e353787",
      "combined_base_feature_manifest_sha256": "03fc0cb41c6dfc6b55f590c453b3954ad229af849219c4e91094f66d33742c20",
      "source_zip_name": "SAR-KuaiRand-1K-training-logq-v1.zip",
      "source_zip_sha256": "d06ab5a84b17bc232223edd54c3cb926076cf2a26fda49d2eb9150a48690537b",
      "source_zip_url": "https://drive.google.com/file/d/1K7Esa-qNEZbo6Mcr58PcEiaDg0LBK4hW/view",
      "training_code_sha256": "96b0554b3d7963314b025bb1ef791971c9c9fe5ee479fb8780ad5838eb21afe0",
      "logq_run_id": "c90679f9d390",
      "epoch_history_source": "Rendered Colab output captured after early stopping; not a fresh rerun or checkpoint-file inspection.",
      "test_evaluated": false,
      "model_weights_or_vectors_in_git": false,
      "training_implementation_note": "This report records the independent 1K Drive source release; the repository training/ module is the earlier Pure implementation. Do not use that Pure entry point to reproduce these 1K results."
    },
    "logq-validation-history.json": {
      "source": "Colab rendered training output",
      "colab_url": "https://colab.research.google.com/drive/1B1Tn4fLzZ983m7gssJfoVbFhHaAZUGrr",
      "run_id": "c90679f9d390",
      "run_url": "https://wandb.ai/SAR-Lab/kuairand-two-tower/runs/c90679f9d390",
      "epochs": [
        {
          "epoch": 1,
          "train_step": 5218,
          "validation/improved": true,
          "validation/epochs_without_improvement": 0,
          "validation/stopped_early": false,
          "validation/recall_at_100": 0.0009717875672378019,
          "validation/ndcg_at_100": 0.00020759926902167708,
          "validation/target_coverage": 0.7573842358060529,
          "validation/queries": 555677,
          "validation/hits": 540,
          "validation/evaluation_seconds": 174.2598716509999,
          "validation/catalog_size": 4371868,
          "validation/ranking": "exhaustive FP32; raw ID ascending tie break; repeats allowed; uncovered targets count as misses"
        },
        {
          "epoch": 2,
          "train_step": 10436,
          "validation/improved": true,
          "validation/epochs_without_improvement": 0,
          "validation/stopped_early": false,
          "validation/recall_at_100": 0.002447464984154464,
          "validation/ndcg_at_100": 0.000572771373771822,
          "validation/target_coverage": 0.7573842358060529,
          "validation/queries": 555677,
          "validation/hits": 1360,
          "validation/evaluation_seconds": 174.28633411200008,
          "validation/catalog_size": 4371868,
          "validation/ranking": "exhaustive FP32; raw ID ascending tie break; repeats allowed; uncovered targets count as misses"
        },
        {
          "epoch": 3,
          "train_step": 15654,
          "validation/improved": true,
          "validation/epochs_without_improvement": 0,
          "validation/stopped_early": false,
          "validation/recall_at_100": 0.0035326277675700093,
          "validation/ndcg_at_100": 0.0008555347109675405,
          "validation/target_coverage": 0.7573842358060529,
          "validation/queries": 555677,
          "validation/hits": 1963,
          "validation/evaluation_seconds": 174.47493001500015,
          "validation/catalog_size": 4371868,
          "validation/ranking": "exhaustive FP32; raw ID ascending tie break; repeats allowed; uncovered targets count as misses"
        },
        {
          "epoch": 4,
          "train_step": 20872,
          "validation/improved": true,
          "validation/epochs_without_improvement": 0,
          "validation/stopped_early": false,
          "validation/recall_at_100": 0.004202081425000495,
          "validation/ndcg_at_100": 0.0010123004639277747,
          "validation/target_coverage": 0.7573842358060529,
          "validation/queries": 555677,
          "validation/hits": 2335,
          "validation/evaluation_seconds": 174.12329511500002,
          "validation/catalog_size": 4371868,
          "validation/ranking": "exhaustive FP32; raw ID ascending tie break; repeats allowed; uncovered targets count as misses"
        },
        {
          "epoch": 5,
          "train_step": 26090,
          "validation/improved": true,
          "validation/epochs_without_improvement": 0,
          "validation/stopped_early": false,
          "validation/recall_at_100": 0.0044630243828699045,
          "validation/ndcg_at_100": 0.0010597903828694368,
          "validation/target_coverage": 0.7573842358060529,
          "validation/queries": 555677,
          "validation/hits": 2480,
          "validation/evaluation_seconds": 174.53214362900007,
          "validation/catalog_size": 4371868,
          "validation/ranking": "exhaustive FP32; raw ID ascending tie break; repeats allowed; uncovered targets count as misses"
        },
        {
          "epoch": 6,
          "train_step": 31308,
          "validation/improved": false,
          "validation/epochs_without_improvement": 1,
          "validation/stopped_early": false,
          "validation/recall_at_100": 0.00443423067717397,
          "validation/ndcg_at_100": 0.0011004491100598131,
          "validation/target_coverage": 0.7573842358060529,
          "validation/queries": 555677,
          "validation/hits": 2464,
          "validation/evaluation_seconds": 174.356463479,
          "validation/catalog_size": 4371868,
          "validation/ranking": "exhaustive FP32; raw ID ascending tie break; repeats allowed; uncovered targets count as misses"
        },
        {
          "epoch": 7,
          "train_step": 36526,
          "validation/improved": true,
          "validation/epochs_without_improvement": 0,
          "validation/stopped_early": false,
          "validation/recall_at_100": 0.004813947671039111,
          "validation/ndcg_at_100": 0.0011549303316214716,
          "validation/target_coverage": 0.7573842358060529,
          "validation/queries": 555677,
          "validation/hits": 2675,
          "validation/evaluation_seconds": 174.54282340900045,
          "validation/catalog_size": 4371868,
          "validation/ranking": "exhaustive FP32; raw ID ascending tie break; repeats allowed; uncovered targets count as misses"
        },
        {
          "epoch": 8,
          "train_step": 41744,
          "validation/improved": true,
          "validation/epochs_without_improvement": 0,
          "validation/stopped_early": false,
          "validation/recall_at_100": 0.004819346490857098,
          "validation/ndcg_at_100": 0.0011313792640067087,
          "validation/target_coverage": 0.7573842358060529,
          "validation/queries": 555677,
          "validation/hits": 2678,
          "validation/evaluation_seconds": 174.34497566900063,
          "validation/catalog_size": 4371868,
          "validation/ranking": "exhaustive FP32; raw ID ascending tie break; repeats allowed; uncovered targets count as misses"
        },
        {
          "epoch": 9,
          "train_step": 46962,
          "validation/improved": false,
          "validation/epochs_without_improvement": 1,
          "validation/stopped_early": false,
          "validation/recall_at_100": 0.004758159866253237,
          "validation/ndcg_at_100": 0.0011742547454398262,
          "validation/target_coverage": 0.7573842358060529,
          "validation/queries": 555677,
          "validation/hits": 2644,
          "validation/evaluation_seconds": 173.97697786400022,
          "validation/catalog_size": 4371868,
          "validation/ranking": "exhaustive FP32; raw ID ascending tie break; repeats allowed; uncovered targets count as misses"
        },
        {
          "epoch": 10,
          "train_step": 52180,
          "validation/improved": false,
          "validation/epochs_without_improvement": 2,
          "validation/stopped_early": false,
          "validation/recall_at_100": 0.004786953571949172,
          "validation/ndcg_at_100": 0.0011155629306487372,
          "validation/target_coverage": 0.7573842358060529,
          "validation/queries": 555677,
          "validation/hits": 2660,
          "validation/evaluation_seconds": 174.46891009999945,
          "validation/catalog_size": 4371868,
          "validation/ranking": "exhaustive FP32; raw ID ascending tie break; repeats allowed; uncovered targets count as misses"
        },
        {
          "epoch": 11,
          "train_step": 57398,
          "validation/improved": true,
          "validation/epochs_without_improvement": 0,
          "validation/stopped_early": false,
          "validation/recall_at_100": 0.0049957079382447,
          "validation/ndcg_at_100": 0.0012236399361786477,
          "validation/target_coverage": 0.7573842358060529,
          "validation/queries": 555677,
          "validation/hits": 2776,
          "validation/evaluation_seconds": 174.43347182899925,
          "validation/catalog_size": 4371868,
          "validation/ranking": "exhaustive FP32; raw ID ascending tie break; repeats allowed; uncovered targets count as misses"
        },
        {
          "epoch": 12,
          "train_step": 62616,
          "validation/improved": false,
          "validation/epochs_without_improvement": 1,
          "validation/stopped_early": false,
          "validation/recall_at_100": 0.004777955538919193,
          "validation/ndcg_at_100": 0.0011610311193707226,
          "validation/target_coverage": 0.7573842358060529,
          "validation/queries": 555677,
          "validation/hits": 2655,
          "validation/evaluation_seconds": 174.10725390400057,
          "validation/catalog_size": 4371868,
          "validation/ranking": "exhaustive FP32; raw ID ascending tie break; repeats allowed; uncovered targets count as misses"
        },
        {
          "epoch": 13,
          "train_step": 67834,
          "validation/improved": false,
          "validation/epochs_without_improvement": 2,
          "validation/stopped_early": false,
          "validation/recall_at_100": 0.004768957505889212,
          "validation/ndcg_at_100": 0.001176592878817976,
          "validation/target_coverage": 0.7573842358060529,
          "validation/queries": 555677,
          "validation/hits": 2650,
          "validation/evaluation_seconds": 174.2384008879999,
          "validation/catalog_size": 4371868,
          "validation/ranking": "exhaustive FP32; raw ID ascending tie break; repeats allowed; uncovered targets count as misses"
        },
        {
          "epoch": 14,
          "train_step": 73052,
          "validation/improved": true,
          "validation/epochs_without_improvement": 0,
          "validation/stopped_early": false,
          "validation/recall_at_100": 0.005139676466724374,
          "validation/ndcg_at_100": 0.0012306225651929504,
          "validation/target_coverage": 0.7573842358060529,
          "validation/queries": 555677,
          "validation/hits": 2856,
          "validation/evaluation_seconds": 174.17385512300007,
          "validation/catalog_size": 4371868,
          "validation/ranking": "exhaustive FP32; raw ID ascending tie break; repeats allowed; uncovered targets count as misses"
        },
        {
          "epoch": 15,
          "train_step": 78270,
          "validation/improved": false,
          "validation/epochs_without_improvement": 1,
          "validation/stopped_early": false,
          "validation/recall_at_100": 0.004594395665107607,
          "validation/ndcg_at_100": 0.0011222338914178643,
          "validation/target_coverage": 0.7573842358060529,
          "validation/queries": 555677,
          "validation/hits": 2553,
          "validation/evaluation_seconds": 174.09760979800012,
          "validation/catalog_size": 4371868,
          "validation/ranking": "exhaustive FP32; raw ID ascending tie break; repeats allowed; uncovered targets count as misses"
        },
        {
          "epoch": 16,
          "train_step": 83488,
          "validation/improved": false,
          "validation/epochs_without_improvement": 2,
          "validation/stopped_early": false,
          "validation/recall_at_100": 0.004707770881285351,
          "validation/ndcg_at_100": 0.0011374941296052466,
          "validation/target_coverage": 0.7573842358060529,
          "validation/queries": 555677,
          "validation/hits": 2616,
          "validation/evaluation_seconds": 174.05002325700116,
          "validation/catalog_size": 4371868,
          "validation/ranking": "exhaustive FP32; raw ID ascending tie break; repeats allowed; uncovered targets count as misses"
        },
        {
          "epoch": 17,
          "train_step": 88706,
          "validation/improved": false,
          "validation/epochs_without_improvement": 3,
          "validation/stopped_early": true,
          "validation/recall_at_100": 0.004562002746199681,
          "validation/ndcg_at_100": 0.0011102385562611439,
          "validation/target_coverage": 0.7573842358060529,
          "validation/queries": 555677,
          "validation/hits": 2535,
          "validation/evaluation_seconds": 174.49152690700066,
          "validation/catalog_size": 4371868,
          "validation/ranking": "exhaustive FP32; raw ID ascending tie break; repeats allowed; uncovered targets count as misses"
        }
      ],
      "early_stopping": {
        "event": "early_stopping",
        "epoch": 17,
        "patience": 3,
        "best_epoch": 14,
        "best_recall_at_100": 0.005139676466724374,
        "best_ndcg_at_100": 0.0012306225651929504
      }
    },
    "popularity-independent-verification.json": {
      "method": "Independent DuckDB GROUP BY training events, exact daily catalog ORDER BY/count DESC/ID ASC, validation LEFT JOIN Top100",
      "queries": 555677,
      "hits": 890,
      "recall_at_100": 0.001601649879336377,
      "ndcg_at_100": 0.00033205040744382317,
      "all_popularity_counts_match": true,
      "all_top100_lists_match": true,
      "brute_force_random_query_checks": 5000,
      "test_evaluated": false
    }
  },
  "configs": {
    "two-tower-mean-baseline-config.yaml": {
      "experiment": {
        "seed": 42,
        "dataset": "KuaiRand-1K"
      },
      "model": {
        "item_categorical_features": [
          "item_id",
          "author_id",
          "video_type",
          "upload_type",
          "music_id",
          "music_type",
          "tag"
        ],
        "item_numeric_features": [
          "video_duration",
          "server_width",
          "server_height",
          "upload_day"
        ],
        "category_dim": 8,
        "user_dim": 32,
        "tab_dim": 8,
        "output_dim": 64,
        "history_length": 50,
        "history_encoder": "mean",
        "user_hidden": [
          128,
          64
        ],
        "item_hidden": [
          64
        ],
        "batch_norm": false,
        "dropout": 0.0,
        "residual_blocks": 0,
        "attention_heads": 4,
        "behavior_content_features": true
      },
      "training": {
        "device": "cuda",
        "deterministic": true,
        "mixed_precision": true,
        "batch_size": 256,
        "epochs": 3,
        "learning_rate": 0.001,
        "betas": [
          0.9,
          0.999
        ],
        "weight_decay": 0.0,
        "temperature": 0.1,
        "log_every": 50,
        "checkpoint_every": 1000
      },
      "evaluation": {
        "candidate_rule": "upload_next_day",
        "query_batch_size": 128,
        "item_batch_size": 16384,
        "k": 100,
        "output_dim": 64
      },
      "wandb": {
        "mode": "online",
        "entity": "SAR-Lab",
        "project": "kuairand-two-tower",
        "name": "1k-features-v2-baseline-seed42",
        "group": "week2-1k-content",
        "tags": [
          "1k",
          "content-15-features",
          "manual",
          "baseline",
          "behavior-content-v2"
        ]
      }
    },
    "attention-logq-epochs20-es3-config.yaml": {
      "experiment": {
        "seed": 42,
        "dataset": "KuaiRand-1K"
      },
      "model": {
        "item_categorical_features": [
          "item_id",
          "author_id",
          "video_type",
          "upload_type",
          "music_id",
          "music_type",
          "tag"
        ],
        "item_numeric_features": [
          "video_duration",
          "server_width",
          "server_height",
          "upload_day"
        ],
        "category_dim": 8,
        "user_dim": 32,
        "tab_dim": 8,
        "output_dim": 64,
        "history_length": 50,
        "history_encoder": "attention",
        "user_hidden": [
          256,
          128,
          64
        ],
        "item_hidden": [
          128,
          64
        ],
        "batch_norm": true,
        "dropout": 0.1,
        "residual_blocks": 1,
        "attention_heads": 4,
        "behavior_content_features": true
      },
      "training": {
        "device": "cuda",
        "deterministic": true,
        "mixed_precision": true,
        "batch_size": 256,
        "epochs": 20,
        "learning_rate": 0.001,
        "betas": [
          0.9,
          0.999
        ],
        "weight_decay": 0.0,
        "temperature": 0.1,
        "log_every": 50,
        "checkpoint_every": 1000,
        "early_stopping_patience": 3,
        "sampling_correction": "logq"
      },
      "evaluation": {
        "candidate_rule": "upload_next_day",
        "query_batch_size": 128,
        "item_batch_size": 16384,
        "k": 100,
        "output_dim": 64
      },
      "wandb": {
        "mode": "online",
        "entity": "SAR-Lab",
        "project": "kuairand-two-tower",
        "name": "1k-attention-logq-d01-lr0.001-bs256-e20-es3-seed42",
        "group": "week2-1k-content",
        "tags": [
          "1k",
          "content-15-features",
          "manual",
          "deeper",
          "behavior-content-v2",
          "attention",
          "residual",
          "batchnorm",
          "combined",
          "dropout-0.1",
          "epochs-20",
          "early-stopping-3",
          "logq"
        ]
      }
    }
  }
}
```

</details>
