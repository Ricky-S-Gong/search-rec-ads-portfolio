# Goodreads Poetry matrix factorization

Reproducible Goodreads Book Graph experiment implementing the shared frozen
data pipeline plus Basic MF, FunkSVD, sparse explicit ALS, masked NMF, and an
additional bias-aware residual ALS diagnostic.

Detailed documentation:

- [`ZIQI_IMPLEMENTATION_GUIDE.md`](ZIQI_IMPLEMENTATION_GUIDE.md): complete
  implementation walkthrough and interview preparation.
- [`TEAM_HANDOFF_RICKY_YUTAO.md`](TEAM_HANDOFF_RICKY_YUTAO.md): immutable data
  contract and step-by-step continuation instructions for teammates and Codex.

## Reproduce the environment and data

From the repository root:

```bash
uv sync --locked
uv run python research/goodbooks-mf/download_data.py
uv run python research/goodbooks-mf/prepare_data.py
uv run python research/goodbooks-mf/verify_data.py
uv run python research/goodbooks-mf/run_experiment.py
uv run pytest
```

`run_experiment.py` is Ziqi's original Basic MF/FunkSVD result reproduction and
reads the frozen test split. Teammates who are still selecting new model
configurations must not use that script for tuning.

## Phase-one unified validation runner

Yutao's phase-one runner trains candidates on the frozen train split and ranks
them using explicit-rating RMSE or MAE from the frozen validation split only:

```bash
uv run python research/goodbooks-mf/verify_data.py
uv run python research/goodbooks-mf/run_all_experiments.py
```

The committed candidate schema is `experiment_config.json`. It fixes dataset
version `goodreads-poetry-v1`, seed `20260830`, rating clipping to `[1, 5]`, and
the auditable candidate list for each local model. A candidate cannot override
the top-level seed.

The runner:

1. verifies the bundle against `canonical_manifest.json`;
2. deserializes only `train.parquet`, `validation.parquet`, and
   `train_explicit.npz`;
3. trains every configured candidate on train;
4. selects within each model using validation only;
5. writes complete phase-one rows to `results/validation_selection.json`.

The test artifact is checksum-verified as part of the bundle, but
`test.parquet` is not deserialized. Future test access is gated behind a frozen
configuration with a matching dataset version, seed, and configuration hash.
The phase-one command intentionally exposes no final-test option.

Current validation-only selected rows are:

| Model | Validation RMSE | Validation MAE |
|---|---:|---:|
| Basic MF | 1.022825 | 0.788908 |
| FunkSVD | 0.844779 | 0.655388 |
| Sparse explicit ALS | 1.014981 | 0.755570 |
| Masked NMF with optional L2 regularization | 0.897692 | 0.675072 |
| Bias-aware residual ALS | 0.843333 | 0.661293 |

These are validation-selection results, not final test metrics. Hyperparameters
are selected from these validation rows only. Final rating and ranking metrics
use the shared evaluation implementation described below; no model owns a
separate Top-K definition.

## Yutao frozen rating test

After the validation configuration review, the selected ALS, masked NMF, and
bias-aware ALS configurations were frozen under seed `20260830` with config
hash:

```text
c9c986bd86acfe24af91b465e773ac9039e16bbb97e51c8e66858aa6fc358cac
```

The one-time explicit-rating test evaluated 17,168 ratings and produced:

| Model | Test RMSE | Test MAE |
|---|---:|---:|
| Sparse explicit ALS | 1.119088 | 0.828761 |
| Masked NMF with L2 | 0.984062 | 0.719640 |
| Bias-aware residual ALS | 0.861570 | 0.675045 |

The immutable inputs and aggregate results are recorded in
`results/yutao_frozen_config.json` and `results/yutao_test_metrics.json`.
The test-result command refuses to replace an existing frozen configuration or
result file. These test metrics must not be used for further tuning. The shared
evaluation runner extends this frozen result with ranking metrics without
changing the selected hyperparameters.

## Shared final evaluation for Yutao's models

Ricky's shared evaluation is integrated without changing the frozen validation
selection. Run it once with:

```bash
uv run python research/goodbooks-mf/run_unified_evaluation.py
```

It constructs one shared `RankingEvaluationData` object and reuses it for ALS,
NMF with L2, and bias-aware ALS. The frozen full-data results evaluate 17,168
explicit ratings from 5,129 users and ranking candidates for 4,765 eligible
users:

| Model | RMSE | MAE | Precision@10 | Recall@10 | NDCG@10 | Train seconds |
|---|---:|---:|---:|---:|---:|---:|
| Sparse explicit ALS | 1.119088 | 0.828761 | 0.001448 | 0.005637 | 0.002728 | 0.814 |
| Masked NMF + L2 | 0.984062 | 0.719640 | 0.000441 | 0.001193 | 0.000835 | 0.931 |
| Bias-aware residual ALS | 0.861570 | 0.675045 | 0.002036 | 0.007762 | 0.005978 | 0.812 |

The complete metrics at K = 5, 10, and 20, timing fields, protocol metadata,
and frozen config hash are stored in
`results/yutao_unified_test_metrics.json`. The compact table is
`results/yutao_unified_test_metrics.csv`; the generated comparison figure is
`results/yutao_unified_test_metrics.png`. These test results are for final
reporting only and must not be used to revise hyperparameters.

## Current team comparison

Ziqi's committed Basic MF and FunkSVD implementations were reproduced with
their frozen configuration and Ricky's shared rating/ranking evaluation. The
rating results exactly reproduce the previously published aggregate values.
Build the current team summary with:

```bash
uv run python research/goodbooks-mf/run_experiment.py \
  --output research/goodbooks-mf/results/ziqi_unified_test_metrics.json
uv run python research/goodbooks-mf/build_team_summary.py
```

| Model | Owner | RMSE | MAE | Precision@10 | Recall@10 | NDCG@10 | Train seconds |
|---|---|---:|---:|---:|---:|---:|---:|
| Basic MF | Ziqi | 1.149409 | 0.882003 | **0.008646** | **0.036178** | **0.023947** | 26.086 |
| FunkSVD | Ziqi | **0.861341** | **0.668944** | 0.001994 | 0.007369 | 0.004920 | 14.270 |
| ALS | Yutao | 1.119088 | 0.828761 | 0.001448 | 0.005637 | 0.002728 | 0.814 |
| NMF + L2 | Yutao | 0.984062 | 0.719640 | 0.000441 | 0.001193 | 0.000835 | 0.931 |
| Bias-aware ALS | Yutao, diagnostic | 0.861570 | 0.675045 | 0.002036 | 0.007762 | 0.005978 | 0.812 |

All rows use 17,168 explicit test ratings, 5,129 rating users, 4,765 ranking
users, and the 5,551-item train catalog. Basic MF has the weakest rating error
but the strongest Top-K result, so rating prediction and ranking quality must
be discussed separately. `results/team_model_comparison.{json,csv,png}` stores
the normalized table and chart. The planned SVD++ row remains pending Ricky's
implementation and shared-evaluation output; bias-aware ALS is explicitly
marked as an additional diagnostic rather than a substitute for SVD++.

The committed `config.json` is the data contract. It fixes the random seed,
k-core thresholds, maximum user count, and temporal split fractions. Every
model must consume the generated bundle and must not independently filter or
split the source.

The processed schema is:

```text
user_idx, item_idx, rating, is_read, is_reviewed, event_time, split
```

`rating > 0` is explicit feedback. Implicit feedback is exactly
`is_read OR is_reviewed OR rating > 0`. The split is chronological per user and
uses explicit-rating boundaries so train, validation, and test each receive an
explicit observation before cold validation/test items are removed.

`manifest.json` records the configuration, seed, table counts, and SHA-256 of
every Parquet and CSR file. The repository's metadata-only
`canonical_manifest.json` freezes those expected values. Teammates should
obtain the canonical processed bundle through access-controlled course storage
and run `verify_data.py` before training. A checksum or canonical-manifest
mismatch is a hard failure.

## Data policy

Raw and processed Goodreads records live below `research/goodbooks-mf/data/`
and are gitignored. The UCSD dataset is for academic use only and asks users not
to redistribute it or use it commercially. Do not commit the dataset or publish
the shared download location. The public repository contains code,
configuration, aggregate metrics, and documentation only.

## Model contract

Basic MF and FunkSVD train over observed `(user_idx, item_idx, rating)`
triplets. ALS, NMF, and bias-aware ALS consume the frozen sparse explicit CSR
matrix. Missing CSR positions are never interpreted as rating zero. `FunkSVD`
implements:

```text
prediction = global_mean + user_bias + item_bias + user_factors @ item_factors
```

Validation RMSE controls SGD early stopping and restores the best epoch. All
models expose raw-score prediction and deterministic recommendation methods;
rating evaluation clips only at metric time. Shared
rating and ranking metrics live in `goodbooks_mf/evaluation.py` and are called
for every model. Rating predictions are clipped to `[1, 5]`; ranking always
uses raw scores.

The ranking protocol evaluates every eligible test user against the complete
train catalog after removing that user's train interactions. Relevance is
`rating >= 4 OR (rating == 0 AND is_read)`, exact score ties are broken by
ascending `item_idx`, and Precision, Recall, and NDCG are reported at 5, 10,
and 20. There is no negative sampling or row-level candidate artifact. Call
`prepare_ranking_data(train, test)` once and pass the returned
`RankingEvaluationData` to `evaluate_ranking` for each model so every model is
scored on the same protocol.
