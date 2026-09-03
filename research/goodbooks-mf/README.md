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

These are validation-selection results, not final test metrics. Precision,
Recall, NDCG, and evaluated ranking users remain `null` until Ricky's shared
evaluation functions and frozen candidate artifact are available. Do not
implement a parallel temporary Top-K definition. The subsequent frozen rating
test is documented below.

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
result file. These test metrics must not be used for further tuning. Ranking
metrics remain pending Ricky's shared evaluation and candidate artifact.

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
rating evaluation clips only at metric time. Precision, Recall, and NDCG will
be added by the shared evaluation owner once the frozen candidate set is
available.
