# Goodreads Poetry matrix factorization

Reproducible Goodreads Book Graph experiment implementing Ziqi's project scope:
data ingestion, cleaning, k-core filtering, a frozen temporal split, shared
Parquet/CSR artifacts, Basic MF, and biased FunkSVD.

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

Both models train directly over observed `(user_idx, item_idx, rating)`
triplets; they never construct or scan a dense user-item matrix. `FunkSVD`
implements:

```text
prediction = global_mean + user_bias + item_bias + user_factors @ item_factors
```

Validation RMSE controls early stopping and restores the best epoch. Shared
rating and ranking metrics live in `goodbooks_mf/evaluation.py` and are called
by `run_experiment.py` for every model. Rating predictions are clipped to
`[1, 5]`; ranking always uses raw scores.

The ranking protocol evaluates every eligible test user against the complete
train catalog after removing that user's train interactions. Relevance is
`rating >= 4 OR (rating == 0 AND is_read)`, exact score ties are broken by
ascending `item_idx`, and Precision, Recall, and NDCG are reported at 5, 10,
and 20. There is no negative sampling or row-level candidate artifact. Call
`prepare_ranking_data(train, test)` once and pass the returned
`RankingEvaluationData` to `evaluate_ranking` for each model so every model is
scored on the same protocol.
