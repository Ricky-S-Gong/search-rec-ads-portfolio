# MovieLens collaborative filtering

CPU benchmark comparing mean-centered User-CF and adjusted-cosine Item-CF on
the official MovieLens 1M ratings.

```bash
uv sync --locked
uv run python research/movielens-cf/download_data.py
uv run python research/movielens-cf/run_experiment.py
uv run pytest
```

The experiment uses a deterministic per-user temporal 80/10/10 split. It
selects neighborhood settings on validation data, refits on train plus
validation, and evaluates test data once. Top-10 metrics use every unseen movie
in the fitted catalog; no sampled negatives are used.

Raw MovieLens files remain under `research/movielens-cf/data/`, which is
gitignored. GroupLens does not permit redistribution without separate
permission. The website receives only compact, derived JSON and SVG artifacts
under `public/artifacts/movielens/`.
