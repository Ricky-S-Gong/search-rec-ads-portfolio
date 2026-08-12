# MovieLens collaborative filtering

CPU benchmark comparing mean-centered User-CF and adjusted-cosine Item-CF on
the official MovieLens 1M ratings.

```bash
uv sync --locked
uv run python research/movielens-cf/download_data.py
uv run python research/movielens-cf/run_experiment.py
.venv/bin/python research/run_python_tests.py
```

For a fast end-to-end check without replacing the published artifacts:

```bash
uv run python research/movielens-cf/run_experiment.py --smoke \
  --output /tmp/movielens-smoke
```

The experiment uses a deterministic per-user temporal 80/10/10 split. It
selects neighborhood settings on validation data, refits on train plus
validation, and evaluates test data once. Top-10 metrics use every unseen movie
in the fitted catalog; no sampled negatives are used.

Top-N ordering uses raw neighborhood estimates. Values clipped to the 1–5
rating scale are reserved for RMSE/MAE and display, preventing distinct scores
above five from collapsing into movie-ID tie breaks. The generated artifacts
include popularity, User-CF, and Item-CF recommendation lists, held-out hits,
and neighbor evidence for the browser explorer.

Exact Item-CF ties are reported rather than hidden. With positive adjusted
cosine weights, a user's all-five-star contributing history can produce an
exact raw estimate of 5.0 for many candidates; equal scores use movie ID as the
deterministic secondary key. Artifacts expose contributing-neighbor count and
total absolute similarity weight as confidence context. The experiment also
reports a paired user bootstrap for User-CF minus popularity.

GitHub Actions continues to run `uv run pytest`. In the audited Codex desktop
runtime, pytest hangs before collection, including with plugin autoload and the
terminal reporter disabled. The repository fallback runner above imports and
executes the same existing Python test functions and supports their current
`tmp_path` fixture usage.

Raw MovieLens files remain under `research/movielens-cf/data/`, which is
gitignored. GroupLens does not permit redistribution without separate
permission. The website receives only compact, derived JSON and SVG artifacts
under `public/artifacts/movielens/`.
