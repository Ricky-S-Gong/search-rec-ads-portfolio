# MovieLens collaborative filtering

CPU benchmark comparing Bayesian popularity, bias-aware User-CF, and
bias-aware Item-CF on the official MovieLens 1M ratings. The published v4
artifacts retain the mean-centered v3 models as a like-for-like diagnostic
baseline.

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

The v4 metrics distinguish Hit Rate@10 from Recall@10. A user has a hit when
their Top-10 contains at least one test movie rated four or five stars; Hit
Rate is the share of evaluated users with a hit. Recall instead divides each
user's retrieved relevant movies by all of that user's relevant test movies.
All three methods report user-bootstrap 95% confidence intervals for Hit Rate,
NDCG, and Recall. Artifacts also include exact split counts, a source-derived
data dictionary, and reproducible Bayesian-prior calculation examples.

The bias baseline is `b_ui = global_mean + user_bias + item_bias`, and every
training residual is `e_ui = r_ui - b_ui`. User-CF computes cosine similarity
between user residual rows; Item-CF computes the equivalent residual cosine
between movie columns. Predictions add the weighted neighbor residual back to
`b_ui`. Top-N ordering uses the unclipped prediction, then Bayesian popularity,
then total absolute similarity evidence, and finally movie ID only as a
deterministic last key. Values clipped to 1–5 are reserved for RMSE/MAE and
display.

The full run first evaluates the orthogonal grid
`k={20,40,80} × min_support={5,10,20} × shrinkage={10,25,50}` with
`min_neighbors=2`, then compares `min_neighbors={2,5}` for the three leading
validation candidates. Selection is lexicographic: Validation NDCG@10,
Recall@10, Hit Rate@10, fully tied list share, then catalog coverage. Bayesian
popularity selects `prior_weight` from `{10,25,50,100}` on Validation only.
Artifacts preserve every candidate, the staged elimination rule, secondary
ordering ablation, exact tie diagnostics, fallback share, and paired user
bootstrap results.

The v4 direction was motivated by tie failures already observed in the v3 test
artifact. It is therefore labeled a post-hoc model iteration rather than a
fully untouched confirmatory experiment. Parameters still use Train and
Validation only; Test is evaluated after configuration lock. `rating >= 5` is
reported only as a read-only sensitivity analysis and never replaces the main
raw-test-rating `rating >= 4` relevance rule.

GitHub Actions and the local release check run `uv run pytest`. `pyproject.toml`
pins the uv tool requirement separately from Python package dependencies. The repository
fallback runner above imports and executes the same existing Python test
functions and supports their current `tmp_path` fixture usage if a constrained
desktop runtime cannot start pytest normally.

Raw MovieLens files remain under `research/movielens-cf/data/`, which is
gitignored. GroupLens does not permit redistribution without separate
permission. The website receives only compact, derived JSON and SVG artifacts
under `public/artifacts/movielens/`.
