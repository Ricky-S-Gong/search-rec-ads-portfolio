# SAR Portfolio Handoff

Last updated: 2026-08-17

This document is the starting context for the next Codex session. Read it before changing the site or adding a project.

## Repository snapshot

- Repository: `https://github.com/Ricky-S-Gong/search-rec-ads-portfolio`
- Local path: `/Users/ricky/Desktop/Rocommendation`
- Production site: `https://ricky-s-gong.github.io/search-rec-ads-portfolio/`
- Current branch: `main`
- Current commit: `7af273a` (`fix: use Yutao Rao across about page (#19)`)
- Local `main` started from `origin/main`. The current worktree contains uncommitted updates that remove Search/Ads candidate projects, mark MovieLens as in progress, keep its detail routes available, and revise this handoff file.
- Site languages: English under `/en/`, Chinese under `/zh/`.

Do not assume GitHub Actions or the deployed site is healthy without checking them in the new session.

## Product state

The site is an interview-oriented Search, Advertising, and Recommendation portfolio. It uses bilingual case studies to connect:

- business problems and data limitations;
- reproducible experiments and generated artifacts;
- algorithm choice, mathematical intuition, code, and limitations;
- offline proxy metrics versus real online business metrics;
- author contributions and reproducibility instructions.

The current public authors/About profiles are Ricky Gong, Ziqi Xu, and Yutao Rao. Project-level author fields still reflect the people who worked on each project.

### Project status

| Domain | Project | Status | Compute | Key state |
| --- | --- | --- | --- | --- |
| Recommendation | Spotify content recommendation | Completed | CPU | Content-based retrieval over nine audio features, with reproducible artifacts and an interactive weighted-cosine lab. |
| Recommendation | MovieLens collaborative filtering | In progress | CPU | Bayesian popularity, User-CF, and Item-CF are implemented, but the case study still needs content and presentation refinements before completion. |
| Search | Not selected | TBD | TBD | Previous candidate projects were removed; choose a dataset and project before implementation. |
| Advertising | Not selected | TBD | TBD | Previous candidate projects were removed; choose a dataset and project before implementation. |

The next Recommendation roadmap stage is model-based recommendation, followed by retrieval/ranking/online systems. Do not invent results for planned projects.

## Important experimental conclusions

### Spotify

Source of truth: `public/artifacts/spotify/metrics.json` and the scripts under `research/spotify-music/`.

- Dataset: 170,653 tracks, 1921–2020, nine audio features.
- Exact cosine median latency: 1.6629 ms.
- K-Means candidate retrieval median latency: 0.3168 ms.
- Cluster Recall@10 against exact cosine: 0.7775.
- Euclidean KNN versus cosine Top-10 overlap: 0.4075.
- Diversity@10: 0.003558; the lists are highly concentrated, not broadly exploratory.
- These are proxy and system metrics. There are no user relevance labels, so do not report Precision@K, NDCG, or preference probability.
- Displayed similarity is weighted cosine multiplied by 100 and formatted as a percentage. It is not a probability or accuracy score; ranking still uses the unchanged raw cosine value.

### MovieLens

Source of truth: `public/artifacts/movielens/metrics.json` and the scripts under `research/movielens-cf/`.

- Dataset: MovieLens 1M, 1,000,209 explicit ratings.
- Evaluation: per-user temporal 80/10/10 split, Top-10 from the full fitted catalog after excluding seen items, relevance threshold rating >= 4.
- Artifact schema: `movielens-cf-v3`, with exact split counts, source-derived field examples, Bayesian-prior examples, per-user hits, and bootstrap confidence intervals.
- Split: 797,758 train ratings, 99,692 validation ratings, 897,450 fitted ratings, and 102,759 test ratings; final ranking uses 3,683 fitted-catalog movies.
- Bayesian popularity: Hit Rate@10 0.15275, NDCG@10 0.02676, Recall@10 0.02247, catalog coverage 0.0209.
- User-CF: Hit Rate@10 0.13935, NDCG@10 0.02368, Recall@10 0.02099, coverage 0.2748, long-tail share 0.4201, RMSE 0.9313.
- Item-CF: Hit Rate@10 0.06667, NDCG@10 0.01057, Recall@10 0.01252, coverage 0.7372, long-tail share 0.5689, RMSE 0.9652.
- Main finding: popularity is the strongest accuracy baseline in this experiment, while personalized CF methods broaden catalog and long-tail exposure. There is no single universal winner.
- Item-CF has exact score ties from sparse five-star evidence: 77.96% of personalized Top-10 lists are fully tied and 87.86% of entries have five-star-only evidence. Top-N uses raw estimates; clipped estimates are only for RMSE/MAE. Keep this distinction intact.
- These are offline explicit-rating results, not CTR, watch time, retention, revenue, or causal lift.

## Architecture and source-of-truth rules

- Astro 5 static site with TypeScript.
- React 19 islands only for interactive experiences, notably `SpotifyLab.tsx` and `MovieLensExplorer.tsx`.
- KaTeX for math; local IBM Plex and Noto fonts.
- Python 3.11+, pandas, NumPy, SciPy, scikit-learn, and Matplotlib for experiments.
- GitHub Pages deployment through `.github/workflows/ci-pages.yml`.
- No backend, database, online inference service, paid hosting, or external runtime CDN.

Key locations:

```text
src/pages/[lang]/                 # bilingual routes
src/components/                   # Astro and React UI
src/components/cosmos/            # layered cosmic hero scene
src/data/site.ts                  # projects, profiles, roadmap, shared copy
src/styles/global.css             # shared design system
public/artifacts/spotify/         # generated Spotify web artifacts
public/artifacts/movielens/       # generated MovieLens web artifacts
research/spotify-music/           # reproducible Spotify pipeline and tests
research/movielens-cf/            # reproducible MovieLens pipeline and tests
.github/workflows/ci-pages.yml    # PR checks and Pages deployment
```

Routing details:

- `src/pages/[lang]/projects/[slug].astro` builds completed and in-progress project pages and selects the Spotify or MovieLens case-study component by slug.
- `src/pages/[lang]/algorithms/[slug].astro` currently builds full detail pages for content-based filtering, User-CF, and Item-CF.
- The homepage and algorithm comparison page organize content by Search, Ads, and Recommendation.

Non-negotiable content rule: experiment scripts generate metrics, charts, and demo data. Pages consume those artifacts. Never hand-edit a published number to make the story cleaner.

## Visual and interaction state

- The homepage uses a full-bleed layered cosmic scene with stars, planets, orbits, asteroids, and reduced-motion support.
- NASA planetary imagery and the Spotify logo are stored locally; attribution and usage notes live in `THIRD_PARTY_NOTICES.md`.
- Search, Recommendation, and Ads retain their established cyan, purple, and gold domain colors.
- Spotify includes weighted feature controls, search, result selection, percentage-formatted cosine similarity, and a radar comparison.
- MovieLens includes a data-driven explorer for comparing recommendation approaches and their trade-offs.
- Preserve keyboard access, responsive layouts, reduced-motion behavior, and page-level overflow protection.

## Recent collaborator work

- PR #15 by Ziqi Xu added the initial MovieLens collaborative-filtering study: research modules, deterministic temporal evaluation, artifacts, bilingual case study, explorer, algorithm pages, and tests. The project remains in progress while review feedback is addressed.
- PRs #13 and #14 by Ziqi Xu decomposed the cosmic hero into reusable layers, added NASA planetary assets and parallax, and replaced asteroid assets with transparent WebP files plus alpha-regression tests.
- PR #17 pinned the uv version to fix GitHub Actions setup.
- PRs #18 and #19 added Yutao Rao to the About page and corrected his displayed name.

Do not overwrite collaborator changes when resolving conflicts. Inspect the latest diff and preserve unrelated work.

## Local setup and validation

```bash
npm install
npm run dev

uv sync --locked

npm test
npm run check
npm run build
uv run pytest
git diff --check
```

The audited Codex desktop environment has previously shown cases where the `pytest` launcher hangs before collection. If that recurs, use the repository fallback runner:

```bash
.venv/bin/python research/run_python_tests.py
```

The fallback runs the same project test modules; document which command was used. Do not claim tests passed unless they were run in the current session.

Useful experiment commands are documented in the project READMEs. Raw datasets must remain outside Git:

```bash
uv run python research/spotify-music/download_data.py
uv run python research/spotify-music/run_experiment.py

uv run python research/movielens-cf/download_data.py
uv run python research/movielens-cf/run_experiment.py
```

MovieLens redistribution restrictions and Spotify/Kaggle licensing uncertainty are reasons to commit only code and derived, compact artifacts.

## Collaboration and release workflow

1. Pull the latest `main` and inspect `git status` before editing.
2. Create a GitHub Issue and a branch named `feature/<issue>-<short-name>` unless the current repository convention has changed.
3. Implement both English and Chinese versions together.
4. Run JS tests, Astro checks/build, Python tests, and `git diff --check`.
5. Push and open a PR. One collaborator approval is required for the PR, not for every commit.
6. Merge only after CI passes.
7. A merge to `main` triggers the GitHub Pages deployment job. Wait for it to succeed, then smoke-test both language routes online.

`CODEOWNERS` includes the collaborators. Never commit Kaggle, GitHub, Spotify, Colab, or other credentials.

## Licensing

- Code: MIT (`LICENSE`).
- Original writing and charts: CC BY 4.0 (`CONTENT_LICENSE.md`).
- Third-party notices: `THIRD_PARTY_NOTICES.md`.
- Raw data, archives, CSV files, and environment files are excluded through `.gitignore`.

## Known follow-ups

1. `README.md` still introduces the portfolio as “Ricky Gong & Ziqi Xu,” while the live site data/About page now includes Yutao Rao. Confirm the intended authorship wording before updating the README.
2. Search and Ads do not have selected projects. Do not restore previous candidates or add placeholders; select datasets and project briefs before implementation.
3. Before adding another project, confirm dataset license, evaluation labels, compute requirements, and whether GPU/Colab is needed.
4. Keep bilingual metadata and prose synchronized, especially project status, metrics, caveats, and contribution fields.
5. Recheck Actions and the deployed site at the start of the next session; this handoff does not certify their current external status.

## Reusable Codex skill

A personal skill was created from the Spotify build-and-refinement process and generalized for future Search, Ads, and Recommendation projects:

```text
/Users/ricky/.codex/skills/add-sar-portfolio-project/SKILL.md
```

Invoke it as `$add-sar-portfolio-project` when adding or substantially improving a project. It covers dataset audit, experiment design, artifact generation, bilingual case-study structure, interactive presentation, evidence rules, validation, and PR release. The skill is installed in the user's Codex skills directory and is not tracked by this repository.

## Suggested next-session opening

Ask Codex to read this file, inspect the latest Git/GitHub state, and then state the exact task. For any new portfolio project, explicitly invoke `$add-sar-portfolio-project`.
