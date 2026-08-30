# Search · Ads · Recommendation Portfolio

[中文](#中文) · [Live site](https://ricky-s-gong.github.io/search-rec-ads-portfolio/)

A bilingual, interview-oriented portfolio by **Ricky Gong & Ziqi Xu**. Each
project is a code-driven case study covering the problem, data, EDA, feature
engineering, algorithm choice, results, evaluation, and improvements.

## Local setup

```bash
npm install
npm run dev
```

Research environment and CPU experiments:

```bash
uv sync
uv run python research/spotify-music/download_data.py
uv run python research/spotify-music/run_experiment.py \
  --data research/spotify-music/data/data.csv
uv run python research/movielens-cf/download_data.py
uv run python research/movielens-cf/run_experiment.py
uv run python research/goodbooks-mf/download_data.py
uv run python research/goodbooks-mf/prepare_data.py
uv run python research/goodbooks-mf/verify_data.py
uv run python research/goodbooks-mf/run_experiment.py
.venv/bin/python research/run_python_tests.py
```

CI uses `uv run pytest`. In the audited Codex desktop runtime the pytest
launcher can hang before collection, so the checked-in fallback runner executes
the same existing Python test functions deterministically for local verification.

Raw datasets are gitignored. The site reads only compact, versioned artifacts
from `public/artifacts/`.

## Collaboration

Use GitHub Issues, `feature/<issue>-<name>` branches, and reviewed pull
requests. See [CONTRIBUTING.md](CONTRIBUTING.md).

## 中文

这是 Ricky Gong 与 Ziqi Xu 联合构建的中英双语搜广推算法作品集。每个项目都会明确业务问题、数据边界、算法选择、可复现实验，以及算法适用和不适用的场景。

当前 Spotify 内容推荐、MovieLens 协同过滤与 Goodreads 矩阵分解项目全程使用 CPU。原始数据不会提交到仓库，页面只读取脚本生成的轻量产物。

Code: MIT. Original prose and charts: CC BY 4.0. Third-party material retains
its original terms.
