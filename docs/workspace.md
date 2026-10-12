# Workspace and repository guide

The canonical checkout on this machine is `/Users/ricky/Desktop/Rocommendation`.
Run repository commands here. Scripts resolve project paths from their source
location; other developers can clone the repository anywhere.

The Desktop checkout was on an older `MF-frontend` branch while KuaiRand work
used isolated Git worktrees. PRs #38 and #39 are now merged into `main`.
The `codex/workspace-organization` branch integrates that main state, retains
the frontend branch history, and contains the directory changes and updated
Week 1 report. Switching to an older branch also restores that branch's older
files; it does not reveal missing work.

The former `kuairand-layout` and `kuairand-week1` worktrees have been archived
with recoverable Git snapshots. Continue edits in the Desktop checkout.

## Commands

```bash
cd /Users/ricky/Desktop/Rocommendation
npm ci
npm run dev
npm run build
npm test
uv sync --locked
uv run pytest
# Fallback when pytest startup hangs in the desktop runtime:
uv run python scripts/run_python_tests.py
uv sync --locked --directory research/kuairand/training --python 3.11.15
uv run --directory research/kuairand/training python run.py check-config
uv run --directory research/kuairand/training pytest tests -q
```

The repository research environment and KuaiRand training environment have
separate Python requirements and lockfiles. Do not copy virtual environments
between checkouts: interpreter paths and entry-point shebangs can be absolute.
The GPU notebook remains
`research/kuairand/training/notebooks/week1_baseline.ipynb`; moving the local
checkout does not change Colab's `/content` or mounted Drive paths.

## Data and reports

Local datasets remain inside their project's ignored `data/` directory.
KuaiRand private datasets, checkpoints, vectors and per-request predictions
remain in shared Drive. Public aggregate evidence is stored in
`research/kuairand/reports/week1/evidence/`; the editable report is
`research/kuairand/reports/week1/technical_report_draft.tex`.

Archived August handoffs are under `docs/archive/`. GoodBooks plans and team
instructions are under `research/goodbooks-mf/docs/`; the existing short-video
plan is preserved under `research/kuairand/docs/project-plan.md`.

`node_modules/`, `.venv/`, `.astro/`, `dist/`, Python bytecode and test caches
are generated tooling directories, not additional project source. They can be
regenerated using the commands above. License and third-party attribution
documents remain at the repository root.
