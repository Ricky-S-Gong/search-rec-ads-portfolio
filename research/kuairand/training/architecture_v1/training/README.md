# Architecture v1: preparation / Stage 1 / B0+R0 controls

Use `../START_HERE_Ziqi_Stage12.ipynb` and `../README_FIRST_中文.md`.
The source is adapted from the actual Drive 1K ablation-colab-v3 release, not the repository's older Pure implementation.

- Entry: `architecture.py matrix|preflight|probe|train|summary|feedback`.
- IDs: B0/R0/A1/A2/A3/A5; A4 aliases B0. Shared seeds: 42, 2026, 3407.
- This notebook prepares all six architectures but starts only B0/R0 seed42, one at a time.
- Existing `run_ablation.py` is historical; its obsolete command-line matrix is disabled. Upstream files are retained for traceability only.
- New full training runs require an empty run directory; resumes require identical config, dataset, source and runtime binding.
- Every epoch saves `epoch-NNN.pt`, `latest.pt`, `best.pt` on improvement, metrics and ranking evidence.
- Online mode uploads epoch checkpoints as W&B Artifacts; failed uploads are retried from persistent checkpoints on resume.
- Validation loss uses fixed stored-order validation events, batches of 256, FP32, no correction; training loss is an event-weighted epoch average.
- Frozen dependency versions are inherited through `pyproject.toml` and `uv.lock`. Colab uses Python 3.11 via `uv sync --locked`.
- Keep data and weights out of public Git. `feedback` collects small aggregate evidence only.

Local verification uses an older CPU environment with a test-only scaler shim; this does not replace locked-runtime CUDA and W&B acceptance. See `../verification/VERIFICATION.md`.
