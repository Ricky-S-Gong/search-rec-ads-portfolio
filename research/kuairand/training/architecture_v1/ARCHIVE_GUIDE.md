# Frozen architecture-v1 experiment source

This folder preserves the original Stage12 runtime pack, extracted without modifying its training source. It is separate from the repository's existing training implementation. The adjacent seed42 report contains completed experiment evidence; older preparation notes describing GPU runs as pending refer to the original pack's creation time.

## Run in Colab

The original notebook expects `KuaiRand_Stage12_Ready_Pack.zip` with an `architecture_stage12/` archive prefix. To reconstruct that upload from a repository checkout, run this from the repository root:

```python
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
source = Path('research/kuairand/training/architecture_v1')
with ZipFile('KuaiRand_Stage12_Ready_Pack.zip', 'w', ZIP_DEFLATED) as archive:
    for path in sorted(source.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts:
            archive.write(path, 'architecture_stage12/' + path.relative_to(source).as_posix())
```

Open `START_HERE_Ziqi_Stage12.ipynb` in Colab and upload that reconstructed ZIP in Step 1. Use the locked Python 3.11 environment. Drive source files require access to the team's frozen release; see `training/data_release.json`. Put WANDB_API_KEY in Colab Secrets, never in a notebook or commit. No training data or checkpoint weights are included here.

## CLI

From this folder's `training/` directory, using the locked environment and real data path:

```bash
uv sync --locked
uv run python architecture.py matrix
uv run python architecture.py preflight --data /path/to/data --output-root /path/to/runs --ids B0 --seeds 2026
uv run python architecture.py train --data /path/to/data --output-root /path/to/runs --ids B0 --seeds 2026
```

Supported IDs are B0, R0, A1, A2, A3, A5. Supported seeds are 42, 2026, 3407. Repeat preflight and train for the desired config/seed. Runs resume automatically from that task's latest.pt. Keep source/config/data bindings unchanged; after a device change rerun preflight. Remove a stale task.lock only after confirming the previous process/session is stopped. Do not delete checkpoints or run concurrent processes for the same config/seed.

The notebook starts B0/R0 only. Other architectures use the same CLI with their own IDs. Feedback packaging is Step 11. Original pack tests and historical helper scripts are preserved; use architecture.py for this experiment matrix.

[Seed42 results](../../reports/architecture_v1/seed42/README.md)
