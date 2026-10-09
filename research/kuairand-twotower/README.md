# KuaiRand two-tower retrieval

An in-progress neural candidate-retrieval project using KuaiRand-1K. The goal
is to retrieve 100 eligible short videos from a user's strictly prior feedback.

## Data and storage

Raw data, processed Parquet files, checkpoints, and WandB runs are private and
gitignored. Store them in the shared Drive; keep this repository for code,
configuration, tests, and compact public results only.

```text
shared-drive/kuairand/
├── raw/KuaiRand-1K.tar.gz
├── datasets/kuairand-1k-v1/
├── runs/<owner>/<run-id>/
└── releases/<stage>/
```

## Get the official 1K archive

```bash
python research/kuairand-twotower/download_data.py \
  --destination /content/drive/MyDrive/kuairand/raw/KuaiRand-1K.tar.gz
```

The downloader validates Zenodo's published MD5. Extract the verified archive
beside it, then use `KuaiRand-1K/data/` as the raw data directory.

## Use KuaiRand-Pure now

Pure is the small smoke-test dataset: it is suitable for validating this data
pipeline, but not for the final retrieval benchmark because its candidate pool
and sequential history are reduced. After extracting your downloaded archive,
run:

```bash
uv run python research/kuairand-twotower/prepare_kuairand_data.py \
  --raw-dir /content/drive/MyDrive/kuairand/raw/KuaiRand-Pure/data \
  --output-dir /content/drive/MyDrive/kuairand/datasets/kuairand-pure-v1
```

Use the same command with `KuaiRand-1K/data` later for the main experiment.

## Build user histories

Each non-random long-view training target gets up to its 50 most recent
**strictly prior**, non-random long-view video IDs. This prevents the model
from seeing a request's own outcome or another request from exactly the same
timestamp.

```bash
PYTHONPATH=research/kuairand-twotower python research/kuairand-twotower/build_user_histories.py \
  --events /content/drive/MyDrive/kuairand/datasets/kuairand-pure-v1/events.parquet \
  --output /content/drive/MyDrive/kuairand/datasets/kuairand-pure-v1/user_histories.parquet
```

## Shared data contract

`normalize_events` emits `request_id`, `user_id`, `item_id`,
`request_time_ms`, `tab`, `long_view`, and `is_random`. The official `time_ms`
field is already a Unix timestamp in milliseconds. Time splits are fixed:
warmup through Apr 14, train Apr 15–28, validation Apr 29–May 3, and test from
May 4. Random-exposure rows are reserved for diagnostics, never training.

## Colab setup

```python
from google.colab import drive
drive.mount('/content/drive')
!git clone https://github.com/Ricky-S-Gong/search-rec-ads-portfolio.git
%cd search-rec-ads-portfolio
!pip install -r research/kuairand-twotower/requirements-colab.txt
```

Do not commit `.env`, API keys, raw CSVs, Parquet files, or checkpoints.
