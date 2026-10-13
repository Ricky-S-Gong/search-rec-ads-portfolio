# KuaiRand-Pure training and Colab / Pure 训练与 Colab

This module publishes the code previously run from a Drive source ZIP. It consumes Ziqi's Pure handoff and pinned Item Tower snapshot; it adds the User Tower, training loop, W&B, checkpoints, validation loss, and export. The existing `../data_pipeline/` data project remains the team data pipeline.

此目录是之前在 Colab 通过源码 ZIP 运行的训练工程。数据处理由 Ziqi 的项目提供，当前训练接入固定版本的 Item Tower，并提供用户塔、训练系统和 Colab 入口。

## Colab

Open [notebooks/week1_baseline.ipynb](notebooks/week1_baseline.ipynb) in Colab. This clean notebook contains no outputs, credentials or datasets. It clones **Ricky-S-Gong/search-rec-ads-portfolio**, resolves the selected branch to an exact commit, then installs this module's own locked environment. The default `REF` is `main`; for an unmerged PR set it to that PR branch; for an exact rerun set `COMMIT` to the printed SHA. Never click Run all: preparation, W&B setup and training are separate manual cells.

在 GitHub notebook 页面点击 Open in Colab，先连接 GPU，按顺序准备和检查，`ready_for_training=true` 后手动运行训练单元。Secret 名称为 `WANDB_API_KEY`，每份 notebook 要单独授权。输出保存到个人 Drive，恢复训练使用相同版本与原输出目录。

[Previously executed notebook / 已执行 notebook](https://colab.research.google.com/drive/1oVUgz912A78T0P6AfV7tsQYF3HmqJZCf) used the source ZIP. Moving it into GitHub changes setup instructions, not the model architecture. For loading the historical checkpoint, use the source ZIP and exact manifest in the shared model package; do not bypass checksum bindings with an arbitrary new checkout.

## Local setup / 本地环境

```bash
cd research/kuairand/training
python3 -m pip install uv==0.12.1
uv sync --locked --python 3.11.15
uv run --locked python run.py check-env
uv run --locked pytest tests -q
```

This isolated lock avoids altering the parent portfolio environment (uv 0.12.1). Local W&B credentials belong in the repository-root `.env`, or environment variables; keep them outside Git.

```bash
bash cmd/train.sh --config configs/pure.yaml --manifest /path/to/prepared/manifest.json --output /path/to/new-run
# Restore with the original output directory, config, source and manifest:
bash cmd/train.sh --config configs/pure.yaml --manifest /path/to/prepared/manifest.json --output /path/to/original-run --resume-latest
```

Training is manual and requires CUDA by default. No real experiments run as part of these checks.

## Interfaces / 接口

- `recommender/user_tower.py`: user ID + time/tab + masked mean history, normalized [B,64].
- `recommender/item_tower.py`: raw-ID/feature bridge to pinned Ziqi Item Tower; target [B,64], history [B,50,64], padding ID -1.
- `recommender/model.py`: normalized dot product / temperature=0.1, diagonal positive CE, duplicate-target negative masking.
- `recommender/training.py`: Adam, deterministic batch cursor, train and validation loss; full optimizer/scaler/RNG checkpoint.
- `recommender/prepare_pure.py`: source hashes, UTC splits, strict-prior history audit; feature fitting uses train-only item IDs.
- `export_test.py`: held-out requests and causal history audit; does not train.
- `colab_fetch_pure.py`: authorized Drive download with checksums. It never prints credentials.

Pinned publisher snapshot: `9359168f8d55b85f456c0b01d1cbbbfa3b1c37f6` (PR #36). Original files and hashes live in `recommender/vendor/data_pipeline_snapshot/provenance.json`. Retaining this snapshot permits reproducing the published model independently of later data-pipeline changes.

## Verified Pure run / 已完成的 Pure 实验

W&B: https://wandb.ai/SAR-Lab/kuairand-two-tower/runs/b2ed5728b9a2

- A100 40GB; 1 epoch; 124,780 train positives; 488 steps.
- 30,780 validation positives; in-batch validation loss 5.516606385154798.
- 7,583 normalized 64D item vectors exported.
- Training loss improved only slightly (~5.54 to ~5.50 on full batches). The final partial batch loss is not directly comparable to full batches.
- Recall@100/NDCG@100 have not been verified; 1K acceptance is not complete. Weighted like feedback is not implemented.

[Shared model/vector package](https://drive.google.com/file/d/1vEBOxLVamtStwH1yhVRoxrIcEKDOdRWg/view) includes the actual checkpoint, exact prepared manifest/tables, source ZIP, config, vectors and integrity inventory.
[Test package folder](https://drive.google.com/drive/u/1/folders/1d1QbCe4QBf4qIvEjyxF2r5jXLcj6SWna): `SAR-Pure-test-handoff-v1.zip`.
Raw data, checkpoints and vectors stay in Drive. Git contains code, tests and small documentation only.

## Test protocol / 测试协议

22,083 nonrandom long-view requests held out by time; test request IDs do not overlap train/validation. Same users/items may appear across splits. Rolling history includes only strictly earlier observed long views, including earlier test events, with no parameter updates. Full 7,583 candidate pool, prior-watched items remain eligible; historical candidate availability is not audited. 884 requests have users unseen in the training vocabulary; report them separately.

Single target Recall@100 is hit/miss; NDCG@100 is 1/log2(rank+1) for a hit, otherwise zero; average over requests. The evaluation implementation and candidate eligibility still need Ricky's review. Test must not be used for tuning.

See [docs/pure-test-handoff.md](docs/pure-test-handoff.md) for the complete handoff protocol. [docs/engineering.md](docs/engineering.md) and [docs/pure-handoff.md](docs/pure-handoff.md) describe the original ZIP layout; use the current notebook and commands above for repository setup.

The Week 1 notebook writes directly to the mounted shared Drive. The one-off historical ZIP copier has been removed; its delivered bundle remains intact in the historical release.

New Week 1 sessions use Python 3.11.15, available in uv 0.12.1’s managed downloads. Historical Python 3.11.16 checkpoints retain their original runtime binding; do not rewrite old run metadata to resume them with another patch version.

## Content feature retraining

Use `notebooks/week1_content_retraining.ipynb` for the publication-content experiment.
Both item towers consume seven categorical attributes and four numeric attributes (60 input values); the enhanced user history pools these content vectors. Current visibility, monthly aggregates and unversioned user profile statistics are excluded. Published attributes are assumed fixed, with the limitation recorded in the manifest.

`run.py prepare-content --config configs/pure-content-basic.yaml --manifest /path/to/base/manifest.json --source /path/to/source --output /path/to/new-content-dataset` preserves the audited requests and candidate identities while fitting metadata encodings on training items. The notebook reads the prepared shared Drive package, writes separate content runs/releases, and preserves previous ID-only results.
