"""Export aligned normalized candidate vectors with immutable shard manifests."""

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from .batch_source import DTYPES, file_sha256, read_manifest
from .checkpoint import load_checkpoint
from .model import build_model
from .training import binding, to_device


def export_items(config, manifest_path, checkpoint, output, start_id, end_id):
    """Encode candidate IDs in [start_id,end_id), sorted and without duplicates.

    Ziqi publishes a globally ID-sorted unique candidate Parquet file. No
    evaluation or candidate eligibility decisions are made by this exporter.
    """
    manifest, root, digest = read_manifest(manifest_path)
    device = torch.device(config["training"]["device"])
    saved = load_checkpoint(checkpoint, binding(config, digest, device))
    model = build_model(config, manifest).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Export directory must be empty")
    if start_id >= end_id:
        raise ValueError("Require a nonempty half-open ID interval")
    output.mkdir(parents=True, exist_ok=True)
    file = pq.ParquetFile(root / manifest["candidates"]["path"])
    shards, count, previous = [], 0, -1
    for table in file.iter_batches(
        batch_size=config["export"]["batch_size"], columns=["item_id", "category", "duration"]
    ):
        if any(table.column(name).null_count for name in ("item_id", "category", "duration")):
            raise ValueError("Candidate features require explicit missing sentinels")
        arrays = {
            name: table.column(name).to_numpy(zero_copy_only=False)
            for name in ("item_id", "category", "duration")
        }
        ids = arrays["item_id"]
        if ids.dtype.kind not in "iu" or np.any(ids < 0):
            raise ValueError("Candidate IDs must be nonnegative integers")
        if arrays["category"].dtype.kind not in "iu" or not np.isfinite(arrays["duration"]).all():
            raise ValueError("Candidate categories must be integer and durations finite")
        if len(ids) and (ids[0] <= previous or np.any(ids[1:] <= ids[:-1])):
            raise ValueError("Candidate IDs must be unique and globally sorted")
        if len(ids):
            previous = int(ids[-1])
        keep = (ids >= start_id) & (ids < end_id)
        if not keep.any():
            continue
        features = {
            name: torch.tensor(values[keep], dtype=DTYPES[name]) for name, values in arrays.items()
        }
        with torch.no_grad():
            vectors = model.encode_items(to_device(features, device)).cpu().numpy()
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5):
            raise ValueError("Candidate vectors must have unit norm")
        path = output / f"items-{len(shards):05d}.npz"
        np.savez(path, item_id=ids[keep], vectors=vectors)
        shards.append(
            {
                "path": path.name,
                "rows": len(vectors),
                "min_id": int(ids[keep][0]),
                "max_id": int(ids[keep][-1]),
                "sha256": file_sha256(path),
            }
        )
        count += len(vectors)
    release = {
        "schema_version": 1,
        "binding": saved["binding"],
        "dimension": 64,
        "checkpoint_sha256": file_sha256(checkpoint),
        "run_id": saved["run_id"],
        "training_complete": saved["epoch"] == config["training"]["epochs"],
        "range": [start_id, end_id],
        "rows": count,
        "shards": shards,
    }
    (output / "manifest.json").write_text(json.dumps(release, indent=2))
    return release
