"""Diagnostic inference using the historical bundle's original frozen Python source.

Run with PYTHONPATH pointing to both the unpacked frozen training root and the
current research/kuairand root. This never resumes training or changes binding.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from evaluation.retrieval import exact_topk, ranking_metrics


def evaluate(bundle, metadata, output):
    from kuairand.batch_source import file_sha256, read_manifest
    from kuairand.checkpoint import load_checkpoint
    from kuairand.model import build_model
    from kuairand.training import code_sha, config_sha

    bundle, output = Path(bundle), Path(output)
    if output.exists():
        raise ValueError("Use a new diagnostic output")
    for name, entry in json.loads((bundle / "checksums.json").read_text()).items():
        if file_sha256(bundle / name) != entry["sha256"]:
            raise ValueError("Historical bundle checksum mismatch: " + name)
    config = json.loads((bundle / "model/config.json").read_text())
    publication = json.loads((bundle / "model/run_manifest.json").read_text())
    manifest, root, digest = read_manifest(bundle / "prepared/manifest.json")
    contract = publication["binding"]
    if (
        digest != contract["manifest_sha256"]
        or code_sha(config) != contract["code_sha256"]
        or config_sha(config) != contract["config_sha256"]
    ):
        raise ValueError("Historical code/config/data mismatch")
    if torch.__version__.split("+")[0] != contract["torch"].split("+")[0]:
        raise ValueError("Inference requires the matching PyTorch base version")
    checkpoint = bundle / "model" / Path(publication["checkpoint"]).name
    saved = load_checkpoint(checkpoint, contract, device="cpu")
    model = build_model(config, manifest).eval()
    model.load_state_dict(saved["model"])
    release = json.loads((bundle / "vectors/manifest.json").read_text())
    if release["binding"] != contract or release["checkpoint_sha256"] != file_sha256(checkpoint):
        raise ValueError("Historical vector binding mismatch")
    ids, blocks = [], []
    for shard in release["shards"]:
        with np.load(bundle / "vectors" / shard["path"], allow_pickle=False) as part:
            ids.extend(part["item_id"].tolist())
            blocks.append(part["vectors"])
    ids = np.array(ids)
    vectors = np.concatenate(blocks)
    if not np.array_equal(ids, manifest["item_ids"]) or not np.allclose(
        np.linalg.norm(vectors, axis=1), 1, atol=1e-5
    ):
        raise ValueError("Historical catalog mismatch")
    request = pq.read_table(root / "validation.parquet")
    table = pq.read_table(metadata).sort_by([("item_id", "ascending")])
    if not np.array_equal(table["item_id"].to_numpy(), ids):
        raise ValueError("Metadata candidate mismatch")
    after = (
        np.array(table["upload_dt"].to_pylist(), dtype="datetime64[D]")
        .astype("datetime64[ms]")
        .astype(np.int64)
        + 86400000
    )
    histories = request["history_item_ids"].to_pylist()
    predictions = []
    torch.set_num_threads(1)
    for start in range(0, len(request), 256):
        stop = min(start + 256, len(request))
        h = np.full((stop - start, 50), -1, dtype=np.int64)
        for row, history in enumerate(histories[start:stop]):
            if history:
                h[row, -len(history) :] = history
        features = {
            name: torch.tensor(
                request[name].slice(start, stop - start).to_numpy(), dtype=torch.long
            )
            for name in ("user_id", "time_ms", "tab")
        }
        with torch.no_grad():
            hv = model.item_tower.encode_items({"item_id": torch.tensor(h)})
            users = model.encode_users(features, hv, torch.tensor(h >= 0)).numpy()
        pred, _ = exact_topk(
            users, vectors, ids, request["time_ms"].slice(start, stop - start).to_numpy(), after
        )
        predictions.append(pred)
    targets = request["item_id"].to_numpy()
    covered = after[np.searchsorted(ids, targets)] <= request["time_ms"].to_numpy()
    result = {
        "scope": "historical diagnostic only; old request-prior histories and snapshot features; not the corrected fair comparison",
        "run_id": publication["run_id"],
        "binding": contract,
        "checkpoint_sha256": file_sha256(checkpoint),
        "frozen_source_zip_sha256": file_sha256(bundle / "source/SAR-colab-integrated-v2.zip"),
        "validation_sha256": file_sha256(root / "validation.parquet"),
        "inference_device": "cpu",
        "metrics": ranking_metrics(np.concatenate(predictions), targets, covered),
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.bundle, args.metadata, args.output), indent=2))
