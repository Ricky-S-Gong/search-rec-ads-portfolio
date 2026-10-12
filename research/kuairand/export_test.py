"""Publish held-out Pure evaluation requests without training or fitting features."""

import argparse
import json
import shutil
from collections import deque
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from kuairand.batch_source import file_sha256, read_manifest


def export_test(source, training_manifest, output):
    """Audit request disjointness and causal histories; export all test exposures and positives.

    Evaluation uses rolling observed history, including earlier test positives.
    No test event updates model parameters or preprocessing statistics.
    """
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError("Use a new output directory")
    manifest, root, manifest_hash = read_manifest(training_manifest)
    for name, digest in manifest["source_sha256"].items():
        if file_sha256(source / name) != digest:
            raise ValueError(f"Source snapshot changed: {name}")
    events = pq.read_table(source / "events.parquet")
    values = {n: events[n].to_numpy(zero_copy_only=False) for n in events.column_names}
    split, times = values["split"], values["request_time_ms"]
    test = split == "test"
    requests = values["request_id"]
    if len(set(requests)) != len(requests):
        raise ValueError("Source request IDs are not globally unique")
    if set(requests[test]) & set(requests[np.isin(split, ["train", "validation"])]):
        raise ValueError("Test requests overlap train or validation")
    if times[test].min() <= times[split == "validation"].max():
        raise ValueError("Test time window overlaps validation")
    positive = (values["long_view"] == 1) & (values["is_random"] == 0)
    histories = pq.read_table(source / "user_histories.parquet")
    history_map = dict(zip(histories["request_id"].to_pylist(),
                           histories["history_item_ids"].to_pylist()))
    positions = np.flatnonzero(positive)
    positions = positions[np.argsort(times[positions], kind="stable")]
    recent, pending, last_time = {}, [], None
    checked = 0
    for p in positions:
        now = int(times[p])
        if now != last_time:
            for user, item in pending:
                recent.setdefault(user, deque(maxlen=manifest["history_length"])).append(item)
            pending, last_time = [], now
        user, item = int(values["user_id"][p]), int(values["item_id"][p])
        if test[p]:
            if history_map.get(requests[p]) != list(recent.get(user, [])):
                raise ValueError("Test history includes future/current events or differs from source")
            checked += 1
        pending.append((user, item))
    keep = test & positive
    selected = events.filter(pa.array(keep))
    selected = selected.rename_columns([
        "time_ms" if name == "request_time_ms" else name for name in selected.column_names
    ])
    selected = selected.append_column("history_item_ids", pa.array(
        [history_map[r] for r in requests[keep]], type=pa.list_(pa.int64())))
    candidate_ids = np.array(manifest["item_ids"])
    if not np.isin(values["item_id"][keep], candidate_ids).all():
        raise ValueError("Test targets missing from candidate catalog")
    output.mkdir(parents=True)
    pq.write_table(events.filter(pa.array(test)), output / "test_events.parquet")
    pq.write_table(selected, output / "test_requests.parquet")
    shutil.copyfile(source / "item_features.parquet", output / "item_features.parquet")
    shutil.copyfile(root / manifest["candidates"]["path"], output / "candidates.parquet")
    (output / "frozen_features.json").write_text(json.dumps({
        "user_ids": manifest["user_ids"], "item_ids": manifest["item_ids"],
        "item_encoding": manifest["item_encoding"], "history_length": manifest["history_length"],
    }) + "\n")
    metadata = {
        "dataset": "KuaiRand-Pure", "local_preparation_manifest_sha256": manifest_hash,
        "source_sha256": manifest["source_sha256"],
        "test_events": int(test.sum()), "test_positive_requests": int(keep.sum()),
        "audited_test_histories": checked, "candidate_items": len(candidate_ids),
        "time_range_ms": [int(times[test].min()), int(times[test].max()) + 1],
        "request_overlap_train_validation": 0,
        "user_item_disjoint": False,
        "history_protocol": "rolling strictly-prior nonrandom long_view; max 50; ties excluded",
        "unknown_user_requests": int((~np.isin(values["user_id"][keep], manifest["user_ids"])).sum()),
        "target_protocol": "one ground-truth item per positive request",
        "candidate_protocol": "full 7583-item static catalog; repeated prior items remain eligible",
        "availability_caveat": "catalog-wide evaluation; per-request historical availability not audited",
        "feature_protocol": "reuse frozen item_encoding/user_ids; do not fit on test",
        "files": {},
    }
    for file in sorted(output.iterdir()):
        metadata["files"][file.name] = {"sha256": file_sha256(file), "bytes": file.stat().st_size}
    (output / "evaluation_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_test(args.source, args.training_manifest, args.output), indent=2))
