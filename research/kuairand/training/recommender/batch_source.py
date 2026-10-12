"""Consume published positive-only Parquet shards; no raw-log cleaning here."""

import hashlib
import json
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

REQUIRED = ("user_id", "item_id", "time_ms", "tab", "category", "duration")
DTYPES = {name: torch.long for name in REQUIRED if name != "duration"}
DTYPES["duration"] = torch.float32


def file_sha256(path):
    """Hash a file sequentially with bounded memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path, *, require_time_protocol=True):
    """Validate publication identity, checksums, training vocabulary and row counts."""
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1 or not manifest.get("positive_only"):
        raise ValueError("Require schema_version=1 and published positive-only training data")
    if manifest.get("label") != "long_view" or manifest.get("duration_unit") != "seconds":
        raise ValueError("Require long_view label and duration in seconds")
    if not manifest.get("dataset_version") or not manifest.get("feature_version"):
        raise ValueError("Missing data/feature versions")
    if require_time_protocol and manifest.get("time_protocol_verified") is not True:
        raise ValueError("Publisher must verify timestamp/date/timezone and feedback protocol")
    ranges = manifest.get("train_time_ranges_ms")
    if (
        not isinstance(ranges, list)
        or not ranges
        or any(
            not isinstance(r, list)
            or len(r) != 2
            or any(type(x) is not int for x in r)
            or r[0] >= r[1]
            for r in ranges
        )
    ):
        raise ValueError("Explicit half-open training timestamp ranges are required")
    ids = manifest.get("user_ids")
    if not isinstance(ids, list) or any(type(x) is not int or x < 0 for x in ids):
        raise ValueError("user_ids must be a training-only nonnegative integer vocabulary")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate vocabulary IDs")
    for shard in (
        manifest["train_shards"] + [manifest["candidates"]] + manifest.get("extra_files", [])
    ):
        resolved = path.parent / shard["path"]
        if file_sha256(resolved) != shard["sha256"]:
            raise ValueError(f"Checksum mismatch: {resolved}")
        if pq.ParquetFile(resolved).metadata.num_rows != shard["rows"]:
            raise ValueError(f"Row count mismatch: {resolved}")
    return manifest, path.parent, file_sha256(path)


class ParquetBatches:
    """Deterministic global row shuffle with a bounded row-group cache.

    The permutation uses ~8 bytes/event; feature data are loaded by row group.
    Row-group size is chosen by the publisher and determines cache byte usage.
    This correctness-first reader is replaceable by Ziqi's optimized reader.
    """

    def __init__(
        self, manifest, root, batch_size, seed, cache_groups, *, allow_unknown_users=False
    ):
        self.allow_unknown_users = allow_unknown_users
        self.batch_size, self.seed, self.cache_groups = batch_size, seed, cache_groups
        self.history_length = manifest.get("history_length", 0)
        self.time_ranges = manifest["train_time_ranges_ms"]
        self.user_vocabulary = np.array(sorted(manifest["user_ids"]), dtype=np.int64)
        self.files, self.groups, self.ends = [], [], []
        self.cache = OrderedDict()
        total = 0
        for shard in manifest["train_shards"]:
            file = pq.ParquetFile(root / shard["path"])
            if not set(REQUIRED + ("long_view", "is_random")).issubset(file.schema.names):
                raise ValueError(f"Missing standardized batch fields: {shard['path']}")
            self.files.append(file)
            for group in range(file.num_row_groups):
                count = file.metadata.row_group(group).num_rows
                self.groups.append((len(self.files) - 1, group, total))
                total += count
                self.ends.append(total)
        self.rows = total
        self.num_batches = (total + batch_size - 1) // batch_size
        if not total:
            raise ValueError("No training positives")

    def _group(self, index):
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        file_id, group, _ = self.groups[index]
        table = self.files[file_id].read_row_group(
            group,
            columns=list(REQUIRED)
            + ["long_view", "is_random"]
            + (["history_item_ids"] if self.history_length else []),
        )
        if table["long_view"].null_count or table["is_random"].null_count:
            raise ValueError("Labels and random flags cannot be null")
        if not np.all(table["long_view"].to_numpy() == 1) or not np.all(
            table["is_random"].to_numpy() == 0
        ):
            raise ValueError("Main training requires long_view=1 and is_random=0")
        arrays = {name: table[name].to_numpy(zero_copy_only=False) for name in REQUIRED}
        if any(table[name].null_count for name in REQUIRED):
            raise ValueError("Use explicit missing sentinels rather than null batch fields")
        if not np.isfinite(arrays["duration"]).all():
            raise ValueError("Use a finite duration missing sentinel")
        if np.any(arrays["user_id"] < 0) or np.any(arrays["item_id"] < 0):
            raise ValueError("Published training IDs must be nonnegative")
        if (
            not self.allow_unknown_users
            and not np.isin(arrays["user_id"], self.user_vocabulary).all()
        ):
            raise ValueError("Training vocabulary does not cover published users")
        timestamps = arrays["time_ms"]
        eligible = np.zeros(len(timestamps), dtype=bool)
        for start, end in self.time_ranges:
            eligible |= (timestamps >= start) & (timestamps < end)
        if not eligible.all():
            raise ValueError("Training shard contains out-of-window events")
        for name in REQUIRED[:-1]:
            if arrays[name].dtype.kind not in "iu":
                raise ValueError(f"{name} must be an integer field")
        if self.history_length:
            arrays["history_item_ids"] = table["history_item_ids"].to_pylist()
            if any(
                len(h) > self.history_length or any(type(i) is not int or i < 0 for i in h)
                for h in arrays["history_item_ids"]
            ):
                raise ValueError("Malformed history IDs")
        self.cache[index] = arrays
        while len(self.cache) > self.cache_groups:
            self.cache.popitem(last=False)
        return arrays

    def batches(self, epoch, start=0):
        """Yield (next_cursor, tensors) in identical order after checkpoint restore."""
        order = np.random.default_rng(self.seed + epoch).permutation(self.rows)
        for cursor in range(start, self.num_batches):
            rows = order[cursor * self.batch_size : (cursor + 1) * self.batch_size]
            result = {
                name: np.empty(len(rows), dtype=np.float32 if name == "duration" else np.int64)
                for name in REQUIRED
            }
            history_ids = np.full((len(rows), self.history_length), -1, dtype=np.int64)
            # Group lookups avoid re-reading a row group per event.
            locations = {}
            for position, row in enumerate(rows):
                group = bisect_right(self.ends, int(row))
                locations.setdefault(group, []).append((position, int(row) - self.groups[group][2]))
            for group, positions in locations.items():
                arrays = self._group(group)
                output, local = zip(*positions)
                for name in REQUIRED:
                    result[name][list(output)] = arrays[name][list(local)]
                if self.history_length:
                    for out, row in positions:
                        history = arrays["history_item_ids"][row]
                        if history:
                            history_ids[out, -len(history) :] = history
            batch = {
                name: torch.tensor(values, dtype=DTYPES[name]) for name, values in result.items()
            }
            batch["history_mask"] = torch.tensor(history_ids >= 0)
            if self.history_length:
                batch["history_item_ids"] = torch.tensor(history_ids)
            yield cursor + 1, batch
