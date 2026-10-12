"""Prepare the conservative Week 1 protocol from hash-pinned official raw logs."""

import hashlib
import heapq
import json
import tarfile
from collections import Counter, deque
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as csv
import pyarrow.parquet as pq

from recommender.batch_source import file_sha256
from recommender.item_tower import fit_item_encoding

DAY = 86400000
OFFSET = 0
CUTOFFS = np.array([1649980800000, 1651190400000, 1651622400000])
RAW_MD5 = "0820331067a3784d9691136f772b35a7"


def completed_histories(users, items, times, available, limit=50):
    """Publish feedback strictly after completion; tie requests cannot see each other.

    Inputs are positive, nonrandom events ordered by request time. History order
    follows original request time, even when plays finish out of order.
    """
    recent, pending, result = {}, [], []
    if limit < 1 or np.any(np.diff(times) < 0) or np.any(available < times):
        raise ValueError("Invalid history timeline")
    for ordinal, (user, item, now, done) in enumerate(zip(users, items, times, available)):
        while pending and pending[0][0] < now:
            _, previous, who, video = heapq.heappop(pending)
            history = recent.setdefault(int(who), [])
            history.append((previous, int(video)))
            history.sort()
            del history[:-limit]
        result.append([video for _, video in recent.get(int(user), [])])
        heapq.heappush(pending, (int(done), ordinal, int(user), int(item)))
    return result


def profile(values):
    """Bounded aggregate field description; values come from warmup/train/validation."""
    values = list(values)
    present = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    result = {
        "count": len(values),
        "missing": len(values) - len(present),
        "cardinality": len({str(v) for v in present}),
        "examples": list(dict.fromkeys(map(str, present)))[:3],
    }
    numeric = [v for v in present if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if numeric and len(numeric) == len(present):
        result["min"] = float(min(numeric))
        result["max"] = float(max(numeric))
        result["quantiles"] = np.quantile(numeric, [0, 0.25, 0.5, 0.75, 0.95, 1]).tolist()
    return result


def prepare_week1(source, archive, output, config):
    """Rebuild training and validation with completion-time histories; no test evaluation."""
    source, archive, output = Path(source), Path(archive), Path(output)
    if output.exists():
        raise ValueError("Use a new immutable prepared directory")
    hashes = {
        name: file_sha256(source / name)
        for name in config["data_adapter"]["verified_source_sha256"]
    }
    if hashes != config["data_adapter"]["verified_source_sha256"]:
        raise ValueError("Published Drive snapshot differs")
    digest = hashlib.md5(archive.read_bytes()).hexdigest()
    if digest != RAW_MD5:
        raise ValueError("Official Pure archive MD5 mismatch")
    published = pq.read_table(
        source / "events.parquet",
        filters=[("request_time_ms", "<", int(CUTOFFS[2])), ("is_random", "=", False)],
    )
    tables, duplicate_rows = [], 0
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if "log_standard_" not in member.name or not member.name.endswith(".csv"):
                continue
            raw = csv.read_csv(tar.extractfile(member))
            raw = raw.append_column("row_number", pa.array(np.arange(len(raw))))
            # Filter before examining feedback: the test period is not used.
            raw = raw.filter(
                pc.and_(pc.less(raw["time_ms"], int(CUTOFFS[2])), pc.equal(raw["is_rand"], 0))
            )
            matrices = np.column_stack(
                [raw[n].to_numpy() for n in raw.column_names if n != "row_number"]
            )
            _, first = np.unique(matrices, axis=0, return_index=True)
            duplicate_rows += len(raw) - len(first)
            raw = raw.take(pa.array(np.sort(first)))
            requests = [
                Path(member.name).name + ":" + str(i) for i in raw["row_number"].to_pylist()
            ]
            tables.append(
                pa.table(
                    {
                        "request_id": requests,
                        "user_id": raw["user_id"],
                        "item_id": raw["video_id"],
                        "time_ms": raw["time_ms"],
                        "tab": raw["tab"],
                        "long_view": raw["long_view"],
                        "is_random": raw["is_rand"],
                        "play_time_ms": raw["play_time_ms"],
                        "duration_ms": raw["duration_ms"],
                        "date": raw["date"],
                    }
                )
            )
    events = pa.concat_tables(tables).sort_by(
        [("time_ms", "ascending"), ("request_id", "ascending")]
    )
    data = {n: events[n].to_numpy(zero_copy_only=False) for n in events.column_names}
    if len(set(data["request_id"])) != len(events):
        raise ValueError("Duplicate request identity")
    # Match raw identity and primary fields to the existing Drive publication.
    old = published.sort_by([("request_id", "ascending")])
    new = events.sort_by([("request_id", "ascending")])
    # Exact raw duplicates can be removed only by full-row equality, with counts retained.
    old = old.filter(pc.is_in(old["request_id"], value_set=new["request_id"].combine_chunks()))
    for raw_name, old_name in [
        ("user_id", "user_id"),
        ("item_id", "item_id"),
        ("time_ms", "request_time_ms"),
        ("tab", "tab"),
        ("long_view", "long_view"),
    ]:
        if not np.array_equal(new[raw_name].to_numpy(), old[old_name].to_numpy()):
            raise ValueError("Raw/Drive event identity mismatch: " + raw_name)
    times = data["time_ms"]
    date_mismatches = {}
    for offset_hours in (0, 8):
        calendar = (
            (times + offset_hours * 3600000)
            .astype("datetime64[ms]")
            .astype("datetime64[D]")
            .astype(str)
        )
        dates_as_int = np.char.replace(calendar, "-", "").astype(np.int64)
        date_mismatches[str(offset_hours)] = int((dates_as_int != data["date"]).sum())
    # date is not a reliable fixed-offset calendar: freeze explicit Unix UTC windows.
    if np.any(data["play_time_ms"] < 0) or np.any(data["duration_ms"] < 0):
        raise ValueError("Invalid feedback duration")
    split = np.where(
        times < CUTOFFS[0], "warmup", np.where(times < CUTOFFS[1], "train", "validation")
    )
    positive = data["long_view"] == 1
    pos = np.flatnonzero(positive)
    available = times[pos] + data["play_time_ms"][pos]
    histories = completed_histories(
        data["user_id"][pos], data["item_id"][pos], times[pos], available
    )
    # Diagnose how often request-prior feedback would still be unavailable.
    prior, pending = {}, []
    changed = 0
    for index, position in enumerate(pos):
        now, who = int(times[position]), int(data["user_id"][position])
        while pending and pending[0][0] < now:
            _, order, user, item = heapq.heappop(pending)
            prior.setdefault(user, deque(maxlen=50)).append(item)
        if list(prior.get(who, [])) != histories[index]:
            changed += 1
        heapq.heappush(pending, (now, index, who, int(data["item_id"][position])))
    metadata = pq.read_table(source / "item_features.parquet").sort_by([("item_id", "ascending")])
    ids = metadata["item_id"].to_numpy()
    if len(np.unique(ids)) != len(ids) or not np.isin(data["item_id"], ids).all():
        raise ValueError("Candidate IDs are incomplete or duplicated")
    dates = np.array(metadata["upload_dt"].to_pylist(), dtype="datetime64[D]")
    eligible_after = dates.astype("datetime64[ms]").astype(np.int64) + DAY - OFFSET
    eligible_after[np.isnat(dates)] = np.iinfo(np.int64).max
    fit_ids = data["item_id"][split == "train"]
    # Metadata lacks as-of versions: use only item ID in the formal comparison.
    encoding = fit_item_encoding(metadata, fit_ids, ("item_id",), ())
    output.mkdir(parents=True)

    def entry(name, table):
        pq.write_table(table, output / name, row_group_size=32768)
        return {"path": name, "rows": len(table), "sha256": file_sha256(output / name)}

    shards = {}
    for name in ("train", "validation"):
        indices = np.flatnonzero(split[pos] == name)
        selected = pos[indices]
        table = pa.table(
            {
                "request_id": data["request_id"][selected],
                "user_id": data["user_id"][selected],
                "item_id": data["item_id"][selected],
                "time_ms": times[selected],
                "tab": data["tab"][selected],
                "category": np.zeros(len(selected), dtype=np.int64),
                "duration": data["duration_ms"][selected] / 1000,
                "long_view": np.ones(len(selected), dtype=np.int64),
                "is_random": np.zeros(len(selected), dtype=np.int64),
                "history_item_ids": pa.array(
                    [histories[i] for i in indices], type=pa.list_(pa.int64())
                ),
            }
        )
        shards[name] = entry(name + ".parquet", table)
    candidates = entry(
        "candidates.parquet",
        pa.table(
            {
                "item_id": ids,
                "category": np.zeros(len(ids), dtype=np.int64),
                "duration": np.zeros(len(ids)),
                "eligible_after_ms": eligible_after,
            }
        ),
    )
    counts = Counter(map(int, data["item_id"][(split == "train") & positive]))
    lengths = np.array([len(h) for h in histories])
    summaries = {}
    for name in ("warmup", "train", "validation"):
        mask = split == name
        summaries[name] = {
            "rows": int(mask.sum()),
            "positives": int((mask & positive).sum()),
            "positive_rate": float(positive[mask].mean()),
            "users": len(np.unique(data["user_id"][mask])),
            "items": len(np.unique(data["item_id"][mask])),
        }
    fields = {
        n: profile(events[n].to_pylist())
        for n in (
            "user_id",
            "item_id",
            "tab",
            "time_ms",
            "long_view",
            "play_time_ms",
            "duration_ms",
        )
    }
    fields.update({n: profile(metadata[n].to_pylist()) for n in metadata.column_names})
    for n in ("user_id", "tab", "time_ms"):
        fields["user_" + n] = profile(events[n].to_pylist())
    validation = pq.read_table(output / "validation.parquet")
    vp = np.searchsorted(ids, validation["item_id"].to_numpy())
    covered = eligible_after[vp] <= validation["time_ms"].to_numpy()
    audit = {
        "protocol": "completion-time-v1",
        "status": "prepared; GPU experiments pending",
        "source_sha256": hashes,
        "raw_archive_sha256": file_sha256(archive),
        "raw_archive_md5": digest,
        "calendar_timezone": "UTC (explicit Unix timestamp windows, independent of source date)",
        "recorded_date_mismatches": date_mismatches,
        "split_cutoffs_ms": CUTOFFS.tolist(),
        "splits": summaries,
        "exact_duplicates_removed": duplicate_rows,
        "request_prior_histories_changed": changed,
        "feedback_rule": "available_ms = time_ms + play_time_ms; available_ms < request time; observed duration proxy, not an independently logged completion timestamp",
        "candidate_count": len(ids),
        "missing_upload_dates": int(np.isnat(dates).sum()),
        "validation_target_coverage": float(covered.mean()),
        "history_length_histogram": np.bincount(lengths, minlength=51).tolist(),
        "empty_history_rate": float((lengths == 0).mean()),
        "train_top100_positive_share": sum(sorted(counts.values(), reverse=True)[:100])
        / sum(counts.values()),
        "train_item_positive_count_quantiles": np.quantile(
            list(counts.values()), [0, 0.25, 0.5, 0.75, 0.95, 1]
        ).tolist(),
        "train_user_event_count_quantiles": np.quantile(
            list(Counter(map(int, data["user_id"][split == "train"])).values()),
            [0, 0.25, 0.5, 0.75, 0.95, 1],
        ).tolist(),
        "features": fields,
        "model_categorical_features": ["item_id"],
        "model_numeric_features": [],
        "excluded_snapshot_features": [n for n in metadata.column_names if n != "item_id"],
        "limitations": [
            "Recorded date disagrees with fixed UTC and UTC+08 calendars; date is not used for splitting.",
            "No logged feedback-completion timestamp; full observed playback used as conservative proxy.",
            "Metadata is an unversioned snapshot; content features excluded, upload date used only as a catalog eligibility proxy.",
            "Pure histories omit videos outside the candidate pool; standard-exposure validation is not an unbiased online estimate.",
        ],
    }
    manifest = {
        "schema_version": 1,
        "dataset_version": "pure-completion-v1-" + audit["raw_archive_sha256"][:12],
        "feature_version": "train-id-only-v1",
        "positive_only": True,
        "label": "long_view",
        "duration_unit": "seconds",
        "time_protocol_verified": True,
        "user_ids": sorted(np.unique(data["user_id"][split == "train"]).tolist()),
        "item_ids": ids.tolist(),
        "history_length": 50,
        "history_policy": audit["feedback_rule"],
        "train_time_ranges_ms": [[int(CUTOFFS[0]), int(CUTOFFS[1])]],
        "validation_time_ranges_ms": [[int(CUTOFFS[1]), int(CUTOFFS[2])]],
        "train_shards": [shards["train"]],
        "extra_files": [shards["validation"]],
        "candidates": candidates,
        "item_encoding": encoding,
        "source_sha256": hashes,
        "protocol": {
            "calendar_timezone": "UTC",
            "repeat_items_allowed": True,
            "eligibility": "day after upload, UTC",
            "test_used": False,
            "raw_archive_sha256": audit["raw_archive_sha256"],
        },
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return audit
