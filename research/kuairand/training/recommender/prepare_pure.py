"""Consumer adapter for Ziqi's published Pure tables; never launches training."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from .batch_source import ParquetBatches, file_sha256, read_manifest
from .model import build_model
from .user_tower import UserTower

FILES = ("events.parquet", "user_histories.parquet", "item_features.parquet", "manifest.json")
EVENT_FIELDS = (
    "request_id",
    "user_id",
    "item_id",
    "request_time_ms",
    "tab",
    "long_view",
    "is_random",
    "split",
)


def require(condition, message):
    """Reject incompatible input instead of silently dropping malformed rows."""
    if not condition:
        raise ValueError(message)


def prepare_pure(source, output, config):
    """Audit all source rows and create a private, reproducible training manifest.

    Only Pure fits this in-memory adapter. Published split assignments are preserved.
    Histories are inventoried but unused by the Week 1 context-only User Tower.
    Timestamp/feedback provenance remains unverified unless explicitly published.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    settings = config["data_adapter"]
    require(not output.exists(), "Use a new output directory; source data remain untouched")
    require(not output.is_relative_to(source), "Keep consumer outputs outside the source folder")
    for name in FILES:
        require((source / name).is_file(), f"Missing {source / name}")
    declaration = json.loads((source / "manifest.json").read_text())
    require(declaration.get("dataset") == "KuaiRand-Pure", "This adapter only supports Pure")
    event_file = pq.ParquetFile(source / "events.parquet")
    require(event_file.metadata.num_rows <= settings["max_source_rows"], "Pure row limit exceeded")
    events = event_file.read()
    require(set(EVENT_FIELDS).issubset(events.column_names), "Missing published event fields")
    require(len(events) == declaration["rows"], "Source manifest row count mismatch")
    require(all(not events[n].null_count for n in EVENT_FIELDS), "Null event fields")
    require(
        all(
            pa.types.is_integer(events[n].type)
            for n in ("user_id", "item_id", "request_time_ms", "tab")
        ),
        "Integer event fields required",
    )
    values = {n: events[n].to_numpy(zero_copy_only=False) for n in EVENT_FIELDS}
    split, times = values["split"], values["request_time_ms"]
    counts = {str(k): int(v) for k, v in Counter(split).items()}
    require(counts == declaration["split_counts"], "Published split counts mismatch")
    require(set(counts) == {"warmup", "train", "validation", "test", "random"}, "Unexpected splits")
    require(np.isin(values["long_view"], [0, 1]).all(), "Invalid long_view")
    require(np.isin(values["is_random"], [0, 1]).all(), "Invalid random flag")
    require(
        ((split == "random") == values["is_random"].astype(bool)).all(), "Random split mismatch"
    )
    require((values["user_id"] >= 0).all() and (values["item_id"] >= 0).all(), "Negative IDs")
    require(
        ((values["tab"] >= 0) & (values["tab"] < config["model"]["num_tabs"])).all(), "Invalid tab"
    )
    ranges = {s: [int(times[split == s].min()), int(times[split == s].max()) + 1] for s in counts}
    for first, second in zip(("warmup", "train", "validation"), ("train", "validation", "test")):
        require(ranges[first][1] <= ranges[second][0], "Temporal split overlap")
    candidates = pq.read_table(source / "item_features.parquet")
    require(
        {"item_id", "tag", "video_duration"}.issubset(candidates.column_names),
        "Missing item fields",
    )
    ids = candidates["item_id"].to_numpy()
    require(
        pa.types.is_integer(candidates["item_id"].type) and not candidates["item_id"].null_count,
        "Integer item IDs required",
    )
    require((ids >= 0).all() and len(np.unique(ids)) == len(ids), "Duplicate/negative item IDs")
    order = np.argsort(ids)
    ids = ids[order]
    require(np.isin(values["item_id"], ids).all(), "Events contain items missing features")
    require(settings["category_policy"] == "first_tag", "Unsupported category policy")
    categories = []
    for tag in candidates["tag"].to_pylist():
        token = str(tag).split(",")[0].strip() if tag is not None else ""
        require(not token or token.lstrip("-").isdigit(), "Malformed category tag")
        categories.append(int(token) if token else settings["category_missing"])
    categories = np.array(categories, dtype=np.int64)[order]
    raw_duration = candidates["video_duration"].to_numpy(zero_copy_only=False)[order]
    require(
        (np.isnan(raw_duration) | (np.isfinite(raw_duration) & (raw_duration >= 0))).all(),
        "Invalid video duration",
    )
    duration = np.where(
        np.isnan(raw_duration),
        settings["duration_missing_seconds"],
        raw_duration / settings["duration_ms_per_second"],
    ).astype(np.float32)
    histories = pq.read_table(source / "user_histories.parquet")
    require(
        {"request_id", "user_id", "history_item_ids"}.issubset(histories.column_names),
        "Missing history fields",
    )
    require(all(not histories[n].null_count for n in histories.column_names), "Null history fields")
    require(
        pa.types.is_list(histories["history_item_ids"].type)
        and pa.types.is_integer(histories["history_item_ids"].type.value_type),
        "History IDs must be integer lists",
    )
    history_ids = histories["history_item_ids"].combine_chunks().flatten().to_numpy()
    require(np.isin(history_ids, ids).all(), "Unknown history items")
    event_pairs = set(zip(values["request_id"], values["user_id"]))
    pairs = list(zip(histories["request_id"].to_pylist(), histories["user_id"].to_pylist()))
    require(len(set(pairs)) == len(pairs), "Duplicate request/user history rows")
    require(all(p in event_pairs for p in pairs), "History request/user missing from events")
    positive = (split == "train") & (values["long_view"] == 1) & (values["is_random"] == 0)
    require(positive.any(), "No training positives")
    item_positions = np.searchsorted(ids, values["item_id"][positive])
    train = pa.table(
        {
            "request_id": values["request_id"][positive],
            "user_id": values["user_id"][positive],
            "item_id": values["item_id"][positive],
            "time_ms": times[positive],
            "tab": values["tab"][positive],
            "category": categories[item_positions],
            "duration": duration[item_positions],
            "long_view": np.ones(int(positive.sum()), dtype=np.int64),
            "is_random": np.zeros(int(positive.sum()), dtype=np.int64),
        }
    )
    candidate_table = pa.table({"item_id": ids, "category": categories, "duration": duration})
    hashes = {name: file_sha256(source / name) for name in FILES}
    manifest = {
        "schema_version": 1,
        "dataset_version": "ziqi-pure-v1-" + hashes["events.parquet"][:12],
        "feature_version": "first-tag-duration-seconds-v1-" + hashes["item_features.parquet"][:12],
        "positive_only": True,
        "label": "long_view",
        "duration_unit": "seconds",
        "time_protocol_verified": declaration.get("time_protocol_verified") is True,
        "train_time_ranges_ms": [ranges["train"]],
        "user_ids": sorted(np.unique(values["user_id"][split == "train"]).tolist()),
        "item_ids": ids.tolist(),
        "source_sha256": hashes,
        "feature_policy": settings,
        "source_directory": str(source),
        "history_policy": "Week 1 ignores histories; causal availability not verified",
    }
    if settings.get("integration"):
        from .item_tower import fit_item_encoding

        history_map = dict(
            zip(histories["request_id"].to_pylist(), histories["history_item_ids"].to_pylist())
        )
        nonrandom_positive = (values["long_view"] == 1) & (values["is_random"] == 0)
        positions = np.flatnonzero(nonrandom_positive)
        positions = positions[np.argsort(times[positions], kind="stable")]
        from collections import deque

        recent, pending = {}, []
        last_time = None
        for position in positions:
            now = int(times[position])
            if now != last_time:
                for user, item in pending:
                    recent.setdefault(user, deque(maxlen=settings["history_length"])).append(item)
                pending = []
                last_time = now
            request, user = values["request_id"][position], int(values["user_id"][position])
            require(request in history_map, "Missing positive request history")
            require(
                history_map[request] == list(recent.get(user, [])),
                "History is not strictly prior or has wrong order",
            )
            pending.append((user, int(values["item_id"][position])))
        require(
            len(history_map) == len(positions), "History coverage differs from nonrandom positives"
        )
        selected = values["request_id"][positive]
        train = train.append_column(
            "history_item_ids",
            pa.array([history_map[r] for r in selected], type=pa.list_(pa.int64())),
        )
        validation_keep = (split == "validation") & nonrandom_positive
        validation_ids = values["item_id"][validation_keep]
        vp = np.searchsorted(ids, validation_ids)
        validation = pa.table(
            {
                "user_id": values["user_id"][validation_keep],
                "item_id": validation_ids,
                "time_ms": times[validation_keep],
                "tab": values["tab"][validation_keep],
                "category": categories[vp],
                "duration": duration[vp],
                "long_view": np.ones(int(validation_keep.sum()), dtype=np.int64),
                "is_random": np.zeros(int(validation_keep.sum()), dtype=np.int64),
                "history_item_ids": pa.array(
                    [history_map[r] for r in values["request_id"][validation_keep]],
                    type=pa.list_(pa.int64()),
                ),
            }
        )
        manifest["item_encoding"] = fit_item_encoding(
            candidates, values["item_id"][split == "train"]
        )
        manifest["history_length"] = settings["history_length"]
        manifest["history_policy"] = (
            "All nonrandom positive requests audited against strictly earlier millisecond events; same-time targets excluded"
        )
        cutoffs = settings["split_cutoffs_ms"]
        expected_split = np.where(
            times < cutoffs[0],
            "warmup",
            np.where(
                times < cutoffs[1], "train", np.where(times < cutoffs[2], "validation", "test")
            ),
        )
        expected_split = np.where(values["is_random"], "random", expected_split)
        require(
            (split == expected_split).all(), "Published split differs from reviewed UTC protocol"
        )
        provenance = json.loads((Path(__file__).parent / "vendor/ziqi/provenance.json").read_text())
        require(settings["publisher_commit"] == provenance["commit"], "Unreviewed publisher commit")
        require(
            all(
                file_sha256(Path(__file__).parent / "vendor/ziqi" / Path(name).name) == digest
                for name, digest in provenance["sha256"].items()
            ),
            "Publisher snapshot changed",
        )
        require(
            hashes == settings["verified_source_sha256"],
            "Source snapshot changed; re-audit it before training",
        )
        manifest["time_protocol_verified"] = True
        manifest["publisher_protocol"] = {
            "commit": provenance["commit"],
            "timezone": "UTC",
            "split_cutoffs_ms": cutoffs,
            "label": "official long_view copied without relabeling",
        }
    output.mkdir(parents=True)
    for name, table in (("train.parquet", train), ("candidates.parquet", candidate_table)):
        pq.write_table(table, output / name, row_group_size=settings["row_group_size"])
    if settings.get("integration"):
        pq.write_table(
            validation, output / "validation.parquet", row_group_size=settings["row_group_size"]
        )
        manifest["extra_files"] = [
            {
                "path": "validation.parquet",
                "rows": len(validation),
                "sha256": file_sha256(output / "validation.parquet"),
            }
        ]
        manifest["validation_time_ranges_ms"] = [ranges["validation"]]
    manifest["train_shards"] = [
        {
            "path": "train.parquet",
            "rows": len(train),
            "sha256": file_sha256(output / "train.parquet"),
        }
    ]
    manifest["candidates"] = {
        "path": "candidates.parquet",
        "rows": len(candidate_table),
        "sha256": file_sha256(output / "candidates.parquet"),
    }
    report = {
        "source_rows": len(events),
        "split_counts": counts,
        "split_time_ranges_ms": ranges,
        "train_positive_rows": len(train),
        "training_users": len(manifest["user_ids"]),
        "candidate_items": len(ids),
        "history_rows": len(histories),
        "missing_duration_items": int(np.isnan(raw_duration).sum()),
        "missing_category_items": int((categories == settings["category_missing"]).sum()),
        "source_sha256": hashes,
        "histories_used": bool(settings.get("integration")),
        "time_protocol_verified": manifest["time_protocol_verified"],
        "scope": "Pure integration preparation; not the 1K Week 1 acceptance experiment",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def check_data(path, config, *, preflight=False):
    """Validate every prepared row and one User Tower forward; no optimizer or run."""
    manifest, root, _ = read_manifest(path, require_time_protocol=False)
    cfg = config["training"]
    batches = ParquetBatches(
        manifest, root, cfg["batch_size"], config["experiment"]["seed"], cfg["cache_row_groups"]
    )
    for index in range(len(batches.groups)):
        batches._group(index)
    if manifest.get("validation_time_ranges_ms"):
        vm = dict(
            manifest,
            train_shards=manifest["extra_files"],
            train_time_ranges_ms=manifest["validation_time_ranges_ms"],
        )
        validation = ParquetBatches(
            vm,
            root,
            cfg["batch_size"],
            config["experiment"]["seed"],
            cfg["cache_row_groups"],
            allow_unknown_users=True,
        )
        for index in range(len(validation.groups)):
            validation._group(index)
    _, sample = next(batches.batches(0))
    tower = UserTower(manifest["user_ids"], config["model"]).eval()
    with torch.no_grad():
        vectors = tower(sample)
    require(
        vectors.shape == (len(sample["user_id"]), 64) and torch.isfinite(vectors).all(),
        "Invalid User Tower output",
    )
    blockers = []
    if not manifest["time_protocol_verified"]:
        blockers.append("Ziqi must confirm timestamp/date/timezone and feedback provenance")
    if preflight:
        try:
            import os

            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(cfg["deterministic"])
            device = torch.device(cfg["device"])
            if device.type == "cuda" and not torch.cuda.is_available():
                raise ValueError("CUDA unavailable; run this preflight on Colab GPU")
            from .model import batch_loss
            from .training import to_device

            model = build_model(config, manifest).to(device)
            batch = to_device(sample, device)
            with torch.autocast(
                device_type=device.type, enabled=cfg["mixed_precision"] and device.type == "cuda"
            ):
                users, items = model(batch)
                loss = batch_loss(users, items, batch["item_id"], cfg["temperature"])
            require(torch.isfinite(loss).item(), "Non-finite preflight loss")
            loss.backward()
            require(
                all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()),
                "Non-finite preflight gradients",
            )
            model.zero_grad(set_to_none=True)
            # No optimizer update and no W&B run are performed.

        except (ImportError, AttributeError, ValueError, KeyError, RuntimeError) as error:
            blockers.append(f"Item Tower interface: {error}")
        if config["wandb"]["mode"] == "online":
            import os

            if not os.environ.get("WANDB_API_KEY"):
                blockers.append("Set WANDB_API_KEY from Colab Secrets before online training")
            else:
                try:
                    import wandb

                    project_path = f"{config['wandb']['entity']}/{config['wandb']['project']}"
                    next(iter(wandb.Api().runs(project_path, per_page=1)), None)
                except Exception as error:
                    blockers.append(f"W&B project access check failed ({type(error).__name__})")
        if cfg["device"] == "cuda" and not torch.cuda.is_available():
            blockers.append("Run GPU preflight in Colab with CUDA available")
    result = {
        "data_valid": True,
        "all_train_rows_checked": batches.rows,
        "user_forward_shape": list(vectors.shape),
        "training_started": False,
        "blockers": blockers,
        "ready_for_training": preflight and not blockers,
    }
    print(json.dumps(result, indent=2))
    if preflight and blockers:
        raise ValueError("Training preflight blocked; see blockers above")
    return result
