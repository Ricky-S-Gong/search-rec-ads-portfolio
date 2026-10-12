"""Evaluate the frozen validation package and fail closed on artifact mismatches."""

import hashlib
import json
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from recommender.batch_source import file_sha256, read_manifest
from recommender.checkpoint import load_checkpoint
from recommender.model import build_model
from recommender.training import binding, code_sha

from .retrieval import exact_topk, ranking_metrics


def validation_protocol(manifest, root, digest):
    validation = next(
        (f for f in manifest["extra_files"] if f["path"] == "validation.parquet"), None
    )
    if validation is None or manifest.get("protocol", {}).get("test_used") is not False:
        raise ValueError("Require a frozen validation-only protocol")
    requests = pq.read_table(root / validation["path"]).sort_by([("request_id", "ascending")])
    request_ids = requests["request_id"].to_pylist()
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("Duplicate validation request ID")
    train_ids = set()
    for shard in manifest["train_shards"]:
        train_ids.update(
            pq.read_table(root / shard["path"], columns=["request_id"])["request_id"].to_pylist()
        )
    if train_ids.intersection(request_ids):
        raise ValueError("Training/validation request overlap")
    candidates = pq.read_table(root / manifest["candidates"]["path"]).sort_by(
        [("item_id", "ascending")]
    )
    ids = candidates["item_id"].to_numpy()
    if len(np.unique(ids)) != len(ids) or not np.array_equal(ids, manifest["item_ids"]):
        raise ValueError("Candidate identity mismatch")
    times = requests["time_ms"].to_numpy()
    lower, upper = manifest["validation_time_ranges_ms"][0]
    if not ((times >= lower) & (times < upper)).all():
        raise ValueError("Out-of-window validation request")
    protocol = {
        "manifest_sha256": digest,
        "requests_sha256": hashlib.sha256(
            json.dumps(request_ids, separators=(",", ":")).encode()
        ).hexdigest(),
        "validation_sha256": validation["sha256"],
        "candidates_sha256": manifest["candidates"]["sha256"],
        "k": 100,
        "tie_break": "score descending, raw item_id ascending",
        **manifest["protocol"],
    }
    return requests, candidates, protocol


def evaluate(
    config,
    manifest_path,
    output,
    checkpoint=None,
    vectors_path=None,
    label="popularity",
    request_batch=256,
):
    manifest, root, digest = read_manifest(manifest_path)
    requests, candidates, protocol = validation_protocol(manifest, root, digest)
    output = Path(output)
    if output.exists():
        raise ValueError("Use a new evaluation directory")
    if request_batch < 1:
        raise ValueError("Positive request batch required")
    ids, after = candidates["item_id"].to_numpy(), candidates["eligible_after_ms"].to_numpy()
    model, item_vectors, artifact = None, None, {}
    device = torch.device(config["training"]["device"])
    if checkpoint is not None:
        if vectors_path is None or label not in ("basic", "enhanced"):
            raise ValueError("Model evaluation requires a named basic/enhanced model and vectors")
        saved = load_checkpoint(checkpoint, binding(config, digest, device))
        if saved["epoch"] != config["training"]["epochs"]:
            raise ValueError("Model training is incomplete")
        vectors_path = Path(vectors_path)
        release = json.loads((vectors_path / "manifest.json").read_text())
        if (
            release["binding"] != saved["binding"]
            or release["checkpoint_sha256"] != file_sha256(checkpoint)
            or not release["training_complete"]
            or release["dimension"] != 64
        ):
            raise ValueError("Vector/checkpoint binding mismatch")
        vector_ids, blocks = [], []
        for shard in release["shards"]:
            path = vectors_path / shard["path"]
            if file_sha256(path) != shard["sha256"]:
                raise ValueError("Vector checksum mismatch")
            with np.load(path, allow_pickle=False) as part:
                if len(part["item_id"]) != shard["rows"]:
                    raise ValueError("Vector row count mismatch")
                vector_ids.extend(part["item_id"].tolist())
                blocks.append(part["vectors"])
        item_vectors = np.concatenate(blocks)
        if (
            not np.array_equal(vector_ids, ids)
            or item_vectors.shape != (len(ids), 64)
            or release["rows"] != len(ids)
            or not np.isfinite(item_vectors).all()
            or not np.allclose(np.linalg.norm(item_vectors, axis=1), 1, atol=1e-5)
        ):
            raise ValueError("Full normalized candidate vector coverage required")
        model = build_model(config, manifest).to(device)
        model.load_state_dict(saved["model"])
        model.eval()
        artifact = {
            "checkpoint_sha256": file_sha256(checkpoint),
            "vector_manifest_sha256": file_sha256(vectors_path / "manifest.json"),
            "binding": saved["binding"],
            "run_id": saved["run_id"],
            "consumed": saved["consumed"],
            "expected_samples": sum(s["rows"] for s in manifest["train_shards"])
            * config["training"]["epochs"],
        }
        if artifact["consumed"] != artifact["expected_samples"]:
            raise ValueError("Training consumption mismatch")
    else:
        if vectors_path is not None or label != "popularity":
            raise ValueError("Popularity evaluation must not claim model artifacts")
        counts = Counter()
        for shard in manifest["train_shards"]:
            counts.update(
                pq.read_table(root / shard["path"], columns=["item_id"])["item_id"].to_pylist()
            )
        item_vectors = np.array([counts.get(int(i), 0) for i in ids], dtype=np.float64)[:, None]
    output.mkdir(parents=True)
    predictions, encoding_seconds, retrieval_seconds, batch_latencies = [], 0.0, 0.0, []
    target = requests["item_id"].to_numpy()
    positions = np.searchsorted(ids, target)
    safe = positions.clip(max=len(ids) - 1)
    covered = (
        (positions < len(ids))
        & (ids[safe] == target)
        & (after[safe] <= requests["time_ms"].to_numpy())
    )
    histories = requests["history_item_ids"].to_pylist()

    def encode(start, stop):
        if model is None:
            return np.ones((stop - start, 1))
        features = {
            n: torch.tensor(
                requests[n].slice(start, stop - start).to_numpy(), dtype=torch.long, device=device
            )
            for n in ("user_id", "time_ms", "tab")
        }
        h = np.full((stop - start, manifest["history_length"]), -1, dtype=np.int64)
        for row, history in enumerate(histories[start:stop]):
            if history:
                h[row, -len(history) :] = history
        with torch.no_grad():
            history_vectors = (
                model.item_tower.encode_items({"item_id": torch.tensor(h, device=device)})
                if model.user_tower.history_dim
                else None
            )
            users = model.encode_users(
                features, history_vectors, torch.tensor(h >= 0, device=device)
            )
        if not torch.isfinite(users).all() or not torch.allclose(
            users.norm(dim=1), torch.ones(len(users), device=device), atol=1e-5
        ):
            raise ValueError("Invalid user vectors")
        return users.cpu().numpy()

    # Warm up the same encoding and retrieval path; exclude this call from timing.
    warm = min(request_batch, len(requests))
    if warm:
        exact_topk(
            encode(0, warm), item_vectors, ids, requests["time_ms"].slice(0, warm).to_numpy(), after
        )
    for start in range(0, len(requests), request_batch):
        stop = min(start + request_batch, len(requests))
        began = time.perf_counter()
        users = encode(start, stop)
        encoded = time.perf_counter()
        pred, _ = exact_topk(
            users,
            item_vectors,
            ids,
            requests["time_ms"].slice(start, stop - start).to_numpy(),
            after,
        )
        end = time.perf_counter()
        encoding_seconds += encoded - began
        retrieval_seconds += end - encoded
        batch_latencies.append(end - began)
        predictions.append(pred)
    predictions = np.concatenate(predictions)
    users = requests["user_id"].to_numpy()
    metrics = ranking_metrics(predictions, target, covered)
    metrics.update(
        unknown_user_rate=float((~np.isin(users, manifest["user_ids"])).mean()),
        empty_history_rate=float(np.array([not h for h in histories]).mean()),
    )
    pq.write_table(
        pa.table(
            {
                "request_id": requests["request_id"],
                "target_id": target,
                "covered": covered,
                "recommendations": pa.array(predictions.tolist(), type=pa.list_(pa.int64())),
            }
        ),
        output / "predictions.parquet",
    )
    result = {
        "schema_version": 1,
        "model": label,
        "protocol": protocol,
        "artifact": artifact,
        "metrics": metrics,
        "performance": {
            "device": str(device) if model else "cpu",
            "platform": platform.platform(),
            "request_batch": request_batch,
            "warmup_batches": 1,
            "encoding_seconds": encoding_seconds,
            "retrieval_seconds": retrieval_seconds,
            "p50_batch_seconds": float(np.quantile(batch_latencies, 0.5)),
            "p95_batch_seconds": float(np.quantile(batch_latencies, 0.95)),
            "includes": "user feature tensor assembly, user encoding, GPU-to-CPU transfer if used, CPU exact retrieval; excludes initial model/data loading",
        },
        "evaluator_sha256": hashlib.sha256(
            b"".join(p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py")))
        ).hexdigest(),
        "training_code_sha256": code_sha(config),
        "predictions_sha256": file_sha256(output / "predictions.parquet"),
    }
    (output / "evaluation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def stage_report(result_paths, output, resume_check=None):
    """Keep incomplete stages explicit; missing experiments are never zero-filled."""
    results = [json.loads(Path(p).read_text()) for p in result_paths]
    if not results or len({r["model"] for r in results}) != len(results):
        raise ValueError("Require distinct evaluation results")
    if any(
        r["protocol"] != results[0]["protocol"]
        or r["evaluator_sha256"] != results[0]["evaluator_sha256"]
        for r in results
    ):
        raise ValueError("Different request/candidate/evaluator protocols cannot be compared")
    models = {r["model"]: r for r in results}
    missing = sorted({"popularity", "basic", "enhanced"} - models.keys())
    resume = json.loads(Path(resume_check).read_text()) if resume_check else None
    gpu_models = all(
        name in models and models[name]["artifact"].get("binding", {}).get("device_type") == "cuda"
        for name in ("basic", "enhanced")
    )
    recovery_ok = bool(
        resume
        and resume.get("passed") is True
        and resume.get("device") == "cuda"
        and resume.get("manifest_sha256") == results[0]["protocol"]["manifest_sha256"]
        and resume.get("checkpoint_sha256")
        == models.get("enhanced", {}).get("artifact", {}).get("checkpoint_sha256")
    )
    blockers = (
        missing
        + ([] if gpu_models else ["real CUDA model runs"])
        + ([] if recovery_ok else ["matching CUDA interruption/resume verification"])
    )
    report = {
        "schema_version": 1,
        "stage": "week1-pure",
        "complete": not blockers,
        "blockers": blockers,
        "protocol": results[0]["protocol"],
        "models": models,
        "resume_verification": resume,
    }
    output = Path(output)
    if output.exists():
        raise ValueError("Use a new stage release directory")
    output.mkdir(parents=True)
    (output / "stage_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    release = {
        "schema_version": 1,
        "stage_report_sha256": file_sha256(output / "stage_report.json"),
        "evaluation_sha256": {
            json.loads(Path(p).read_text())["model"]: file_sha256(p) for p in result_paths
        },
        "complete": not blockers,
        "protocol": report["protocol"],
    }
    (output / "release_manifest.json").write_text(json.dumps(release, indent=2) + "\n")
    lines = [
        "# Week 1 Pure stage",
        f"Complete: {not blockers}",
        "",
        "| Model | Requests | Recall@100 | NDCG@100 | Coverage |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, r in models.items():
        m = r["metrics"]
        lines.append(
            f"| {name} | {m['requests']} | {m['recall_at_100']:.6f} | {m['ndcg_at_100']:.6f} | {m['target_coverage']:.6f} |"
        )
    lines += ["", "Outstanding: " + (", ".join(blockers) or "none")]
    (output / "stage_report.md").write_text("\n".join(lines) + "\n")
    return report
