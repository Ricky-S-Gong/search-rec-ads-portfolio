"""Public single-process trainer, manually launched, with deterministic cursors."""

import hashlib
import importlib
import inspect
import json
import os
import platform
import random
import time
import uuid
from pathlib import Path

import numpy as np
import torch

from .batch_source import ParquetBatches, file_sha256, read_manifest
from .checkpoint import load_checkpoint, random_state, restore_random, save_checkpoint
from .model import batch_loss, build_model


def config_sha(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()


def code_sha(config):
    """Bind all public Python components and item factory module to checkpoint."""
    files = list(Path(__file__).parent.rglob("*.py"))
    module = importlib.import_module(config["model"]["item_factory"].split(":")[0])
    files.append(Path(inspect.getfile(module)))
    hashes = sorted(
        (
            str(path.relative_to(Path(__file__).parent))
            if path.is_relative_to(Path(__file__).parent)
            else path.name,
            file_sha256(path),
        )
        for path in files
    )
    return hashlib.sha256(json.dumps(hashes).encode()).hexdigest()


def binding(config, manifest_hash, device):
    return {
        "config_sha256": config_sha(config),
        "manifest_sha256": manifest_hash,
        "code_sha256": code_sha(config),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "device_type": torch.device(device).type,
    }


def to_device(batch, device):
    return {name: value.to(device) for name, value in batch.items()}


def train(config, manifest_path, output, resume=None, stop_after=None):
    """Train published positives and save complete progress.

    stop_after is a bounded interruption hook for correctness tests. Returning a
    checkpoint does not imply full traversal; run_manifest records actual counts.
    """
    manifest, root, manifest_hash = read_manifest(manifest_path)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = torch.device(config["training"]["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; select a Colab GPU runtime")
    seed = config["experiment"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(config["training"]["deterministic"])
    source = ParquetBatches(
        manifest,
        root,
        config["training"]["batch_size"],
        seed,
        config["training"]["cache_row_groups"],
    )
    validation = None
    if manifest.get("validation_time_ranges_ms"):
        validation_manifest = dict(
            manifest,
            train_shards=manifest["extra_files"],
            train_time_ranges_ms=manifest["validation_time_ranges_ms"],
        )
        validation = ParquetBatches(
            validation_manifest,
            root,
            config["training"]["batch_size"],
            seed,
            config["training"]["cache_row_groups"],
            allow_unknown_users=True,
        )
    expected = binding(config, manifest_hash, device)
    model = build_model(config, manifest).to(device)
    settings = config["training"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=settings["learning_rate"],
        betas=tuple(settings["betas"]),
        weight_decay=settings["weight_decay"],
    )
    amp = settings["mixed_precision"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    epoch, cursor, step, consumed = 0, 0, 0, 0
    run_id = uuid.uuid4().hex[:12]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if resume:
        saved = load_checkpoint(resume, expected)
        # Prevent mixing logs/checkpoints into an unrelated output directory.
        if saved["output"] != str(output.resolve()):
            raise ValueError("Resume must use the original persistent output directory")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        epoch, cursor = saved["epoch"], saved["next_batch"]
        step, consumed, run_id = saved["global_step"], saved["consumed"], saved["run_id"]
    elif any(output.iterdir()):
        raise ValueError("New run requires an empty output directory")

    tracking = None
    if config["wandb"]["mode"] != "disabled":
        import wandb

        options = config["wandb"]
        tracking = wandb.init(
            entity=options["entity"],
            project=options["project"],
            id=run_id,
            name=options["name"],
            group=options["group"],
            tags=options["tags"],
            config=config,
            mode=options["mode"],
            job_type="train",
            resume="must" if resume else "never",
            dir=str(output),
        )
        tracking.define_metric("train_step")
        tracking.define_metric("train/*", step_metric="train_step")
        if resume:
            tracking.log({"resume/from_checkpoint_step": step})
    # Restore last, after model/optimizer/W&B initialization.
    if resume:
        restore_random(saved["rng"])
    model.train()
    (output / "config.json").write_text(json.dumps(config, indent=2))
    last_checkpoint = None

    def persist():
        nonlocal last_checkpoint
        state = {
            "schema_version": 1,
            "binding": expected,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "rng": random_state(),
            "epoch": epoch,
            "next_batch": cursor,
            "global_step": step,
            "consumed": consumed,
            "run_id": run_id,
            "wandb": {k: config["wandb"][k] for k in ("entity", "project")},
            "output": str(output.resolve()),
        }
        last_checkpoint = save_checkpoint(output / "checkpoints", state)
        report = {
            "binding": expected,
            "dataset_version": manifest["dataset_version"],
            "feature_version": manifest["feature_version"],
            "run_id": run_id,
            "consumed": consumed,
            "target_samples": source.rows * settings["epochs"],
            "complete": epoch == settings["epochs"],
            "checkpoint": str(last_checkpoint.resolve()),
        }
        temp = output / "run_manifest.pending"
        temp.write_text(json.dumps(report, indent=2))
        temp.replace(output / "run_manifest.json")

    try:
        if not resume:
            persist()
        while epoch < settings["epochs"]:
            iterator = iter(source.batches(epoch, cursor))
            while True:
                waiting = time.perf_counter()
                try:
                    next_cursor, cpu_batch = next(iterator)
                except StopIteration:
                    break
                data_seconds = time.perf_counter() - waiting
                start = time.perf_counter()
                batch = to_device(cpu_batch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp):
                    users, items = model(batch)
                    loss = batch_loss(users, items, batch["item_id"], settings["temperature"])
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite loss; stop and inspect inputs")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in model.parameters()
                ):
                    raise ValueError("Non-finite gradient; no optimizer update performed")
                scaler.step(optimizer)
                scaler.update()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                seconds = time.perf_counter() - start
                step, cursor, consumed = step + 1, next_cursor, consumed + len(batch["user_id"])
                metrics = {
                    "train_step": step,
                    "train/loss": loss.item(),
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                    "train/consumed": consumed,
                    "train/data_wait_seconds": data_seconds,
                    "train/samples_per_second": len(batch["user_id"])
                    / max(seconds + data_seconds, 1e-9),
                    "epoch": epoch,
                    "next_batch": cursor,
                }
                metrics["train/step_seconds"] = seconds
                if device.type == "cuda":
                    metrics["train/peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated()
                with (output / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(metrics) + "\n")
                if step % settings["log_every_steps"] == 0 or cursor == source.num_batches:
                    print(json.dumps(metrics), flush=True)
                    if tracking:
                        tracking.log(metrics)
                if step % settings["checkpoint_every_steps"] == 0:
                    persist()
                if stop_after is not None and step >= stop_after:
                    persist()
                    return last_checkpoint
            if validation is not None:
                model.eval()
                loss_sum, count = 0.0, 0
                with torch.no_grad():
                    for _, cpu_batch in validation.batches(0):
                        batch = to_device(cpu_batch, device)
                        with torch.autocast(device_type=device.type, enabled=amp):
                            users, items = model(batch)
                            value = batch_loss(
                                users, items, batch["item_id"], settings["temperature"]
                            )
                        if not torch.isfinite(value):
                            raise ValueError("Non-finite validation loss")
                        size = len(batch["user_id"])
                        loss_sum += value.item() * size
                        count += size
                validation_metrics = {
                    "train_step": step,
                    "validation/inbatch_loss": loss_sum / count,
                    "validation/samples": count,
                }
                print(json.dumps(validation_metrics), flush=True)
                if tracking:
                    tracking.log(validation_metrics)
                with (output / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(validation_metrics) + "\n")
                model.train()
            epoch, cursor = epoch + 1, 0
            persist()
        if last_checkpoint is None:
            persist()
        return last_checkpoint
    finally:
        if tracking:
            tracking.finish()
