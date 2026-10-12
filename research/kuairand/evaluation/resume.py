"""Real-runtime interruption verification through the existing public trainer."""

import json
from pathlib import Path

import numpy as np
import torch

from recommender.batch_source import file_sha256, read_manifest
from recommender.checkpoint import latest_checkpoint, load_checkpoint
from recommender.training import binding, train


def verify_resume(config, manifest_path, reference, output):
    """Re-run the same deterministic experiment, interrupt, then compare full state."""
    output = Path(output)
    _, _, digest = read_manifest(manifest_path)
    expected = binding(config, digest, config["training"]["device"])
    full = load_checkpoint(reference, expected)
    if full["epoch"] != config["training"]["epochs"]:
        raise ValueError("Reference traversal must be complete")
    if output.exists():
        stop = latest_checkpoint(output / "checkpoints")
        progress = load_checkpoint(stop, expected)
        if progress["output"] != str(output.resolve()):
            raise ValueError("Recovery directory identity differs")
    else:
        stop = train(config, manifest_path, output, stop_after=25)
        progress = load_checkpoint(stop, expected)
    interrupted = []
    for marker in (output / "checkpoints").glob("*.complete.json"):
        path = marker.with_suffix("").with_suffix(".pt")
        candidate = load_checkpoint(path, expected)
        if candidate["global_step"] == 25 and candidate["epoch"] < config["training"]["epochs"]:
            interrupted.append(candidate)
    if not interrupted:
        raise ValueError("No verified step-25 interruption; require at least 26 batches")
    partial = interrupted[0]
    final = (
        stop
        if progress["epoch"] == config["training"]["epochs"]
        else train(config, manifest_path, output, resume=stop)
    )
    resumed = load_checkpoint(final, expected)

    def compare(a, b):
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a.cpu(), b.cpu(), rtol=0, atol=0)
        elif isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        elif isinstance(a, dict):
            if a.keys() != b.keys():
                raise ValueError("Recovery state keys differ")
            for key in a:
                compare(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            if len(a) != len(b):
                raise ValueError("Recovery state length differs")
            for x, y in zip(a, b):
                compare(x, y)
        elif a != b:
            raise ValueError("Recovery state differs")

    for name in ("model", "optimizer", "scaler", "epoch", "next_batch", "global_step", "consumed"):
        compare(full[name], resumed[name])
    compare(full["rng"]["torch"], resumed["rng"]["torch"])
    compare(full["rng"]["numpy"], resumed["rng"]["numpy"])
    compare(full["rng"]["python"], resumed["rng"]["python"])
    if expected["device_type"] == "cuda":
        compare(full["rng"]["cuda"], resumed["rng"]["cuda"])
    result = {
        "passed": True,
        "device": expected["device_type"],
        "manifest_sha256": digest,
        "checkpoint_sha256": file_sha256(reference),
        "resumed_checkpoint_sha256": file_sha256(final),
        "interruption_step": partial["global_step"],
        "next_batch_at_interruption": partial["next_batch"],
        "consumed": resumed["consumed"],
        "global_step": resumed["global_step"],
        "tolerance": "exact tensor equality (rtol=0, atol=0)",
    }
    (output / "resume_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
