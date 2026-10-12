"""Checksummed checkpoint publication and exact random-state restoration."""

import json
import os
import random
import uuid
from pathlib import Path

import numpy as np
import torch

from .batch_source import file_sha256


def random_state():
    """Capture all random streams used by the public trainer."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_random(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise ValueError("CUDA checkpoint requires a CUDA runtime")
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def save_checkpoint(directory, state):
    """Publish .pt then .complete.json; incomplete writes are never selected."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"step-{state['global_step']:09d}-{uuid.uuid4().hex[:8]}.pt"
    temporary = path.with_suffix(".pending")
    with temporary.open("wb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    marker = path.with_suffix(".complete.json")
    temp_marker = marker.with_suffix(".pending")
    temp_marker.write_text(
        json.dumps({"sha256": file_sha256(path), "global_step": state["global_step"]})
    )
    os.replace(temp_marker, marker)
    return path


def load_checkpoint(path, expected, device="cpu"):
    """Load a trusted private checkpoint only after checksum and version checks.

    torch pickle loading must never be used on untrusted downloaded checkpoints.
    """
    path = Path(path)
    marker = path.with_suffix(".complete.json")
    if not marker.exists():
        raise ValueError("Checkpoint has no completion marker")
    if json.loads(marker.read_text())["sha256"] != file_sha256(path):
        raise ValueError("Checkpoint checksum mismatch")
    state = torch.load(path, map_location=device, weights_only=False)
    if state.get("schema_version") != 1 or state.get("binding") != expected:
        raise ValueError("Checkpoint data/config/code/device versions do not match")
    return state


def latest_checkpoint(directory):
    """Find latest intact publication; ignore incomplete or corrupt files."""
    choices = []
    for marker in Path(directory).glob("*.complete.json"):
        try:
            metadata = json.loads(marker.read_text())
            path = marker.with_suffix("").with_suffix(".pt")
            if file_sha256(path) == metadata["sha256"]:
                choices.append((metadata["global_step"], marker.stat().st_mtime_ns, path))
        except (OSError, ValueError, KeyError):
            continue
    if not choices:
        raise ValueError("No complete checkpoint found")
    return max(choices)[2]
