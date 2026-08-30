from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from scipy import sparse


FILES = (
    "interactions.parquet",
    "train.parquet",
    "validation.parquet",
    "test.parquet",
    "user_mapping.parquet",
    "item_mapping.parquet",
    "train_explicit.npz",
    "train_implicit.npz",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _matrix(frame: pd.DataFrame, values: pd.Series, shape: tuple[int, int]):
    return sparse.csr_matrix(
        (values, (frame["user_idx"], frame["item_idx"])), shape=shape
    )


def write_bundle(
    output_dir: Path,
    interactions: pd.DataFrame,
    users: pd.DataFrame,
    items: pd.DataFrame,
    *,
    seed: int,
    config: dict,
) -> dict:
    """Write the canonical shared dataset plus a machine-verifiable manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    required = {
        "user_idx", "item_idx", "rating", "is_read", "is_reviewed", "event_time", "split"
    }
    missing = required.difference(interactions.columns)
    if missing:
        raise ValueError(f"missing processed columns: {', '.join(sorted(missing))}")
    interactions.to_parquet(output_dir / "interactions.parquet", index=False)
    for split in ("train", "validation", "test"):
        interactions[interactions["split"] == split].to_parquet(
            output_dir / f"{split}.parquet", index=False
        )
    users.to_parquet(output_dir / "user_mapping.parquet", index=False)
    items.to_parquet(output_dir / "item_mapping.parquet", index=False)
    train = interactions[interactions["split"] == "train"]
    shape = (len(users), len(items))
    explicit = train[train["rating"] > 0]
    sparse.save_npz(
        output_dir / "train_explicit.npz",
        _matrix(explicit, explicit["rating"].astype("float32"), shape),
    )
    implicit = train[
        train["is_read"] | train["is_reviewed"] | train["rating"].gt(0)
    ]
    sparse.save_npz(
        output_dir / "train_implicit.npz",
        _matrix(implicit, pd.Series(1, index=implicit.index, dtype="float32"), shape),
    )
    manifest = {
        "version": str(config.get("version", "poetry-v1")),
        "seed": int(seed),
        "config": config,
        "counts": {
            "interactions": int(len(interactions)),
            "users": int(len(users)),
            "items": int(len(items)),
            "train": int((interactions["split"] == "train").sum()),
            "validation": int((interactions["split"] == "validation").sum()),
            "test": int((interactions["split"] == "test").sum()),
        },
        "sha256": {name: _sha256(output_dir / name) for name in FILES},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def verify_bundle(output_dir: Path, expected_manifest_path: Path | None = None) -> dict:
    """Refuse to train when artifacts or the canonical manifest differ."""
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("manifest.json is required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected in manifest.get("sha256", {}).items():
        path = output_dir / name
        if not path.exists() or _sha256(path) != expected:
            raise ValueError(f"checksum mismatch for {name}")
    if expected_manifest_path is not None:
        if not expected_manifest_path.exists():
            raise FileNotFoundError(f"canonical manifest is required: {expected_manifest_path}")
        expected_manifest = json.loads(expected_manifest_path.read_text(encoding="utf-8"))
        if manifest != expected_manifest:
            raise ValueError("local bundle does not match the canonical manifest")
    return manifest
