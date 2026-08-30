import json

import pandas as pd
import pytest

from goodbooks_mf.artifacts import verify_bundle, write_bundle


def test_processed_bundle_has_checksums_and_verifies_on_another_machine(tmp_path):
    interactions = pd.DataFrame(
        {
            "user_idx": [0, 0, 0],
            "item_idx": [0, 1, 2],
            "rating": [5, 4, 3],
            "is_read": [True, True, True],
            "is_reviewed": [False, True, False],
            "event_time": pd.date_range("2017-01-01", periods=3, tz="UTC"),
            "split": ["train", "validation", "test"],
        }
    )
    users = pd.DataFrame({"user_id": ["u"], "user_idx": [0]})
    items = pd.DataFrame({"item_id": ["a", "b", "c"], "item_idx": [0, 1, 2]})

    manifest = write_bundle(tmp_path, interactions, users, items, seed=42, config={"version": "v1"})

    assert verify_bundle(tmp_path) == manifest
    persisted = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert persisted["counts"]["interactions"] == 3
    assert "interactions.parquet" in persisted["sha256"]
    assert (tmp_path / "train_explicit.npz").exists()
    assert (tmp_path / "train_implicit.npz").exists()


def test_verify_bundle_detects_a_modified_artifact(tmp_path):
    interactions = pd.DataFrame(
        {
            "user_idx": [0], "item_idx": [0], "rating": [5],
            "is_read": [True], "is_reviewed": [False],
            "event_time": [pd.Timestamp("2017-01-01", tz="UTC")], "split": ["train"],
        }
    )
    write_bundle(
        tmp_path,
        interactions,
        pd.DataFrame({"user_id": ["u"], "user_idx": [0]}),
        pd.DataFrame({"item_id": ["b"], "item_idx": [0]}),
        seed=42,
        config={},
    )
    (tmp_path / "interactions.parquet").write_bytes(b"changed")

    with pytest.raises(ValueError, match="checksum"):
        verify_bundle(tmp_path)
