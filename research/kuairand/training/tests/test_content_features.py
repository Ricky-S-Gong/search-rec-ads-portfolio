"""Publication content retains the frozen task and reaches target/history encoders."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import yaml
from test_item_integration import metadata

from evaluation.protocol import CONTENT_CATEGORICAL, CONTENT_NUMERIC, prepare_content
from recommender.batch_source import file_sha256
from recommender.item_tower import fit_item_encoding
from recommender.model import batch_loss, build_model


def content_config():
    return yaml.safe_load((Path(__file__).parents[1] / "configs/pure-content-enhanced.yaml").read_text())


def test_content_features_fit_training_only_and_history_uses_shared_tower():
    table = metadata()
    original = fit_item_encoding(table, [0, 1], CONTENT_CATEGORICAL, CONTENT_NUMERIC)
    changed = table.to_pydict()
    changed["author_id"][2] = 999
    changed["video_duration"][2] = 1e12
    newer = fit_item_encoding(pa.table(changed), [0, 1], CONTENT_CATEGORICAL, CONTENT_NUMERIC)
    for field in ("vocabularies", "median", "mean", "std"):
        assert original[field] == newer[field]
    assert "visible_status" not in original["categorical_names"]
    torch.manual_seed(42)
    model = build_model(content_config(), {"user_ids": [1, 2], "item_encoding": original})
    assert model.item_tower.tower.mlp[0].in_features == 60
    assert model.user_tower.mlp[0].in_features == 108
    batch = {
        "user_id": torch.tensor([1, 2]), "item_id": torch.tensor([0, 2]),
        "time_ms": torch.tensor([1000, 1001]), "tab": torch.tensor([0, 0]),
        "history_item_ids": torch.tensor([[1], [1]]), "history_mask": torch.ones(2, 1, dtype=torch.bool),
    }
    users, items = model(batch)
    assert users.shape == items.shape == (2, 64)
    torch.testing.assert_close(users.norm(dim=1), torch.ones(2))
    batch_loss(users, items, batch["item_id"], 0.1).backward()
    author_index = original["categorical"][1][1]
    assert model.item_tower.tower.embeddings[1].weight.grad[author_index].abs().sum() > 0
    # Holding ID fixed while changing a content value must affect the item vector.
    before = model.encode_items({"item_id": torch.tensor([0])}).detach()
    model.item_tower.numeric[1, 0] += 10
    after = model.encode_items({"item_id": torch.tensor([0])}).detach()
    assert not torch.allclose(before, after)


def test_content_package_preserves_records_and_rejects_unsafe_policy(tmp_path):
    source, base = tmp_path / "source", tmp_path / "base"
    source.mkdir()
    base.mkdir()
    pq.write_table(metadata(), source / "item_features.parquet")
    config = content_config()
    hashes = {"item_features.parquet": file_sha256(source / "item_features.parquet")}
    config["data_adapter"]["verified_source_sha256"] = hashes
    for name, ids, durations in [("train", [0, 1], [1.0, 3.0]), ("validation", [2], [100.0])]:
        pq.write_table(pa.table({"item_id": ids, "duration": durations}), base / (name + ".parquet"))
    pq.write_table(pa.table({"item_id": [0, 1, 2]}), base / "candidates.parquet")
    def entry(name):
        return {"path": name, "rows": pq.ParquetFile(base / name).metadata.num_rows, "sha256": file_sha256(base / name)}
    manifest = {
        "schema_version": 1, "positive_only": True, "label": "long_view", "duration_unit": "seconds",
        "dataset_version": "synthetic-completion-v1", "feature_version": "id-only", "time_protocol_verified": True,
        "train_time_ranges_ms": [[1000, 2000]], "user_ids": [1, 2], "item_ids": [0, 1, 2],
        "train_shards": [entry("train.parquet")], "extra_files": [entry("validation.parquet")],
        "candidates": entry("candidates.parquet"), "protocol": {"test_used": False}, "source_sha256": hashes,
        "item_encoding": fit_item_encoding(metadata(), [0, 1], ("item_id",), ()),
    }
    path = base / "manifest.json"
    path.write_text(json.dumps(manifest))
    (base / "audit.json").write_text(json.dumps({"limitations": []}))
    out = tmp_path / "content"
    result = prepare_content(path, source, out, config)
    upgraded = json.loads((out / "manifest.json").read_text())
    assert result["metadata_policy"]["duration_request_checks"] == 3
    assert not result["metadata_policy"]["asof_versions_available"]
    assert result["item_input_dim"] == 60
    for field in ("train_shards", "extra_files", "candidates", "protocol"):
        assert upgraded[field] == manifest[field]
    for record in manifest["train_shards"] + manifest["extra_files"] + [manifest["candidates"]]:
        assert file_sha256(out / record["path"]) == record["sha256"]
    config["model"]["item_categorical_features"].append("visible_status")
    with pytest.raises(ValueError, match="feature policy"):
        prepare_content(path, source, tmp_path / "unsafe", config)
    assert not (tmp_path / "unsafe").exists()
