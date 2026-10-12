import copy
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import yaml
from run import load_config

from recommender.batch_source import ParquetBatches, file_sha256, read_manifest
from recommender.checkpoint import latest_checkpoint, load_checkpoint
from recommender.export import export_items
from recommender.model import batch_loss, build_model
from recommender.training import binding, train
from recommender.user_tower import UserTower


@pytest.fixture
def config():
    path = Path(__file__).resolve().parents[1] / "configs/week1.yaml"
    cfg = yaml.safe_load(path.read_text())
    cfg["model"].update(item_factory="fixture_item:build", dropout=0.2, batch_norm=True)
    cfg["training"].update(
        device="cpu", epochs=2, batch_size=3, checkpoint_every_steps=100, log_every_steps=1
    )
    cfg["wandb"]["mode"] = "disabled"
    return cfg


@pytest.fixture
def manifest(tmp_path):
    rows = 7  # singleton final batch must be consumed, not dropped
    data = {
        "user_id": [1, 2, 3, 1, 2, 3, 1],
        "item_id": [5, 5, 6, 7, 8, 9, 10],
        "time_ms": list(range(1000, 1000 + rows)),
        "tab": [0, 1, 2, 0, 1, 2, 0],
        "category": [0] * rows,
        "duration": [1.0] * rows,
        "long_view": [1] * rows,
        "is_random": [0] * rows,
    }
    pq.write_table(pa.table(data), tmp_path / "train.parquet", row_group_size=2)
    pq.write_table(
        pa.table({"item_id": [5, 6, 7, 8, 9, 10], "category": [0] * 6, "duration": [1.0] * 6}),
        tmp_path / "items.parquet",
    )

    def shard(name, count):
        return {"path": name, "rows": count, "sha256": file_sha256(tmp_path / name)}

    description = {
        "schema_version": 1,
        "dataset_version": "SYNTHETIC-TEST-ONLY",
        "feature_version": "fixture",
        "positive_only": True,
        "duration_unit": "seconds",
        "label": "long_view",
        "user_ids": [1, 2, 3],
        "time_protocol_verified": True,
        "train_time_ranges_ms": [[1000, 2000]],
        "train_shards": [shard("train.parquet", rows)],
        "candidates": shard("items.parquet", 6),
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(description))
    return path


def test_unknown_users_empty_history_singleton(config):
    tower = UserTower([1, 2], config["model"])
    for batch_size in (1, 3):
        features = {
            "user_id": torch.full((batch_size,), 999),
            "tab": torch.full((batch_size,), -1),
            "time_ms": torch.full((batch_size,), 1_650_000_000_000),
        }
        vector = tower.encode_users(features, mask=torch.zeros(batch_size, 0, dtype=torch.bool))
        assert vector.shape == (batch_size, 64)
        torch.testing.assert_close(vector.norm(dim=1), torch.ones(batch_size))
        vector.sum().backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in tower.parameters())


def test_duplicate_target_loss_hand_calculation():
    vectors = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    loss = batch_loss(vectors, vectors, torch.tensor([5, 5, 6]), 1)
    expected = (2 * np.log1p(np.exp(-1)) + np.log1p(2 * np.exp(-1))) / 3
    assert loss.item() == pytest.approx(expected)
    assert batch_loss(vectors, vectors, torch.tensor([5, 5, 5]), 1).item() == 0


def test_resume_same_parameters_optimizer_rng_and_coverage(config, manifest, tmp_path):
    full = train(config, manifest, tmp_path / "full")
    interrupted = train(config, manifest, tmp_path / "resume", stop_after=2)
    resumed = train(config, manifest, tmp_path / "resume", resume=interrupted)
    _, _, digest = read_manifest(manifest)
    expected = binding(config, digest, "cpu")
    a, b = load_checkpoint(full, expected), load_checkpoint(resumed, expected)
    assert a["consumed"] == b["consumed"] == 14
    assert a["global_step"] == b["global_step"] == 6
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
    for param_id, state in a["optimizer"]["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(
                value, b["optimizer"]["state"][param_id][key], rtol=0, atol=0
            )
    torch.testing.assert_close(a["rng"]["torch"], b["rng"]["torch"], rtol=0, atol=0)
    full_log = (tmp_path / "full/metrics.jsonl").read_text()
    resume_log = (tmp_path / "resume/metrics.jsonl").read_text()
    assert [json.loads(x)["train/loss"] for x in full_log.splitlines()] == [
        json.loads(x)["train/loss"] for x in resume_log.splitlines()
    ]
    assert json.loads((tmp_path / "resume/run_manifest.json").read_text())["complete"]


def test_checksum_versions_latest_and_export(config, manifest, tmp_path):
    first = train(config, manifest, tmp_path / "run", stop_after=1)
    final = train(config, manifest, tmp_path / "run", resume=first)
    _, _, digest = read_manifest(manifest)
    expected = binding(config, digest, "cpu")
    assert latest_checkpoint(tmp_path / "run/checkpoints") == final
    changed = dict(expected, config_sha256="wrong")
    with pytest.raises(ValueError, match="versions"):
        load_checkpoint(final, changed)
    release = export_items(config, manifest, final, tmp_path / "vectors", 6, 10)
    assert release["rows"] == 4
    ids, vectors = [], []
    for shard in release["shards"]:
        with np.load(tmp_path / "vectors" / shard["path"]) as data:
            ids.extend(data["item_id"].tolist())
            vectors.extend(data["vectors"])
    assert ids == [6, 7, 8, 9]
    model = build_model(config, read_manifest(manifest)[0])
    model.load_state_dict(load_checkpoint(final, expected)["model"])
    model.eval()
    reference = model.encode_items({"item_id": torch.tensor(ids)})
    torch.testing.assert_close(torch.tensor(np.array(vectors)), reference)
    final.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        load_checkpoint(final, expected)
    assert latest_checkpoint(tmp_path / "run/checkpoints") != final


def test_permutation_cursor_covers_every_event(manifest):
    metadata, root, _ = read_manifest(manifest)
    reader = ParquetBatches(metadata, root, 3, 42, 1)
    batches = list(reader.batches(0))
    times = torch.cat([x[1]["time_ms"] for x in batches]).tolist()
    assert sorted(times) == list(range(1000, 1007))
    assert torch.equal(list(reader.batches(0, start=1))[0][1]["time_ms"], batches[1][1]["time_ms"])
    assert len(reader.cache) <= 1


def test_config_rejects_invalid_values(config, tmp_path):
    path = tmp_path / "config.yaml"
    for field, value in (("batch_size", 0), ("learning_rate", float("nan")), ("betas", [1, 0])):
        invalid = copy.deepcopy(config)
        invalid["training"][field] = value
        path.write_text(yaml.safe_dump(invalid))
        with pytest.raises(ValueError):
            load_config(path)


def test_reader_rejects_future_and_random_events(manifest):
    data = json.loads(manifest.read_text())
    data["train_time_ranges_ms"] = [[1000, 1002]]
    manifest.write_text(json.dumps(data))
    metadata, root, _ = read_manifest(manifest)
    reader = ParquetBatches(metadata, root, 3, 42, 1)
    with pytest.raises(ValueError, match="out-of-window"):
        list(reader.batches(0))
    data["train_time_ranges_ms"] = [[1000, 2000]]
    shard = root / "train.parquet"
    table = pq.read_table(shard)
    table = table.set_column(
        table.schema.get_field_index("is_random"), "is_random", pa.array([1] * table.num_rows)
    )
    pq.write_table(table, shard)
    data["train_shards"][0]["sha256"] = file_sha256(shard)
    manifest.write_text(json.dumps(data))
    metadata, root, _ = read_manifest(manifest)
    with pytest.raises(ValueError, match="Main training"):
        list(ParquetBatches(metadata, root, 3, 42, 1).batches(0))
