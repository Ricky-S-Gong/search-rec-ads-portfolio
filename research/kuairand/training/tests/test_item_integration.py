"""Verify actual upstream ItemTower plus history pooling, restore and export."""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

from recommender.batch_source import file_sha256, read_manifest
from recommender.checkpoint import load_checkpoint
from recommender.export import export_items
from recommender.item_tower import fit_item_encoding
from recommender.training import binding, train


def metadata():
    return pa.table(
        {
            "item_id": [0, 1, 2],
            "author_id": [3, 4, 9],
            "video_type": ["NORMAL", "NORMAL", "AD"],
            "upload_type": ["Web", "Web", "Other"],
            "visible_status": [0, 0, 0],
            "music_id": [8, 9, 10],
            "music_type": [1, 2, None],
            "tag": ["1", "2", "99"],
            "video_duration": [1000.0, 3000.0, 100000.0],
            "server_width": [100.0, 200.0, 9999.0],
            "server_height": [300.0, 400.0, 9999.0],
            "upload_dt": ["2022-04-09"] * 3,
        }
    )


def test_fit_ignores_heldout_metadata():
    a = metadata()
    original = fit_item_encoding(a, [0, 1])
    columns = a.to_pydict()
    columns["video_duration"][2] = 1e12
    columns["video_type"][2] = "FUTURE_ONLY"
    modified = fit_item_encoding(pa.table(columns), [0, 1])
    assert original["median"] == modified["median"]
    assert original["mean"] == modified["mean"]
    assert original["std"] == modified["std"]
    assert original["vocabularies"] == modified["vocabularies"]
    assert modified["categorical"][2][0] == 0


def test_actual_towers_resume_and_export(tmp_path):
    config = yaml.safe_load((Path(__file__).parents[1] / "configs/pure.yaml").read_text())
    config["training"].update(
        device="cpu", mixed_precision=False, epochs=1, batch_size=2, log_every_steps=1
    )
    config["wandb"]["mode"] = "disabled"
    rows = pa.table(
        {
            "user_id": [1, 2, 1, 2, 1],
            "item_id": [0, 1, 0, 1, 0],
            "time_ms": [1000, 1001, 1002, 1003, 1004],
            "tab": [0] * 5,
            "category": [1] * 5,
            "duration": [1.0] * 5,
            "long_view": [1] * 5,
            "is_random": [0] * 5,
            "history_item_ids": pa.array([[], [0], [1], [0, 1], [1]], type=pa.list_(pa.int64())),
        }
    )
    pq.write_table(rows, tmp_path / "train.parquet")
    pq.write_table(
        pa.table({"item_id": [0, 1, 2], "category": [1] * 3, "duration": [1.0] * 3}),
        tmp_path / "candidates.parquet",
    )
    val = rows.slice(0, 2).set_column(
        rows.schema.get_field_index("time_ms"), "time_ms", pa.array([2000, 2001])
    )
    pq.write_table(val, tmp_path / "validation.parquet")

    def entry(n):
        return {
            "path": n,
            "rows": pq.ParquetFile(tmp_path / n).metadata.num_rows,
            "sha256": file_sha256(tmp_path / n),
        }

    m = {
        "schema_version": 1,
        "positive_only": True,
        "label": "long_view",
        "duration_unit": "seconds",
        "dataset_version": "SYNTHETIC-INTEGRATION-ONLY",
        "feature_version": "train-only-test",
        "time_protocol_verified": True,
        "user_ids": [1, 2],
        "train_time_ranges_ms": [[1000, 1100]],
        "validation_time_ranges_ms": [[2000, 2100]],
        "history_length": 50,
        "train_shards": [entry("train.parquet")],
        "candidates": entry("candidates.parquet"),
        "extra_files": [entry("validation.parquet")],
        "item_encoding": fit_item_encoding(metadata(), [0, 1]),
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(m))
    full = train(config, path, tmp_path / "full")
    stop = train(config, path, tmp_path / "resume", stop_after=1)
    resumed = train(config, path, tmp_path / "resume", resume=stop)
    expected = binding(config, read_manifest(path)[2], "cpu")
    a, b = load_checkpoint(full, expected), load_checkpoint(resumed, expected)
    assert a["consumed"] == b["consumed"] == 5
    for k in a["model"]:
        torch.testing.assert_close(a["model"][k], b["model"][k], rtol=0, atol=0)
    assert (
        json.loads((tmp_path / "full/metrics.jsonl").read_text().splitlines()[-1])[
            "validation/samples"
        ]
        == 2
    )
    release = export_items(config, path, resumed, tmp_path / "export", 0, 3)
    assert release["rows"] == 3
    with np.load(tmp_path / "export/items-00000.npz") as z:
        assert z["item_id"].tolist() == [0, 1, 2]
        assert np.allclose(np.linalg.norm(z["vectors"], axis=1), 1, atol=1e-5)


def test_id_only_tower_ignores_all_snapshot_metadata():
    from recommender.item_tower import ItemAdapter

    config = yaml.safe_load((Path(__file__).parents[1] / "configs/pure-basic.yaml").read_text())
    encoding = fit_item_encoding(metadata(), [0, 1], ("item_id",), ())
    assert encoding["numeric"] == [[], [], []]
    model = ItemAdapter(config, {"item_encoding": encoding})
    vectors = model.encode_items({"item_id": torch.tensor([0, 1, 2])})
    assert vectors.shape == (3, 64)
    assert torch.isfinite(vectors).all()
    torch.testing.assert_close(vectors.norm(dim=1), torch.ones(3))
    bad = dict(config, model=dict(config["model"], item_categorical_features=["tag"]))
    import pytest

    with pytest.raises(ValueError, match="features differ"):
        ItemAdapter(bad, {"item_encoding": encoding})
