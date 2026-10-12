"""Test split isolation, unit conversion and fail-closed published data intake."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kuairand.batch_source import read_manifest
from kuairand.prepare_pure import check_data, prepare_pure
from run import load_config


def source_tables(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    splits = ["warmup", "train", "train", "train", "validation", "test", "random"]
    events = pa.table(
        {
            "request_id": [str(i) for i in range(7)],
            "user_id": [1, 1, 2, 3, 9, 10, 11],
            "item_id": [0, 0, 1, 1, 0, 1, 0],
            "request_time_ms": [1000, 2000, 2001, 2002, 3000, 4000, 2500],
            "tab": [1] * 7,
            "long_view": [True, True, False, True, True, True, True],
            "is_random": [False] * 6 + [True],
            "split": splits,
        }
    )
    pq.write_table(events, source / "events.parquet")
    pq.write_table(
        pa.table({"item_id": [1, 0], "tag": [None, "12,65"], "video_duration": [None, 17200.0]}),
        source / "item_features.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "request_id": ["1"],
                "user_id": [1],
                "history_item_ids": pa.array([[1]], type=pa.list_(pa.int64())),
            }
        ),
        source / "user_histories.parquet",
    )
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": "KuaiRand-Pure",
                "rows": 7,
                "split_counts": {"warmup": 1, "train": 3, "validation": 1, "test": 1, "random": 1},
            }
        )
    )
    config, _ = load_config(__import__("pathlib").Path(__file__).parents[1] / "configs/pure.yaml")
    config["data_adapter"]["integration"] = False
    config["model"]["history_pooling"] = "none"
    return source, config


def test_pure_adapter_isolates_training_and_preserves_provenance(tmp_path):
    source, config = source_tables(tmp_path)
    output = tmp_path / "prepared"
    report = prepare_pure(source, output, config)
    train = pq.read_table(output / "train.parquet").to_pydict()
    assert train["request_id"] == ["1", "3"]
    assert train["category"] == [12, -1]
    assert train["duration"][0] == pytest.approx(17.2)
    assert train["duration"][1] == 0
    manifest, _, _ = read_manifest(output / "manifest.json", require_time_protocol=False)
    assert manifest["user_ids"] == [1, 2, 3]
    assert not manifest["time_protocol_verified"]
    assert report["histories_used"] is False
    assert check_data(output / "manifest.json", config)["all_train_rows_checked"] == 2
    with pytest.raises(ValueError, match="Publisher must verify"):
        read_manifest(output / "manifest.json")
    with pytest.raises(ValueError, match="preflight blocked"):
        check_data(output / "manifest.json", config, preflight=True)


@pytest.mark.parametrize("fault", ["random", "time", "missing_item", "history_user"])
def test_pure_rejects_incompatible_publication(tmp_path, fault):
    source, config = source_tables(tmp_path)
    name = "user_histories.parquet" if fault == "history_user" else "events.parquet"
    table = pq.read_table(source / name)
    data = table.to_pydict()
    if fault == "random":
        data["is_random"][1] = True
    elif fault == "time":
        data["request_time_ms"][1] = 4001
    elif fault == "missing_item":
        data["item_id"][1] = 999
    else:
        data["user_id"][0] = 999
    pq.write_table(pa.Table.from_pydict(data, schema=table.schema), source / name)
    with pytest.raises(ValueError):
        prepare_pure(source, tmp_path / "prepared", config)
    assert not (tmp_path / "prepared").exists()
