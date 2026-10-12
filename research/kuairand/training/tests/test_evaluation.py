import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))
from evaluation.evaluate import stage_report
from evaluation.protocol import completed_histories
from evaluation.retrieval import exact_topk, ranking_metrics


def test_feedback_completion_and_tied_requests():
    histories = completed_histories(
        np.array([1, 1, 1, 1, 1]),
        np.array([5, 6, 7, 8, 9]),
        np.array([0, 5, 10, 10, 21]),
        np.array([20, 6, 10, 10, 22]),
        limit=3,
    )
    assert histories == [[], [], [6], [6], [6, 7, 8]]
    # Out-of-order completion keeps original request order, bounded to recent 3.
    assert completed_histories(
        np.array([1, 1, 1]), np.array([5, 6, 7]), np.array([0, 5, 21]), np.array([20, 6, 22])
    )[-1] == [5, 6]


def test_exact_blocks_ties_eligibility_and_missing_slots():
    ids = np.array([9, 2, 4, 1])
    vectors = np.array([[1.0, 0], [1, 0], [0, 1], [1, 0]])
    users = np.array([[1.0, 0], [0, 1]])
    times = np.array([10, 20])
    after = np.array([0, 0, 11, 20])
    a, s = exact_topk(users, vectors, ids, times, after, k=5, item_block=1)
    b, _ = exact_topk(users, vectors, ids, times, after, k=5, item_block=99)
    np.testing.assert_array_equal(a, b)
    assert a.tolist() == [[2, 9, -1, -1, -1], [4, 1, 2, 9, -1]]
    assert np.isneginf(s[0, 2:]).all()
    with pytest.raises(ValueError, match="unique"):
        exact_topk(users, vectors, np.array([1, 1, 2, 3]), times, after)
    with pytest.raises(ValueError, match="Finite"):
        exact_topk(users * np.nan, vectors, ids, times, after)


def test_metrics_keep_uncovered_target_in_denominator():
    m = ranking_metrics(
        np.array([[2, 3, -1], [4, 2, 3], [2, 3, 4]]),
        np.array([2, 2, 99]),
        np.array([True, True, False]),
    )
    assert m["recall_at_100"] == pytest.approx(2 / 3)
    assert m["ndcg_at_100"] == pytest.approx((1 + 1 / np.log2(3)) / 3)
    assert m["covered_recall_at_100"] == 1
    assert m["target_coverage"] == pytest.approx(2 / 3)
    assert m["insufficient_k_rate"] == pytest.approx(1 / 3)
    assert (
        ranking_metrics(np.empty((0, 100)), np.array([]), np.array([], dtype=bool))["recall_at_100"]
        is None
    )


def test_stage_cannot_claim_completion_or_compare_different_protocols(tmp_path):
    def write(name, protocol):
        p = tmp_path / (name + ".json")
        p.write_text(
            json.dumps(
                {
                    "model": name,
                    "protocol": protocol,
                    "evaluator_sha256": "fixed",
                    "metrics": {
                        "requests": 1,
                        "recall_at_100": 0.0,
                        "ndcg_at_100": 0.0,
                        "target_coverage": 1.0,
                    },
                    "artifact": {},
                }
            )
        )
        return p

    pop = write("popularity", {"requests_sha256": "a"})
    report = stage_report([pop], tmp_path / "stage")
    assert not report["complete"] and "basic" in report["blockers"]
    with pytest.raises(ValueError, match="Different"):
        stage_report([pop, write("basic", {"requests_sha256": "b"})], tmp_path / "bad")


def test_real_model_export_evaluate_pipeline_and_binding_rejection(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import yaml

    from evaluation.evaluate import evaluate
    from recommender.batch_source import file_sha256
    from recommender.export import export_items
    from recommender.item_tower import fit_item_encoding
    from recommender.training import train

    config = yaml.safe_load((Path(__file__).parents[1] / "configs/pure-basic.yaml").read_text())
    config["training"].update(device="cpu", mixed_precision=False, batch_size=2)
    config["wandb"]["mode"] = "disabled"
    train_table = pa.table(
        {
            "request_id": ["a", "b", "c"],
            "user_id": [1, 2, 1],
            "item_id": [0, 1, 0],
            "time_ms": [1000, 1001, 1002],
            "tab": [1, 1, 1],
            "category": [0, 0, 0],
            "duration": [1.0, 1.0, 1.0],
            "long_view": [1, 1, 1],
            "is_random": [0, 0, 0],
            "history_item_ids": pa.array([[], [], [0]], type=pa.list_(pa.int64())),
        }
    )
    validation = pa.table(
        {
            "request_id": ["v1", "v2"],
            "user_id": [1, 999],
            "item_id": [0, 1],
            "time_ms": [2000, 2001],
            "tab": [1, 1],
            "category": [0, 0],
            "duration": [1.0, 1.0],
            "long_view": [1, 1],
            "is_random": [0, 0],
            "history_item_ids": pa.array([[], [0]], type=pa.list_(pa.int64())),
        }
    )
    candidates = pa.table(
        {
            "item_id": [0, 1],
            "category": [0, 0],
            "duration": [0.0, 0.0],
            "eligible_after_ms": [0, 2002],
        }
    )
    for name, table in [
        ("train.parquet", train_table),
        ("validation.parquet", validation),
        ("candidates.parquet", candidates),
    ]:
        pq.write_table(table, tmp_path / name)

    def entry(name):
        return {
            "path": name,
            "sha256": file_sha256(tmp_path / name),
            "rows": pq.ParquetFile(tmp_path / name).metadata.num_rows,
        }

    manifest = {
        "schema_version": 1,
        "positive_only": True,
        "label": "long_view",
        "duration_unit": "seconds",
        "dataset_version": "SYNTHETIC-PIPELINE-TEST",
        "feature_version": "id-only",
        "user_ids": [1, 2],
        "item_ids": [0, 1],
        "time_protocol_verified": True,
        "train_time_ranges_ms": [[1000, 1100]],
        "validation_time_ranges_ms": [[2000, 2100]],
        "train_shards": [entry("train.parquet")],
        "extra_files": [entry("validation.parquet")],
        "candidates": entry("candidates.parquet"),
        "history_length": 50,
        "item_encoding": fit_item_encoding(pa.table({"item_id": [0, 1]}), [0, 1], ("item_id",), ()),
        "protocol": {"test_used": False, "repeat_items_allowed": True},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    checkpoint = train(config, path, tmp_path / "model")
    export_items(config, path, checkpoint, tmp_path / "vectors", 0, 2)
    basic = evaluate(
        config, path, tmp_path / "basic", checkpoint, tmp_path / "vectors", "basic", request_batch=1
    )
    pop = evaluate(config, path, tmp_path / "popularity", request_batch=1)
    assert basic["protocol"] == pop["protocol"]
    assert basic["metrics"]["recall_at_100"] == 0.5
    assert basic["metrics"]["target_coverage"] == 0.5
    assert basic["metrics"]["unknown_user_rate"] == 0.5
    assert basic["metrics"]["empty_history_rate"] == 0.5
    vector_manifest = tmp_path / "vectors/manifest.json"
    release = json.loads(vector_manifest.read_text())
    release["checkpoint_sha256"] = "wrong"
    vector_manifest.write_text(json.dumps(release))
    with pytest.raises(ValueError, match="binding mismatch"):
        evaluate(config, path, tmp_path / "bad", checkpoint, tmp_path / "vectors", "basic")
