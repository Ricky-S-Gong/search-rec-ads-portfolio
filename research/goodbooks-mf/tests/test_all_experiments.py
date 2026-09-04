import json
from pathlib import Path

import pandas as pd
import pytest

from goodbooks_mf.artifacts import write_bundle
from goodbooks_mf.experiment import (
    RESULT_FIELDS,
    freeze_validation_configs,
    load_experiment_config,
    load_test_after_freeze,
    run_frozen_rating_test,
    run_frozen_unified_test,
    run_validation_selection,
)
from run_unified_evaluation import write_summary_chart, write_summary_table


def tiny_bundle(tmp_path: Path) -> Path:
    interactions = pd.DataFrame(
        {
            "user_idx": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "item_idx": [0, 1, 2, 1, 2, 0, 2, 0, 1],
            "rating": [5, 4, 3, 4, 5, 2, 3, 2, 5],
            "is_read": [True] * 9,
            "is_reviewed": [False] * 9,
            "event_time": pd.date_range("2017-01-01", periods=9, tz="UTC"),
            "split": ["train", "validation", "test"] * 3,
        }
    )
    users = pd.DataFrame({"user_id": ["u0", "u1", "u2"], "user_idx": range(3)})
    items = pd.DataFrame({"item_id": ["b0", "b1", "b2"], "item_idx": range(3)})
    data_dir = tmp_path / "data"
    write_bundle(data_dir, interactions, users, items, seed=7, config={"version": "tiny-v1"})
    return data_dir


def tiny_config() -> dict:
    return {
        "schema_version": "goodbooks-experiment-config-v1",
        "dataset_version": "tiny-v1",
        "seed": 7,
        "selection_metric": "rmse",
        "rating_clip": [1.0, 5.0],
        "models": {
            "basic_mf": {
                "candidates": [
                    {
                        "n_factors": 2,
                        "learning_rate": 0.01,
                        "reg_lambda": 0.02,
                        "n_epochs": 2,
                        "patience": 1,
                    }
                ]
            },
            "funksvd": {
                "candidates": [
                    {
                        "n_factors": 2,
                        "learning_rate": 0.01,
                        "reg_lambda": 0.02,
                        "n_epochs": 2,
                        "patience": 1,
                    }
                ]
            },
            "als": {
                "candidates": [
                    {"n_factors": 2, "reg_lambda": 0.1, "n_iterations": 2}
                ]
            },
            "nmf": {
                "candidates": [
                    {
                        "n_factors": 2,
                        "max_iter": 2,
                        "reg_lambda": 0.0,
                        "tol": 0.0001,
                    }
                ]
            },
            "bias_aware_als": {
                "candidates": [
                    {
                        "n_factors": 2,
                        "reg_lambda": 1.0,
                        "n_iterations": 2,
                        "bias_reg_lambda": 1.0,
                        "bias_iterations": 2,
                    }
                ]
            },
        },
    }


def test_validation_runner_never_deserializes_test_split(tmp_path):
    data_dir = tiny_bundle(tmp_path)
    reads: list[str] = []

    def recording_reader(path):
        reads.append(Path(path).name)
        if Path(path).name == "test.parquet":
            raise AssertionError("validation selection must not read test.parquet")
        return pd.read_parquet(path)

    payload = run_validation_selection(
        data_dir,
        tiny_config(),
        parquet_reader=recording_reader,
    )

    assert reads == ["train.parquet", "validation.parquet"]
    assert payload["phase"] == "validation_selection"
    assert payload["test_accessed"] is False


def test_validation_runner_supports_all_local_models_and_complete_schema(tmp_path):
    payload = run_validation_selection(tiny_bundle(tmp_path), tiny_config())

    assert set(payload["best_configs"]) == {
        "basic_mf",
        "funksvd",
        "als",
        "nmf",
        "bias_aware_als",
    }
    assert len(payload["results"]) == 5
    for row in payload["results"]:
        assert set(RESULT_FIELDS).issubset(row)
        assert row["evaluation_split"] == "validation"
        assert row["evaluated_rating_count"] == 3
        assert row["precision_at_5"] is None
        assert row["evaluated_rating_users"] is None
        assert row["evaluated_ranking_users"] is None
        assert row["candidate_policy"] is None
        assert row["training_seconds"] >= 0
        assert row["inference_seconds"] >= 0


def test_runner_selects_exactly_one_lowest_validation_metric_per_model(tmp_path):
    config = tiny_config()
    config["models"] = {
        "als": {
            "candidates": [
                {"n_factors": 1, "reg_lambda": 0.1, "n_iterations": 1},
                {"n_factors": 2, "reg_lambda": 1.0, "n_iterations": 3},
            ]
        }
    }
    payload = run_validation_selection(tiny_bundle(tmp_path), config)
    rows = payload["results"]
    selected = [row for row in rows if row["selected"]]

    assert len(selected) == 1
    assert selected[0]["rmse"] == min(row["rmse"] for row in rows)
    assert payload["best_configs"]["als"]["validation_metric"] == selected[0]["rmse"]


def test_test_loader_requires_frozen_matching_configuration(tmp_path):
    data_dir = tiny_bundle(tmp_path)
    payload = run_validation_selection(data_dir, tiny_config())
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="frozen"):
        load_test_after_freeze(data_dir, selection_path)

    frozen_path = tmp_path / "frozen.json"
    frozen = freeze_validation_configs(payload, frozen_path)
    loaded = load_test_after_freeze(data_dir, frozen_path)

    assert frozen["status"] == "frozen"
    assert len(loaded) == 3


def test_frozen_rating_test_uses_selected_subset_and_refuses_overwrite(tmp_path):
    data_dir = tiny_bundle(tmp_path)
    payload = run_validation_selection(data_dir, tiny_config())
    frozen_path = tmp_path / "frozen.json"
    frozen = freeze_validation_configs(
        payload,
        frozen_path,
        model_names={"als", "nmf", "bias_aware_als"},
    )
    output_path = tmp_path / "test-results.json"

    result = run_frozen_rating_test(data_dir, frozen_path, output_path=output_path)

    assert set(frozen["best_configs"]) == {"als", "nmf", "bias_aware_als"}
    assert result["phase"] == "frozen_rating_test"
    assert result["test_accessed"] is True
    assert {row["model"] for row in result["results"]} == {
        "als",
        "nmf",
        "bias_aware_als",
    }
    assert all(row["evaluation_split"] == "test" for row in result["results"])
    assert all(row["evaluated_rating_count"] == 3 for row in result["results"])
    with pytest.raises(FileExistsError, match="already exists"):
        run_frozen_rating_test(data_dir, frozen_path, output_path=output_path)


def test_frozen_unified_test_uses_shared_rating_and_ranking_protocol(tmp_path):
    data_dir = tiny_bundle(tmp_path)
    selection = run_validation_selection(data_dir, tiny_config())
    frozen_path = tmp_path / "frozen.json"
    frozen = freeze_validation_configs(
        selection,
        frozen_path,
        model_names={"als", "nmf", "bias_aware_als"},
    )
    output_path = tmp_path / "unified-test-results.json"

    result = run_frozen_unified_test(
        data_dir,
        frozen_path,
        output_path=output_path,
    )

    assert result["phase"] == "frozen_unified_test"
    assert result["frozen_config_hash"] == frozen["config_hash"]
    assert result["selection_source"] == "validation_only"
    assert result["ranking_protocol"]["candidate_policy"] == (
        "full_train_catalog_excluding_seen"
    )
    assert {row["model"] for row in result["results"]} == {
        "als",
        "nmf",
        "bias_aware_als",
    }
    for row in result["results"]:
        assert set(RESULT_FIELDS).issubset(row)
        assert row["evaluated_rating_count"] == 3
        assert row["evaluated_rating_users"] == 3
        assert row["evaluated_ranking_users"] == 1
        assert row["catalog_size"] == 3
        assert row["candidate_policy"] == "full_train_catalog_excluding_seen"
        assert row["precision_at_5"] >= 0
        assert row["recall_at_10"] >= 0
        assert row["ndcg_at_20"] >= 0
    with pytest.raises(FileExistsError, match="already exists"):
        run_frozen_unified_test(
            data_dir,
            frozen_path,
            output_path=output_path,
        )

    table_path = tmp_path / "summary.csv"
    chart_path = tmp_path / "summary.png"
    write_summary_table(result, table_path)
    write_summary_chart(result, chart_path)
    assert table_path.read_text(encoding="utf-8").startswith(
        "model,best_validation_metric,rmse,mae,"
    )
    assert chart_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_config_rejects_candidate_seed_and_unknown_model(tmp_path):
    config = tiny_config()
    config["models"]["als"]["candidates"][0]["seed"] = 99
    path = tmp_path / "bad-seed.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="seed"):
        load_experiment_config(path)

    config = tiny_config()
    config["models"]["mystery_model"] = {"candidates": [{}]}
    path = tmp_path / "bad-model.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported model"):
        load_experiment_config(path)
