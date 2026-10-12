import json
from pathlib import Path

import pandas as pd
import pytest

from goodbooks_mf.artifacts import write_bundle
from goodbooks_mf.frontend_artifacts import publish_frontend_artifact


def _bundle(tmp_path: Path) -> Path:
    interactions = pd.DataFrame(
        {
            "user_idx": pd.Series([0, 0, 1, 1, 1, 0], dtype="int32"),
            "item_idx": pd.Series([0, 1, 0, 1, 2, 2], dtype="int32"),
            "rating": pd.Series([5, 0, 4, 3, 0, 5], dtype="int64"),
            "is_read": [True, True, True, True, False, True],
            "is_reviewed": [False, False, True, False, False, False],
            "event_time": pd.date_range("2017-01-01", periods=6, tz="UTC"),
            "split": ["train", "train", "train", "validation", "test", "test"],
        }
    )
    users = pd.DataFrame({"user_id": ["u0", "u1"], "user_idx": [0, 1]})
    items = pd.DataFrame({"item_id": ["b0", "b1", "b2"], "item_idx": [0, 1, 2]})
    return_path = tmp_path / "bundle"
    write_bundle(return_path, interactions, users, items, seed=7, config={"version": "tiny-v1"})
    return return_path


def _summary() -> dict:
    rows = []
    for index, model in enumerate(
        ("basic_mf", "funksvd", "als", "nmf", "svdpp", "bias_aware_als")
    ):
        rows.append(
            {
                "model": model,
                "model_label": model,
                "owner": "team",
                "model_role": "additional diagnostic" if model == "bias_aware_als" else "planned",
                "best_validation_rmse": 1.0,
                "rmse": 1.0,
                "mae": 0.8,
                "precision_at_5": 0.1,
                "precision_at_10": 0.1,
                "precision_at_20": 0.1,
                "recall_at_5": 0.1,
                "recall_at_10": 0.1,
                "recall_at_20": 0.1,
                "ndcg_at_5": 0.1,
                "ndcg_at_10": 0.1 + index / 100,
                "ndcg_at_20": 0.1,
                "evaluated_rating_count": 4,
                "evaluated_rating_users": 2,
                "evaluated_ranking_users": 2,
                "training_seconds": 1.0,
                "inference_seconds": 0.1,
                "hyperparameters": {"n_factors": 2},
            }
        )
    return {
        "schema_version": "goodbooks-team-comparison-v1",
        "status": "complete",
        "pending_models": [],
        "dataset_version": "tiny-v1",
        "seed": 7,
        "data_counts": {
            "interactions": 6,
            "users": 2,
            "items": 3,
            "train": 3,
            "validation": 1,
            "test": 2,
        },
        "ranking_protocol": {
            "candidate_policy": "full_train_catalog_excluding_seen",
            "relevance": "rating >= 4 OR (rating == 0 AND is_read)",
            "k_values": [5, 10, 20],
        },
        "included_models": [row["model"] for row in rows],
        "results": rows,
    }


def test_publisher_writes_aggregate_frontend_artifact_and_excludes_rows(tmp_path: Path):
    bundle = _bundle(tmp_path)
    summary_path = tmp_path / "team.json"
    summary_path.write_text(json.dumps(_summary()), encoding="utf-8")
    output_path = tmp_path / "metrics.json"

    artifact = publish_frontend_artifact(bundle, summary_path, output_path)

    assert artifact["schema_version"] == "goodbooks-frontend-artifact-v1"
    assert artifact["champion"]["model"] == "svdpp"
    assert artifact["dataset"]["fields"][0]["name"] == "user_idx"
    assert artifact["dataset"]["matrices"]["explicit"]["shape"] == [2, 3]
    assert {row["model"] for row in artifact["models"]} == {
        "basic_mf", "funksvd", "als", "nmf", "svdpp", "bias_aware_als"
    }
    published = json.loads(output_path.read_text(encoding="utf-8"))
    assert {field["name"] for field in published["dataset"]["fields"]} == {
        "user_idx", "item_idx", "rating", "is_read", "is_reviewed", "event_time", "split"
    }


def test_publisher_rejects_incomplete_or_inconsistent_team_summary(tmp_path: Path):
    bundle = _bundle(tmp_path)
    summary = _summary()
    summary["pending_models"] = ["svdpp"]
    summary["status"] = "awaiting SVD++"
    summary_path = tmp_path / "team.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="complete"):
        publish_frontend_artifact(bundle, summary_path, tmp_path / "metrics.json")


def test_publisher_rejects_inconsistent_or_private_summary_fields(tmp_path: Path):
    bundle = _bundle(tmp_path)
    cases = [
        (lambda summary: summary.update(dataset_version="wrong-v1"), "dataset_version"),
        (lambda summary: summary.update(seed=8), "seed"),
        (lambda summary: summary["ranking_protocol"].update(k_values=[10]), "ranking_protocol"),
        (lambda summary: summary["results"][0].update(evaluated_ranking_users=1), "evaluation population"),
        (lambda summary: summary["results"][5].update(model_role="planned"), "model roles"),
        (lambda summary: summary["results"][0].update(rmse=float("nan")), "finite"),
        (lambda summary: summary.update(candidate_list=[1, 2]), "private field"),
    ]
    for index, (mutation, message) in enumerate(cases):
        summary = _summary()
        mutation(summary)
        summary_path = tmp_path / f"team-{index}.json"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            publish_frontend_artifact(bundle, summary_path, tmp_path / "metrics.json")
