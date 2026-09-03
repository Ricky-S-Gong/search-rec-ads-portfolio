import pandas as pd

from goodbooks_mf.artifacts import write_bundle
from run_experiment import run


def test_experiment_result_records_data_counts_hyperparameters_and_training_time(tmp_path):
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

    payload = run(data_dir, tmp_path / "metrics.json", smoke=True)

    assert payload["data_counts"]["interactions"] == 9
    assert payload["ranking_protocol"] == {
        "candidate_policy": "full_train_catalog_excluding_seen",
        "relevance": "rating >= 4 OR (rating == 0 AND is_read)",
        "k_values": [5, 10, 20],
    }
    for result in payload["models"].values():
        assert result["training_seconds"] >= 0
        assert result["hyperparameters"]["n_factors"] == 8
        assert result["catalog_size"] == 3
        assert "precision_at_5" in result
        assert "recall_at_10" in result
        assert "ndcg_at_20" in result
