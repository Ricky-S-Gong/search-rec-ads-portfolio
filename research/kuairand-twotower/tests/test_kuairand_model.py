import pandas as pd
import pytest

torch = pytest.importorskip("torch", reason="PyTorch is unavailable in this Intel macOS development environment")

from kuairand_twotower.dataset import build_dataloader
from kuairand_twotower.item_tower import ItemTower


def test_item_tower_returns_normalized_vectors_for_targets_and_histories():
    tower = ItemTower(categorical_cardinalities=[4] * 8, numeric_feature_count=4)
    target = tower(torch.tensor([[1] * 8, [2] * 8]), torch.zeros(2, 4))
    history = tower(torch.tensor([[[1] * 8, [0] * 8]]), torch.zeros(1, 2, 4))

    assert target.shape == (2, 64)
    assert history.shape == (1, 2, 64)
    assert torch.isfinite(history).all()
    assert torch.allclose(target.norm(dim=-1), torch.ones(2), atol=1e-5)


def test_batch_loader_filters_training_positives_and_pads_histories(tmp_path):
    events = tmp_path / "events.parquet"
    histories = tmp_path / "user_histories.parquet"
    item_features = tmp_path / "item_features.parquet"
    pd.DataFrame(
        {
            "request_id": ["train-empty", "train-history", "validation", "test", "random", "not-positive"],
            "user_id": [1] * 6,
            "item_id": [1, 2, 1, 2, 1, 2],
            "long_view": [True, True, True, True, True, False],
            "is_random": [False, False, False, False, True, False],
            "split": ["train", "train", "validation", "test", "random", "train"],
        }
    ).to_parquet(events, index=False)
    pd.DataFrame(
        {
            "request_id": ["train-empty", "train-history", "validation", "test", "random"],
            "user_id": [1] * 5,
            "history_item_ids": [[], [1], [2], [1], [2]],
        }
    ).to_parquet(histories, index=False)
    pd.DataFrame(
        {
            "item_id": [1, 2],
            "author_id": [10, 20],
            "video_type": ["NORMAL", "AD"],
            "upload_dt": ["2022-04-01", "2022-04-02"],
            "upload_type": ["A", "B"],
            "visible_status": [0, 1],
            "video_duration": [10.0, 20.0],
            "server_width": [720.0, 1080.0],
            "server_height": [1280.0, 1920.0],
            "music_id": [100, 200],
            "music_type": [1, 2],
            "tag": [3, 4],
        }
    ).to_parquet(item_features, index=False)

    loader, schema = build_dataloader(events, histories, item_features, split="train", batch_size=2, max_history_length=3)
    batch = next(iter(loader))

    assert len(loader.dataset) == 2
    assert schema.categorical_cardinalities[0] == 3
    assert batch["target_categorical"].shape == (2, 8)
    assert batch["target_numeric"].shape == (2, 4)
    assert batch["history_categorical"].shape == (2, 3, 8)
    assert batch["history_numeric"].shape == (2, 3, 4)
    assert batch["history_mask"].tolist() == [[False, False, False], [False, False, True]]


def test_batch_loader_rejects_positive_events_missing_histories(tmp_path):
    events = tmp_path / "events.parquet"
    histories = tmp_path / "user_histories.parquet"
    item_features = tmp_path / "item_features.parquet"
    pd.DataFrame(
        {"request_id": ["train"], "item_id": [1], "long_view": [True], "is_random": [False], "split": ["train"]}
    ).to_parquet(events, index=False)
    pd.DataFrame({"request_id": [], "history_item_ids": []}).to_parquet(histories, index=False)
    pd.DataFrame(
        {
            "item_id": [1], "author_id": [1], "video_type": ["NORMAL"], "upload_dt": ["2022-04-01"],
            "upload_type": ["A"], "visible_status": [0], "video_duration": [1.0], "server_width": [1.0],
            "server_height": [1.0], "music_id": [1], "music_type": [1], "tag": [1],
        }
    ).to_parquet(item_features, index=False)

    with pytest.raises(ValueError, match="missing histories"):
        build_dataloader(events, histories, item_features, split="train", batch_size=1)


def test_batch_loader_rejects_referenced_items_missing_features(tmp_path):
    events = tmp_path / "events.parquet"
    histories = tmp_path / "user_histories.parquet"
    item_features = tmp_path / "item_features.parquet"
    pd.DataFrame(
        {"request_id": ["train"], "item_id": [2], "long_view": [True], "is_random": [False], "split": ["train"]}
    ).to_parquet(events, index=False)
    pd.DataFrame({"request_id": ["train"], "history_item_ids": [[]]}).to_parquet(histories, index=False)
    pd.DataFrame(
        {
            "item_id": [1], "author_id": [1], "video_type": ["NORMAL"], "upload_dt": ["2022-04-01"],
            "upload_type": ["A"], "visible_status": [0], "video_duration": [1.0], "server_width": [1.0],
            "server_height": [1.0], "music_id": [1], "music_type": [1], "tag": [1],
        }
    ).to_parquet(item_features, index=False)

    with pytest.raises(ValueError, match="missing 1 referenced item IDs"):
        build_dataloader(events, histories, item_features, split="train", batch_size=1)
