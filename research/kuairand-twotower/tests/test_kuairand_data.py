import pandas as pd

from kuairand_twotower.data import build_histories, normalize_events, split_events
from prepare_kuairand_data import prepare


def test_normalize_events_preserves_official_unix_millisecond_timestamps():
    raw = pd.DataFrame(
        {
            "user_id": [1, 1],
            "video_id": [10, 11],
            "date": [20220415, 20220415],
            "time_ms": [1649980801000, 1649980802000],
            "tab": [1, 2],
            "long_view": [1, 0],
            "is_rand": [0, 1],
        }
    )

    events = normalize_events(raw, source="log_standard.csv")

    assert events["request_id"].is_unique
    assert events["request_time_ms"].tolist() == [1649980801000, 1649980802000]
    assert events["is_random"].tolist() == [False, True]


def test_random_exposure_events_are_excluded_from_training_split():
    events = pd.DataFrame(
        {
            "request_time_ms": [1649376000000, 1649980800000, 1651190400000, 1651536000000],
            "is_random": [False, False, False, True],
        }
    )

    assert split_events(events)["split"].tolist() == ["warmup", "train", "validation", "random"]


def test_prepare_writes_parquet_events_and_manifest(tmp_path):
    raw_dir = tmp_path / "KuaiRand-Pure" / "data"
    raw_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "user_id": [1], "video_id": [10], "date": [20220415], "time_ms": [1649980801000],
            "tab": [1], "long_view": [1], "is_rand": [0],
        }
    ).to_csv(raw_dir / "log_standard_4_22_to_5_08_pure.csv", index=False)

    manifest = prepare(raw_dir, tmp_path / "dataset")

    assert manifest.exists()
    assert (manifest.parent / "events.parquet").exists()


def test_build_histories_uses_only_strictly_prior_positive_nonrandom_events():
    events = pd.DataFrame(
        {
            "request_id": ["one", "two", "three", "four", "five"],
            "user_id": [7, 7, 7, 7, 7],
            "item_id": [10, 11, 12, 13, 14],
            "request_time_ms": [1, 2, 2, 3, 3],
            "long_view": [True, False, True, True, True],
            "is_random": [False, False, False, True, False],
        }
    )

    histories = build_histories(events, limit=2)

    assert histories["request_id"].tolist() == ["one", "three", "five"]
    assert histories["history_item_ids"].tolist() == [[], [10], [10, 12]]
