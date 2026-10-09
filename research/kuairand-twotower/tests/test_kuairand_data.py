import pandas as pd

from kuairand_twotower.data import normalize_events, split_events


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
