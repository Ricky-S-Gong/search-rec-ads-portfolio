import gzip
import json

import pandas as pd

from goodbooks_mf.data import (
    encode_ids,
    iterative_k_core,
    load_staged,
    normalize_interactions,
    stage_to_sqlite,
)


def test_normalize_interactions_cleans_invalid_rows_and_keeps_best_duplicate():
    rows = [
        {
            "user_id": "u1",
            "book_id": "b1",
            "rating": 4,
            "is_read": True,
            "is_reviewed": False,
            "date_updated": "Mon Jan 01 00:00:00 -0700 2017",
        },
        {
            "user_id": "u1",
            "book_id": "b1",
            "rating": 5,
            "is_read": True,
            "is_reviewed": True,
            "date_updated": "Tue Jan 02 00:00:00 -0700 2017",
        },
        {"user_id": "", "book_id": "b2", "rating": 4, "date_updated": "2017-01-01"},
        {"user_id": "u2", "book_id": "b2", "rating": 8, "date_updated": "2017-01-01"},
        {"user_id": "u2", "book_id": "b3", "rating": 3, "date_updated": ""},
    ]

    cleaned = normalize_interactions(rows)

    assert cleaned[["user_id", "item_id", "rating"]].to_dict("records") == [
        {"user_id": "u1", "item_id": "b1", "rating": 5}
    ]
    assert bool(cleaned.iloc[0]["is_reviewed"])
    assert str(cleaned["event_time"].dt.tz) == "UTC"


def test_iterative_k_core_repeats_until_user_and_item_constraints_are_stable():
    frame = pd.DataFrame(
        [
            ("u1", "a", 5),
            ("u1", "b", 4),
            ("u2", "a", 5),
            ("u2", "b", 0),
            ("u3", "b", 5),
            ("u3", "c", 4),
        ],
        columns=["user_id", "item_id", "rating"],
    )

    filtered = iterative_k_core(
        frame,
        min_user_interactions=2,
        min_user_ratings=1,
        min_item_interactions=2,
        min_item_ratings=1,
    )

    assert set(filtered["user_id"]) == {"u1", "u2"}
    assert set(filtered["item_id"]) == {"a", "b"}


def test_encode_ids_is_stable_and_uses_sorted_original_ids():
    frame = pd.DataFrame(
        [("z-user", "20", 5), ("a-user", "10", 4)],
        columns=["user_id", "item_id", "rating"],
    )

    encoded, users, items = encode_ids(frame)

    assert users.to_dict("records") == [
        {"user_id": "a-user", "user_idx": 0},
        {"user_id": "z-user", "user_idx": 1},
    ]
    assert items.to_dict("records") == [
        {"item_id": "10", "item_idx": 0},
        {"item_id": "20", "item_idx": 1},
    ]
    assert encoded[["user_idx", "item_idx"]].to_dict("records") == [
        {"user_idx": 1, "item_idx": 1},
        {"user_idx": 0, "item_idx": 0},
    ]


def test_staging_rebuild_does_not_retain_rows_from_an_older_source(tmp_path):
    raw_path = tmp_path / "source.json.gz"
    database_path = tmp_path / "staging.sqlite3"

    def write(records):
        with gzip.open(raw_path, "wt", encoding="utf-8") as output:
            for record in records:
                output.write(json.dumps(record) + "\n")

    shared = {
        "rating": 5,
        "is_read": True,
        "date_updated": "2017-01-01T00:00:00Z",
    }
    write([
        {**shared, "user_id": "u1", "book_id": "b1"},
        {**shared, "user_id": "u2", "book_id": "b2"},
    ])
    stage_to_sqlite(raw_path, database_path)
    write([{**shared, "user_id": "u1", "book_id": "b1"}])

    stage_to_sqlite(raw_path, database_path)

    assert load_staged(database_path)[["user_id", "item_id"]].to_dict("records") == [
        {"user_id": "u1", "item_id": "b1"}
    ]
