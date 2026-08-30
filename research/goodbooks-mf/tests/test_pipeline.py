import gzip
import json

from goodbooks_mf.artifacts import verify_bundle
from prepare_data import build


def test_build_creates_a_reproducible_bundle_from_streamed_json_gzip(tmp_path):
    raw_path = tmp_path / "interactions.json.gz"
    records = []
    for user in range(4):
        for position in range(4):
            item = (user + position) % 4
            records.append(
                {
                    "user_id": f"u{user}",
                    "book_id": f"b{item}",
                    "rating": (position % 5) + 1,
                    "is_read": True,
                    "is_reviewed": position % 2 == 0,
                    "date_updated": f"2017-01-{position + 1:02d}T00:00:00Z",
                }
            )
    with gzip.open(raw_path, "wt", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record) + "\n")
    config = {
        "version": "test-v1",
        "seed": 17,
        "min_user_interactions": 3,
        "min_user_ratings": 3,
        "min_item_interactions": 2,
        "min_item_ratings": 2,
        "max_users": None,
        "train_fraction": 0.5,
        "validation_fraction": 0.25,
    }
    output_dir = tmp_path / "processed" / "test-v1"

    manifest = build(raw_path, output_dir, config)

    assert manifest["version"] == "test-v1"
    assert manifest["counts"] == {
        "interactions": 16,
        "users": 4,
        "items": 4,
        "train": 8,
        "validation": 4,
        "test": 4,
    }
    assert verify_bundle(output_dir) == manifest
