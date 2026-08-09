from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd

MODULE_PATH = Path(__file__).parents[1] / "run_experiment.py"
SPEC = spec_from_file_location("spotify_experiment", MODULE_PATH)
experiment = module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(experiment)


def fixture_tracks() -> pd.DataFrame:
    rows = []
    for index in range(30):
        row = {
            "id": str(index if index < 29 else 0),
            "name": f"Track {index}",
            "artists": f"['Artist {index % 4}']",
            "year": 1990 + index,
            "popularity": index,
        }
        row.update({feature: (index + offset) / 40 for offset, feature in enumerate(experiment.FEATURES)})
        rows.append(row)
    return pd.DataFrame(rows)


def test_prepare_deduplicates_and_parses_artist():
    tracks = experiment.prepare_tracks(fixture_tracks())
    assert len(tracks) == 29
    assert tracks.iloc[0]["artist"] == "Artist 0"
    assert tracks[experiment.FEATURES].notna().all().all()


def test_exact_search_excludes_seed_and_is_deterministic():
    matrix = np.eye(12, dtype=np.float32)
    first, scores = experiment.exact_top_k(matrix, 2, k=4)
    second, _ = experiment.exact_top_k(matrix, 2, k=4)
    assert 2 not in first
    assert len(first) == 4
    assert np.array_equal(first, second)
    assert np.all((scores >= 0) & (scores <= 1))


def test_stratified_demo_respects_requested_size():
    tracks = experiment.prepare_tracks(fixture_tracks())
    demo = experiment.stratified_demo(tracks, 12)
    assert len(demo) == 12
    assert demo["id"].is_unique
