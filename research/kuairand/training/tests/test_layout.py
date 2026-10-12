"""Exercise relocated entry points without a dataset, GPU or remote writes."""

import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path


def test_default_cli_config_from_another_directory(tmp_path):
    project = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(project / "run.py"), "check-config"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    config = json.loads(result.stdout)["config"]
    module, factory = config["model"]["item_factory"].split(":")
    assert callable(getattr(importlib.import_module(module), factory))
    from run import ROOT

    assert ROOT == project.parents[2]
    assert (ROOT / "research/kuairand/data_pipeline/prepare_kuairand_data.py").is_file()


def test_frozen_upstream_snapshot_preserves_published_checksums():
    import recommender

    vendor = Path(recommender.__file__).parent / "vendor/ziqi"
    provenance = json.loads((vendor / "provenance.json").read_text())
    for original, expected in provenance["sha256"].items():
        assert hashlib.sha256((vendor / Path(original).name).read_bytes()).hexdigest() == expected
