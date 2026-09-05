"""Freeze Ricky's selected SVD++ config and run the shared test once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from goodbooks_mf.experiment import freeze_validation_configs, run_frozen_unified_test


ROOT = Path(__file__).parent
DEFAULT_DATA = ROOT / "data" / "processed" / "goodreads-poetry-v1"
DEFAULT_SELECTION = ROOT / "results" / "ricky_validation_selection.json"
DEFAULT_FROZEN = ROOT / "results" / "ricky_frozen_config.json"
DEFAULT_OUTPUT = ROOT / "results" / "ricky_unified_test_metrics.json"
CANONICAL_MANIFEST = ROOT / "canonical_manifest.json"


def run(
    data_dir: Path,
    selection_path: Path,
    frozen_config_path: Path,
    output_path: Path,
    *,
    expected_manifest_path: Path = CANONICAL_MANIFEST,
) -> dict:
    """Freeze SVD++ selection and perform its one permitted shared test run."""
    for path in (frozen_config_path, output_path):
        if path.exists():
            raise FileExistsError(f"refusing to replace existing artifact: {path}")

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    frozen = freeze_validation_configs(
        selection,
        frozen_config_path,
        model_names={"svdpp"},
    )
    payload = run_frozen_unified_test(
        data_dir,
        frozen_config_path,
        output_path=output_path,
        expected_manifest_path=expected_manifest_path,
    )
    return {"frozen": frozen, "test": payload}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the selected SVD++ config and evaluate test exactly once."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--frozen-config", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    result = run(
        args.data,
        args.selection,
        args.frozen_config,
        args.output,
    )
    payload = result["test"]
    print(
        json.dumps(
            {
                "dataset_version": payload["dataset_version"],
                "seed": payload["seed"],
                "frozen_config_hash": result["frozen"]["config_hash"],
                "models": [row["model"] for row in payload["results"]],
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
