"""Freeze Yutao's selected configs and run their rating test exactly once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from goodbooks_mf.experiment import freeze_validation_configs, run_frozen_rating_test


ROOT = Path(__file__).parent
DEFAULT_DATA = ROOT / "data" / "processed" / "goodreads-poetry-v1"
DEFAULT_SELECTION = ROOT / "results" / "validation_selection.json"
DEFAULT_FROZEN = ROOT / "results" / "yutao_frozen_config.json"
DEFAULT_OUTPUT = ROOT / "results" / "yutao_test_metrics.json"
CANONICAL_MANIFEST = ROOT / "canonical_manifest.json"
YUTAO_MODELS = {"als", "nmf", "bias_aware_als"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the one-time frozen ALS/NMF/Bias-aware ALS rating test."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--frozen-config", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.frozen_config.exists():
        raise FileExistsError(
            f"frozen config already exists; refusing to replace it: {args.frozen_config}"
        )
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    frozen = freeze_validation_configs(
        selection,
        args.frozen_config,
        model_names=YUTAO_MODELS,
    )
    payload = run_frozen_rating_test(
        args.data,
        args.frozen_config,
        output_path=args.output,
        expected_manifest_path=CANONICAL_MANIFEST,
    )
    print(
        json.dumps(
            {
                "dataset_version": payload["dataset_version"],
                "seed": payload["seed"],
                "frozen_config_hash": frozen["config_hash"],
                "test_accessed": payload["test_accessed"],
                "results": [
                    {
                        "model": row["model"],
                        "rmse": row["rmse"],
                        "mae": row["mae"],
                        "evaluated_rating_count": row["evaluated_rating_count"],
                    }
                    for row in payload["results"]
                ],
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
