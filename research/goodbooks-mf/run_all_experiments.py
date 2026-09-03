"""Run phase-one train/validation model selection without reading test data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from goodbooks_mf.experiment import load_experiment_config, run_validation_selection


ROOT = Path(__file__).parent
DEFAULT_DATA = ROOT / "data" / "processed" / "goodreads-poetry-v1"
DEFAULT_CONFIG = ROOT / "experiment_config.json"
DEFAULT_OUTPUT = ROOT / "results" / "validation_selection.json"
CANONICAL_MANIFEST = ROOT / "canonical_manifest.json"


def _select_models(config: dict, requested: str | None, max_candidates: int | None) -> dict:
    if requested:
        names = {name.strip() for name in requested.split(",") if name.strip()}
        unknown = names.difference(config["models"])
        if unknown:
            raise ValueError(f"models not present in config: {', '.join(sorted(unknown))}")
        config["models"] = {
            name: specification
            for name, specification in config["models"].items()
            if name in names
        }
    if max_candidates is not None:
        if max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        for specification in config["models"].values():
            specification["candidates"] = specification["candidates"][:max_candidates]
    return config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select Goodreads MF configs using train/validation only."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--models",
        help="Optional comma-separated subset, for example: als,nmf,bias_aware_als",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        help="Optional per-model limit for a validation-only smoke run.",
    )
    args = parser.parse_args()

    config = _select_models(
        load_experiment_config(args.config),
        args.models,
        args.max_candidates,
    )
    payload = run_validation_selection(
        args.data,
        config,
        output_path=args.output,
        expected_manifest_path=CANONICAL_MANIFEST,
    )
    print(
        json.dumps(
            {
                "phase": payload["phase"],
                "dataset_version": payload["dataset_version"],
                "seed": payload["seed"],
                "test_accessed": payload["test_accessed"],
                "best_configs": payload["best_configs"],
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
