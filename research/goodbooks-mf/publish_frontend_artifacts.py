"""Publish the verified aggregate artifact consumed by the GoodBooks case study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from goodbooks_mf.frontend_artifacts import publish_frontend_artifact


ROOT = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish aggregate GoodBooks frontend facts.")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "goodreads-poetry-v1")
    parser.add_argument("--summary", type=Path, default=ROOT / "results" / "team_model_comparison.json")
    parser.add_argument("--output", type=Path, default=ROOT.parents[1] / "public" / "artifacts" / "goodbooks" / "metrics.json")
    parser.add_argument("--canonical-manifest", type=Path, default=ROOT / "canonical_manifest.json")
    args = parser.parse_args()
    artifact = publish_frontend_artifact(
        args.data,
        args.summary,
        args.output,
        expected_manifest_path=args.canonical_manifest,
    )
    print(json.dumps({"output": str(args.output), "champion": artifact["champion"]}, indent=2))


if __name__ == "__main__":
    main()
