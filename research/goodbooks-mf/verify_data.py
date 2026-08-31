"""Verify that a local processed bundle is byte-identical to the team dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from goodbooks_mf.artifacts import verify_bundle


DEFAULT_DATA = Path(__file__).parent / "data" / "processed" / "goodreads-poetry-v1"
CANONICAL_MANIFEST = Path(__file__).parent / "canonical_manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path, nargs="?", default=DEFAULT_DATA)
    args = parser.parse_args()
    manifest = verify_bundle(args.data_dir, expected_manifest_path=CANONICAL_MANIFEST)
    print(json.dumps({"version": manifest["version"], "counts": manifest["counts"]}, indent=2))


if __name__ == "__main__":
    main()
