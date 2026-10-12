"""Convert official KuaiRand logs into a compact, versioned training table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from kuairand_data.data import normalize_events, split_events


def prepare(raw_dir: Path, output_dir: Path) -> Path:
    files = sorted(raw_dir.glob("log_*.csv"))
    if not files:
        raise FileNotFoundError(f"no log_*.csv files found in {raw_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    events = pd.concat(
        [split_events(normalize_events(pd.read_csv(path), source=path.name)) for path in files],
        ignore_index=True,
    ).sort_values("request_time_ms", kind="stable")
    events.to_parquet(output_dir / "events.parquet", index=False)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset": raw_dir.parent.name,
                "rows": len(events),
                "schema": list(events.columns),
                "split_counts": events["split"].value_counts().sort_index().to_dict(),
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare KuaiRand logs for two-tower training")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.raw_dir, args.output_dir))


if __name__ == "__main__":
    main()
