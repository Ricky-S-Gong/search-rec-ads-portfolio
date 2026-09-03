"""Run shared final evaluation for Yutao's frozen ALS/NMF configurations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from goodbooks_mf.experiment import run_frozen_unified_test


ROOT = Path(__file__).parent
DEFAULT_DATA = ROOT / "data" / "processed" / "goodreads-poetry-v1"
DEFAULT_FROZEN = ROOT / "results" / "yutao_frozen_config.json"
DEFAULT_OUTPUT = ROOT / "results" / "yutao_unified_test_metrics.json"
DEFAULT_TABLE = ROOT / "results" / "yutao_unified_test_metrics.csv"
DEFAULT_CHART = ROOT / "results" / "yutao_unified_test_metrics.png"
CANONICAL_MANIFEST = ROOT / "canonical_manifest.json"


def write_summary_table(payload: dict, output_path: Path) -> None:
    """Write one compact, analysis-ready row per evaluated model."""
    fields = [
        "model",
        "best_validation_metric",
        "rmse",
        "mae",
        "precision_at_5",
        "precision_at_10",
        "precision_at_20",
        "recall_at_5",
        "recall_at_10",
        "recall_at_20",
        "ndcg_at_5",
        "ndcg_at_10",
        "ndcg_at_20",
        "evaluated_rating_count",
        "evaluated_rating_users",
        "evaluated_ranking_users",
        "training_seconds",
        "inference_seconds",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in payload["results"]:
            writer.writerow({field: row[field] for field in fields})


def write_summary_chart(payload: dict, output_path: Path) -> None:
    """Plot rating error, Top-10 ranking quality, and training duration."""
    rows = payload["results"]
    labels = [
        {
            "als": "ALS",
            "nmf": "NMF + L2",
            "bias_aware_als": "Bias-aware ALS",
        }.get(row["model"], row["model"])
        for row in rows
    ]
    positions = list(range(len(rows)))
    width = 0.34
    colors = ("#2563eb", "#f59e0b", "#10b981")

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
    axes[0].bar(
        [position - width / 2 for position in positions],
        [row["rmse"] for row in rows],
        width,
        label="RMSE",
        color=colors[0],
    )
    axes[0].bar(
        [position + width / 2 for position in positions],
        [row["mae"] for row in rows],
        width,
        label="MAE",
        color=colors[1],
    )
    axes[0].set_title("Explicit-rating error (lower is better)")
    axes[0].legend(frameon=False)

    metric_names = ("precision_at_10", "recall_at_10", "ndcg_at_10")
    metric_labels = ("Precision@10", "Recall@10", "NDCG@10")
    for index, (metric, label, color) in enumerate(
        zip(metric_names, metric_labels, colors)
    ):
        offsets = [position + (index - 1) * 0.24 for position in positions]
        axes[1].bar(offsets, [row[metric] for row in rows], 0.22, label=label, color=color)
    axes[1].set_title("Top-10 ranking quality (higher is better)")
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].bar(positions, [row["training_seconds"] for row in rows], color=colors[2])
    axes[2].set_title("Training time (seconds)")

    for axis in axes:
        axis.set_xticks(positions, labels, rotation=18, ha="right")
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Goodreads Poetry: frozen shared-evaluation results")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, format=output_path.suffix.lstrip("."), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen ALS/NMF models with the shared protocol."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--frozen-config", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--chart", type=Path, default=DEFAULT_CHART)
    args = parser.parse_args()

    payload = run_frozen_unified_test(
        args.data,
        args.frozen_config,
        output_path=args.output,
        expected_manifest_path=CANONICAL_MANIFEST,
    )
    write_summary_table(payload, args.table)
    write_summary_chart(payload, args.chart)
    print(
        json.dumps(
            {
                "dataset_version": payload["dataset_version"],
                "seed": payload["seed"],
                "frozen_config_hash": payload["frozen_config_hash"],
                "models": [row["model"] for row in payload["results"]],
                "output": str(args.output),
                "table": str(args.table),
                "chart": str(args.chart),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
