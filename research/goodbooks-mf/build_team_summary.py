"""Combine frozen GoodBooks model results into one validated team report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
DEFAULT_ZIQI = RESULTS / "ziqi_unified_test_metrics.json"
DEFAULT_YUTAO = RESULTS / "yutao_unified_test_metrics.json"
DEFAULT_VALIDATION = RESULTS / "validation_selection.json"
DEFAULT_OUTPUT = RESULTS / "team_model_comparison.json"
DEFAULT_TABLE = RESULTS / "team_model_comparison.csv"
DEFAULT_CHART = RESULTS / "team_model_comparison.png"

MODEL_ORDER = ("basic_mf", "funksvd", "als", "nmf", "bias_aware_als")
MODEL_META = {
    "basic_mf": ("Basic MF", "Ziqi", "planned"),
    "funksvd": ("FunkSVD", "Ziqi", "planned"),
    "als": ("ALS", "Yutao", "planned"),
    "nmf": ("NMF + L2", "Yutao", "planned"),
    "bias_aware_als": ("Bias-aware ALS", "Yutao", "additional diagnostic"),
}
METRIC_FIELDS = (
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
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def combine_results(
    ziqi: dict[str, Any],
    yutao: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Any]:
    """Validate shared protocol metadata and return normalized model rows."""
    for key in ("dataset_version", "seed", "data_counts"):
        if ziqi.get(key) != yutao.get(key):
            raise ValueError(f"team result mismatch for {key}")
    if ziqi.get("ranking_protocol") != yutao.get("ranking_protocol"):
        raise ValueError("team result mismatch for ranking_protocol")
    if validation.get("dataset_version") != ziqi["dataset_version"]:
        raise ValueError("validation dataset_version does not match test results")
    if validation.get("seed") != ziqi["seed"]:
        raise ValueError("validation seed does not match test results")

    source_rows = dict(ziqi["models"])
    source_rows.update({row["model"]: row for row in yutao["results"]})
    missing_models = set(MODEL_ORDER).difference(source_rows)
    if missing_models:
        raise ValueError(f"missing model results: {', '.join(sorted(missing_models))}")

    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        source = source_rows[model]
        missing_fields = set(METRIC_FIELDS).difference(source)
        if missing_fields:
            raise ValueError(
                f"incomplete result for {model}: {', '.join(sorted(missing_fields))}"
            )
        label, owner, role = MODEL_META[model]
        validation_entry = validation["best_configs"].get(model)
        row = {
            "model": model,
            "model_label": label,
            "owner": owner,
            "model_role": role,
            "best_validation_rmse": (
                validation_entry["validation_metric"] if validation_entry else None
            ),
            **{field: source[field] for field in METRIC_FIELDS},
            "hyperparameters": source["hyperparameters"],
        }
        rows.append(row)

    shared_rating_counts = {row["evaluated_rating_count"] for row in rows}
    shared_rating_users = {row["evaluated_rating_users"] for row in rows}
    shared_ranking_users = {row["evaluated_ranking_users"] for row in rows}
    if any(
        len(values) != 1
        for values in (shared_rating_counts, shared_rating_users, shared_ranking_users)
    ):
        raise ValueError("models were not evaluated on identical population counts")

    return {
        "schema_version": "goodbooks-team-comparison-v1",
        "dataset_version": ziqi["dataset_version"],
        "seed": ziqi["seed"],
        "data_counts": ziqi["data_counts"],
        "ranking_protocol": ziqi["ranking_protocol"],
        "included_models": list(MODEL_ORDER),
        "pending_models": ["svdpp"],
        "status": "awaiting Ricky SVD++ result",
        "results": rows,
    }


def write_table(payload: dict[str, Any], output_path: Path) -> None:
    fields = (
        "model",
        "model_label",
        "owner",
        "model_role",
        "best_validation_rmse",
        *METRIC_FIELDS,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in payload["results"]:
            writer.writerow({field: row[field] for field in fields})


def write_chart(payload: dict[str, Any], output_path: Path) -> None:
    rows = payload["results"]
    labels = [row["model_label"] for row in rows]
    positions = list(range(len(rows)))
    width = 0.34
    colors = ("#2563eb", "#f59e0b", "#10b981")

    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
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

    for index, (metric, label, color) in enumerate(
        zip(
            ("precision_at_10", "recall_at_10", "ndcg_at_10"),
            ("Precision@10", "Recall@10", "NDCG@10"),
            colors,
        )
    ):
        offsets = [position + (index - 1) * 0.24 for position in positions]
        axes[1].bar(
            offsets,
            [row[metric] for row in rows],
            0.22,
            label=label,
            color=color,
        )
    axes[1].set_title("Top-10 ranking quality (higher is better)")
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].bar(
        positions,
        [row["training_seconds"] for row in rows],
        color=colors[2],
    )
    axes[2].set_title("Training time on one machine (seconds)")

    for axis in axes:
        axis.set_xticks(positions, labels, rotation=22, ha="right")
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Goodreads Poetry: shared-evaluation model comparison")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, format=output_path.suffix.lstrip("."), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the shared GoodBooks model report.")
    parser.add_argument("--ziqi", type=Path, default=DEFAULT_ZIQI)
    parser.add_argument("--yutao", type=Path, default=DEFAULT_YUTAO)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--chart", type=Path, default=DEFAULT_CHART)
    args = parser.parse_args()

    payload = combine_results(_load(args.ziqi), _load(args.yutao), _load(args.validation))
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_table(payload, args.table)
    write_chart(payload, args.chart)
    print(
        json.dumps(
            {
                "included_models": payload["included_models"],
                "pending_models": payload["pending_models"],
                "output": str(args.output),
                "table": str(args.table),
                "chart": str(args.chart),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
