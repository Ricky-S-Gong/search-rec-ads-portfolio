"""Publish aggregate-only GoodBooks facts for the static portfolio."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from scipy import sparse

from .artifacts import verify_bundle


MODEL_ORDER = (
    "basic_mf",
    "funksvd",
    "als",
    "nmf",
    "svdpp",
    "bias_aware_als",
)
PLANNED_MODELS = MODEL_ORDER[:-1]
MODEL_ROLES = (*(("planned",) * len(PLANNED_MODELS)), "additional diagnostic")
RANKING_PROTOCOL = {
    "candidate_policy": "full_train_catalog_excluding_seen",
    "relevance": "rating >= 4 OR (rating == 0 AND is_read)",
    "k_values": [5, 10, 20],
}
PRIVATE_FIELD_NAMES = {
    "user_mapping",
    "item_mapping",
    "candidate_list",
    "candidates",
    "row_level_interactions",
    "records",
    "local_path",
    "file_path",
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


def _load_summary(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _reject_private_fields(value: Any) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).lower() in PRIVATE_FIELD_NAMES:
                raise ValueError(f"private field is not allowed in team summary: {key}")
            _reject_private_fields(nested)
    elif isinstance(value, list):
        for nested in value:
            _reject_private_fields(nested)


def _require_complete_summary(summary: dict[str, Any], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    _reject_private_fields(summary)
    if summary.get("status") != "complete" or summary.get("pending_models"):
        raise ValueError("a complete team summary without pending models is required")
    for key, expected in (
        ("dataset_version", manifest["version"]),
        ("seed", manifest["seed"]),
        ("data_counts", manifest["counts"]),
    ):
        if summary.get(key) != expected:
            raise ValueError(f"team summary {key} does not match the verified bundle")
    if summary.get("ranking_protocol") != RANKING_PROTOCOL:
        raise ValueError("team summary ranking_protocol does not match the shared evaluation")

    rows = summary.get("results", [])
    if tuple(row.get("model") for row in rows) != MODEL_ORDER:
        raise ValueError("team summary must contain the six canonical model rows in order")
    if tuple(summary.get("included_models", [])) != MODEL_ORDER:
        raise ValueError("team summary included_models does not match the canonical model order")
    if tuple(row.get("model_role") for row in rows) != MODEL_ROLES:
        raise ValueError("team summary model roles do not match the planned and diagnostic models")
    for row in rows:
        missing = set(METRIC_FIELDS).difference(row)
        if missing:
            raise ValueError(f"incomplete metrics for {row['model']}: {sorted(missing)}")
        for field in METRIC_FIELDS + ("best_validation_rmse",):
            value = row[field]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"team summary metric must be finite and non-negative: {row['model']}.{field}")
    populations = {
        tuple(row[field] for field in (
            "evaluated_rating_count",
            "evaluated_rating_users",
            "evaluated_ranking_users",
        ))
        for row in rows
    }
    if len(populations) != 1:
        raise ValueError("all model rows must use the same evaluation population")
    return rows


def _field_profile(interactions: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {"name": "user_idx", "type": str(interactions["user_idx"].dtype), "range": [int(interactions["user_idx"].min()), int(interactions["user_idx"].max())]},
        {"name": "item_idx", "type": str(interactions["item_idx"].dtype), "range": [int(interactions["item_idx"].min()), int(interactions["item_idx"].max())]},
        {"name": "rating", "type": str(interactions["rating"].dtype), "range": [int(interactions["rating"].min()), int(interactions["rating"].max())]},
        {"name": "is_read", "type": str(interactions["is_read"].dtype), "values": [False, True]},
        {"name": "is_reviewed", "type": str(interactions["is_reviewed"].dtype), "values": [False, True]},
        {"name": "event_time", "type": str(interactions["event_time"].dtype), "range": [interactions["event_time"].min().isoformat(), interactions["event_time"].max().isoformat()]},
        {"name": "split", "type": str(interactions["split"].dtype), "values": ["train", "validation", "test"]},
    ]


def _matrix_profile(matrix: sparse.csr_matrix) -> dict[str, Any]:
    return {
        "shape": [int(value) for value in matrix.shape],
        "nonzero": int(matrix.nnz),
        "value_range": [float(matrix.data.min()), float(matrix.data.max())],
    }


def publish_frontend_artifact(
    data_dir: Path,
    team_summary_path: Path,
    output_path: Path,
    *,
    expected_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Write public, aggregate-only portfolio facts from verified project artifacts."""
    data_dir = Path(data_dir)
    manifest = verify_bundle(data_dir, expected_manifest_path)
    summary = _load_summary(team_summary_path)
    rows = _require_complete_summary(summary, manifest)
    interactions = pd.read_parquet(data_dir / "interactions.parquet")
    explicit = sparse.load_npz(data_dir / "train_explicit.npz").tocsr()
    implicit = sparse.load_npz(data_dir / "train_implicit.npz").tocsr()
    champion = max(
        (row for row in rows if row["model"] in PLANNED_MODELS),
        key=lambda row: (row["ndcg_at_10"], -PLANNED_MODELS.index(row["model"])),
    )
    explicit_counts = {
        split: int(((interactions["split"] == split) & interactions["rating"].gt(0)).sum())
        for split in ("train", "validation", "test")
    }
    artifact = {
        "schema_version": "goodbooks-frontend-artifact-v1",
        "dataset": {
            "version": manifest["version"],
            "seed": manifest["seed"],
            "counts": manifest["counts"],
            "fields": _field_profile(interactions),
            "explicit_counts": explicit_counts,
            "matrices": {
                "explicit": _matrix_profile(explicit),
                "implicit": _matrix_profile(implicit),
            },
            "source": "UCSD Goodreads Book Graph / Poetry",
            "license_note": "Academic use only; do not redistribute or use commercially.",
        },
        "evaluation": summary["ranking_protocol"],
        "models": [
            {
                key: row[key]
                for key in (
                    "model", "model_label", "owner", "model_role", "best_validation_rmse",
                    *METRIC_FIELDS, "hyperparameters",
                )
            }
            for row in rows
        ],
        "champion": {
            "model": champion["model"],
            "model_label": champion["model_label"],
            "selection_metric": "ndcg_at_10",
            "selection_value": champion["ndcg_at_10"],
        },
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return artifact
