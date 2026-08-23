from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_value(value), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def validate_frontend_artifacts(metrics: dict, samples: dict, profile: dict) -> None:
    """Fail fast when public artifacts lose provenance or demo fields."""
    required = (
        "version", "generatedAtUtc", "experimentCodeVersion", "dataset", "seed",
        "split", "relevance", "candidatePolicy", "models",
    )
    missing = [key for key in required if key not in metrics]
    if missing:
        raise ValueError(f"metrics missing required field: {missing[0]}")
    if metrics.get("version") != "movielens-cf-v4":
        raise ValueError("metrics version must be movielens-cf-v4")
    if not metrics.get("postHocLimitation"):
        raise ValueError("metrics missing postHocLimitation")
    if not metrics.get("sensitivity", {}).get("ratingAtLeast5", {}).get("readOnly"):
        raise ValueError("metrics missing read-only ratingAtLeast5 sensitivity")
    split_counts = metrics.get("splitCounts", {})
    for key in ("trainRatings", "validationRatings", "fittedRatings", "testRatings"):
        if key not in split_counts:
            raise ValueError(f"splitCounts missing field: {key}")
    methods = [metrics.get("baselines", {}).get("bayesianPopularity", {}), *metrics.get("models", [])]
    for method in methods:
        hit_rate = method.get("test", {}).get("hit_rate_at_10")
        if hit_rate is None or not 0 <= hit_rate <= 1:
            raise ValueError("method missing valid hit_rate_at_10")
    for model in metrics.get("models", []):
        if model.get("algorithmVariant") not in ("mean_centered", "bias_aware"):
            raise ValueError("model missing algorithmVariant")
        if model.get("selection", {}).get("primaryMetric") != "ndcg_at_10":
            raise ValueError("model missing validation selection rule")
        tie_stats = model.get("system", {}).get("rankingTieStats")
        if not isinstance(tie_stats, dict) or "fullyTiedListShare" not in tie_stats:
            raise ValueError("model missing rankingTieStats")
        if model.get("key") == "item-cf" and "top10FiveStarOnlyEvidenceShare" not in model.get(
            "system", {}
        ):
            raise ValueError("item-cf missing top10FiveStarOnlyEvidenceShare")
    bayesian = metrics.get("baselines", {}).get("bayesianPopularity", {})
    if len(bayesian.get("examples", [])) != 2:
        raise ValueError("Bayesian popularity requires two examples")
    field_names = {field.get("name") for field in profile.get("fields", [])}
    if field_names != {"user_id", "movie_id", "rating", "timestamp", "title", "genres"}:
        raise ValueError("profile fields must describe both MovieLens input tables")
    if samples.get("version") != "movielens-samples-v4":
        raise ValueError("samples version must be movielens-samples-v4")
    for user in samples.get("users", []):
        for count_field in ("historyTotal", "relevantTestTotal"):
            if count_field not in user:
                raise ValueError(f"sample missing {count_field}")
        methods = user.get("methods", {})
        for method in ("popularity", "userCf", "itemCf"):
            if method not in methods:
                raise ValueError(f"sample missing method: {method}")
            for item in methods[method]:
                if not {"movieId", "rankScore", "hit"}.issubset(item):
                    raise ValueError(f"{method} recommendation missing ranking fields")
                if method != "popularity" and "similarityWeight" not in item:
                    raise ValueError(f"{method} recommendation missing confidence weight")
        if "relevantTest" not in user:
            raise ValueError("sample missing relevantTest")
