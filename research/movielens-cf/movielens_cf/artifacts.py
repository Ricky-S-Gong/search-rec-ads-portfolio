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


def validate_frontend_artifacts(metrics: dict, samples: dict) -> None:
    """Fail fast when public artifacts lose provenance or demo fields."""
    required = (
        "version", "generatedAtUtc", "experimentCodeVersion", "dataset", "seed",
        "split", "relevance", "candidatePolicy", "models",
    )
    missing = [key for key in required if key not in metrics]
    if missing:
        raise ValueError(f"metrics missing required field: {missing[0]}")
    if samples.get("version") != "movielens-samples-v2":
        raise ValueError("samples version must be movielens-samples-v2")
    for user in samples.get("users", []):
        methods = user.get("methods", {})
        for method in ("popularity", "userCf", "itemCf"):
            if method not in methods:
                raise ValueError(f"sample missing method: {method}")
            for item in methods[method]:
                if not {"movieId", "rankScore", "hit"}.issubset(item):
                    raise ValueError(f"{method} recommendation missing ranking fields")
        if "relevantTest" not in user:
            raise ValueError("sample missing relevantTest")
