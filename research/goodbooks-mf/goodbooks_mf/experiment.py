"""Validation-only experiment orchestration for the frozen Goodreads bundle.

This module keeps model selection separate from final test evaluation. The
validation runner never deserializes ``test.parquet``; final rating and ranking
evaluation can access it only through a frozen, checksum-protected
configuration artifact.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from .als import ALS
from .artifacts import verify_bundle
from .bias_aware_als import BiasAwareALS
from .evaluation import evaluate_ranking, evaluate_ratings, prepare_ranking_data, rmse
from .models import BasicMF, FunkSVD
from .nmf import NMF


SUPPORTED_MODELS = {
    "basic_mf",
    "funksvd",
    "als",
    "nmf",
    "bias_aware_als",
}

RESULT_FIELDS = (
    "dataset_version",
    "evaluation_split",
    "model",
    "seed",
    "n_factors",
    "learning_rate",
    "reg_lambda",
    "iterations_or_epochs",
    "selection_metric",
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
    "candidate_policy",
    "catalog_size",
    "training_seconds",
    "inference_seconds",
    "hyperparameters",
    "selected",
)


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_experiment_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a defensive copy after validating the phase-one config schema."""
    copied = json.loads(json.dumps(config))
    if copied.get("schema_version") != "goodbooks-experiment-config-v1":
        raise ValueError("unsupported experiment config schema_version")
    if not isinstance(copied.get("dataset_version"), str):
        raise ValueError("dataset_version is required")
    if not isinstance(copied.get("seed"), int):
        raise ValueError("one fixed integer seed is required")
    if copied.get("selection_metric") not in {"rmse", "mae"}:
        raise ValueError("selection_metric must be rmse or mae")
    rating_clip = copied.get("rating_clip")
    if (
        not isinstance(rating_clip, list)
        or len(rating_clip) != 2
        or not all(isinstance(value, (int, float)) for value in rating_clip)
        or rating_clip[0] >= rating_clip[1]
    ):
        raise ValueError("rating_clip must contain increasing numeric bounds")

    models = copied.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("at least one model configuration is required")
    enabled = 0
    for name, specification in models.items():
        if name not in SUPPORTED_MODELS:
            raise ValueError(f"unsupported model: {name}")
        if not isinstance(specification, dict):
            raise ValueError(f"model specification must be an object: {name}")
        if specification.get("enabled", True) is False:
            continue
        enabled += 1
        candidates = specification.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"model candidates must be a non-empty list: {name}")
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise ValueError(f"candidate must be an object: {name}")
            if "seed" in candidate:
                raise ValueError("candidate seed is forbidden; use the one fixed top-level seed")
    if enabled == 0:
        raise ValueError("at least one model must be enabled")
    return copied


def load_experiment_config(path: Path) -> dict[str, Any]:
    """Read and validate a JSON experiment configuration."""
    return validate_experiment_config(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def _mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def _build_model(
    name: str,
    parameters: Mapping[str, Any],
    *,
    n_users: int,
    n_items: int,
    seed: int,
):
    kwargs = dict(parameters)
    if name in {"basic_mf", "funksvd"}:
        kwargs.update(n_users=n_users, n_items=n_items, seed=seed)
        model_class = BasicMF if name == "basic_mf" else FunkSVD
        return model_class(**kwargs)
    kwargs["seed"] = seed
    if name == "als":
        return ALS(**kwargs)
    if name == "nmf":
        return NMF(**kwargs)
    if name == "bias_aware_als":
        return BiasAwareALS(**kwargs)
    raise ValueError(f"unsupported model: {name}")


def _iterations(parameters: Mapping[str, Any]) -> int | None:
    for name in ("n_epochs", "n_iterations", "max_iter"):
        if name in parameters:
            return int(parameters[name])
    return None


def _result_row(
    *,
    manifest: Mapping[str, Any],
    model_name: str,
    parameters: Mapping[str, Any],
    selection_metric: str,
    actual: np.ndarray,
    predicted: np.ndarray,
    training_seconds: float,
    inference_seconds: float,
    evaluation_split: str = "validation",
    best_validation_metric: float | None = None,
) -> dict[str, Any]:
    row = {
        "dataset_version": manifest["version"],
        "evaluation_split": evaluation_split,
        "model": model_name,
        "seed": int(manifest["seed"]),
        "n_factors": parameters.get("n_factors"),
        "learning_rate": parameters.get("learning_rate"),
        "reg_lambda": parameters.get("reg_lambda"),
        "iterations_or_epochs": _iterations(parameters),
        "selection_metric": selection_metric,
        "best_validation_metric": best_validation_metric,
        "rmse": rmse(actual, predicted),
        "mae": _mae(actual, predicted),
        "precision_at_5": None,
        "precision_at_10": None,
        "precision_at_20": None,
        "recall_at_5": None,
        "recall_at_10": None,
        "recall_at_20": None,
        "ndcg_at_5": None,
        "ndcg_at_10": None,
        "ndcg_at_20": None,
        "evaluated_rating_count": int(len(actual)),
        "evaluated_rating_users": None,
        "evaluated_ranking_users": None,
        "candidate_policy": None,
        "catalog_size": None,
        "training_seconds": float(training_seconds),
        "inference_seconds": float(inference_seconds),
        "hyperparameters": dict(parameters),
        "selected": False,
    }
    missing = set(RESULT_FIELDS).difference(row)
    if missing:
        raise AssertionError(f"incomplete result schema: {sorted(missing)}")
    return row


def run_validation_selection(
    data_dir: Path,
    config: Mapping[str, Any] | Path,
    *,
    output_path: Path | None = None,
    expected_manifest_path: Path | None = None,
    parquet_reader: Callable[[Path], pd.DataFrame] = pd.read_parquet,
) -> dict[str, Any]:
    """Train candidates on train and select only from validation metrics."""
    data_dir = Path(data_dir)
    validated = (
        load_experiment_config(config)
        if isinstance(config, Path)
        else validate_experiment_config(config)
    )
    manifest = verify_bundle(data_dir, expected_manifest_path)
    if manifest.get("version") != validated["dataset_version"]:
        raise ValueError("experiment config dataset_version does not match the bundle")
    if int(manifest.get("seed")) != validated["seed"]:
        raise ValueError("experiment config seed does not match the frozen bundle")

    # Deliberately do not read test.parquet in this phase.
    train = parquet_reader(data_dir / "train.parquet")
    validation = parquet_reader(data_dir / "validation.parquet")
    validation_ratings = validation[validation["rating"] > 0]
    if validation_ratings.empty:
        raise ValueError("validation contains no explicit ratings")
    explicit_matrix = sparse.load_npz(data_dir / "train_explicit.npz").tocsr()
    n_users, n_items = explicit_matrix.shape
    actual = validation_ratings["rating"].to_numpy(dtype=np.float64)
    validation_users = validation_ratings["user_idx"].to_numpy(dtype=np.int64)
    validation_items = validation_ratings["item_idx"].to_numpy(dtype=np.int64)
    clip_min, clip_max = validated["rating_clip"]

    results: list[dict[str, Any]] = []
    for model_name, specification in validated["models"].items():
        if specification.get("enabled", True) is False:
            continue
        for parameters in specification["candidates"]:
            model = _build_model(
                model_name,
                parameters,
                n_users=n_users,
                n_items=n_items,
                seed=validated["seed"],
            )
            started = time.perf_counter()
            if model_name in {"basic_mf", "funksvd"}:
                model.fit(train, validation)
            else:
                model.fit(explicit_matrix)
            training_seconds = time.perf_counter() - started

            started = time.perf_counter()
            raw_predictions = np.asarray(
                model.predict(validation_users, validation_items),
                dtype=np.float64,
            )
            inference_seconds = time.perf_counter() - started
            clipped_predictions = np.clip(raw_predictions, clip_min, clip_max)
            results.append(
                _result_row(
                    manifest=manifest,
                    model_name=model_name,
                    parameters=parameters,
                    selection_metric=validated["selection_metric"],
                    actual=actual,
                    predicted=clipped_predictions,
                    training_seconds=training_seconds,
                    inference_seconds=inference_seconds,
                )
            )

    best_configs: dict[str, dict[str, Any]] = {}
    metric = validated["selection_metric"]
    for model_name in sorted({row["model"] for row in results}):
        candidates = [row for row in results if row["model"] == model_name]
        best = min(
            candidates,
            key=lambda row: (
                row[metric],
                json.dumps(row["hyperparameters"], sort_keys=True),
            ),
        )
        best["selected"] = True
        best["best_validation_metric"] = best[metric]
        best_configs[model_name] = {
            "hyperparameters": best["hyperparameters"],
            "selection_metric": metric,
            "validation_metric": best[metric],
        }

    payload = {
        "schema_version": "goodbooks-validation-results-v1",
        "phase": "validation_selection",
        "dataset_version": manifest["version"],
        "seed": int(manifest["seed"]),
        "selection_metric": metric,
        "data_counts": manifest["counts"],
        "test_accessed": False,
        "ranking_metrics_status": "pending Ricky shared evaluation and candidate set",
        "best_configs": best_configs,
        "results": results,
    }
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return payload


def freeze_validation_configs(
    validation_payload: Mapping[str, Any],
    output_path: Path,
    *,
    model_names: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Create the artifact required before any future test deserialization."""
    if validation_payload.get("phase") != "validation_selection":
        raise ValueError("only validation selection results can be frozen")
    if validation_payload.get("test_accessed") is not False:
        raise ValueError("cannot freeze a selection that accessed test")
    best_configs = dict(validation_payload["best_configs"])
    if model_names is not None:
        requested = set(model_names)
        unknown = requested.difference(best_configs)
        if unknown:
            raise ValueError(
                f"cannot freeze unknown models: {', '.join(sorted(unknown))}"
            )
        best_configs = {
            name: value for name, value in best_configs.items() if name in requested
        }
    if not best_configs:
        raise ValueError("at least one selected model is required for freezing")
    artifact = {
        "schema_version": "goodbooks-frozen-config-v1",
        "status": "frozen",
        "dataset_version": validation_payload["dataset_version"],
        "seed": validation_payload["seed"],
        "selection_metric": validation_payload["selection_metric"],
        "best_configs": best_configs,
        "ranking_evaluation_status": "pending Ricky shared evaluation and candidate set",
    }
    artifact["config_hash"] = _canonical_hash(artifact)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return artifact


def _validated_frozen_artifact(
    data_dir: Path,
    frozen_config_path: Path,
    expected_manifest_path: Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact = json.loads(Path(frozen_config_path).read_text(encoding="utf-8"))
    if artifact.get("status") != "frozen":
        raise ValueError("test access requires a frozen configuration")
    stored_hash = artifact.get("config_hash")
    hash_payload = {key: value for key, value in artifact.items() if key != "config_hash"}
    if stored_hash != _canonical_hash(hash_payload):
        raise ValueError("frozen configuration hash mismatch")
    manifest = verify_bundle(Path(data_dir), expected_manifest_path)
    if artifact.get("dataset_version") != manifest.get("version"):
        raise ValueError("frozen configuration dataset_version mismatch")
    if artifact.get("seed") != manifest.get("seed"):
        raise ValueError("frozen configuration seed mismatch")
    return artifact, manifest


def load_test_after_freeze(
    data_dir: Path,
    frozen_config_path: Path,
    *,
    expected_manifest_path: Path | None = None,
    parquet_reader: Callable[[Path], pd.DataFrame] = pd.read_parquet,
) -> pd.DataFrame:
    """Gate future test access behind a matching frozen config artifact."""
    _validated_frozen_artifact(data_dir, frozen_config_path, expected_manifest_path)
    return parquet_reader(Path(data_dir) / "test.parquet")


def run_frozen_rating_test(
    data_dir: Path,
    frozen_config_path: Path,
    *,
    output_path: Path,
    expected_manifest_path: Path | None = None,
    parquet_reader: Callable[[Path], pd.DataFrame] = pd.read_parquet,
) -> dict[str, Any]:
    """Train frozen explicit models, then evaluate the test ratings once."""
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"test result already exists: {output_path}")
    artifact, manifest = _validated_frozen_artifact(
        Path(data_dir),
        Path(frozen_config_path),
        expected_manifest_path,
    )
    unsupported = set(artifact["best_configs"]).difference(
        {"als", "nmf", "bias_aware_als"}
    )
    if unsupported:
        raise ValueError(
            "phase-one frozen rating test supports only ALS/NMF variants: "
            + ", ".join(sorted(unsupported))
        )

    explicit_matrix = sparse.load_npz(Path(data_dir) / "train_explicit.npz").tocsr()
    n_users, n_items = explicit_matrix.shape
    trained: dict[str, tuple[Any, float, Mapping[str, Any], float]] = {}
    for model_name, selection in artifact["best_configs"].items():
        parameters = selection["hyperparameters"]
        model = _build_model(
            model_name,
            parameters,
            n_users=n_users,
            n_items=n_items,
            seed=int(artifact["seed"]),
        )
        started = time.perf_counter()
        model.fit(explicit_matrix)
        trained[model_name] = (
            model,
            time.perf_counter() - started,
            parameters,
            float(selection["validation_metric"]),
        )

    # This is the only content-level test read, after all configs are frozen
    # and every model has already been trained without access to test.
    test = parquet_reader(Path(data_dir) / "test.parquet")
    test_ratings = test[test["rating"] > 0]
    if test_ratings.empty:
        raise ValueError("test contains no explicit ratings")
    users = test_ratings["user_idx"].to_numpy(dtype=np.int64)
    items = test_ratings["item_idx"].to_numpy(dtype=np.int64)
    actual = test_ratings["rating"].to_numpy(dtype=np.float64)

    results = []
    for model_name, (model, training_seconds, parameters, validation_metric) in trained.items():
        started = time.perf_counter()
        raw_predictions = np.asarray(model.predict(users, items), dtype=np.float64)
        inference_seconds = time.perf_counter() - started
        row = _result_row(
            manifest=manifest,
            model_name=model_name,
            parameters=parameters,
            selection_metric=artifact["selection_metric"],
            actual=actual,
            predicted=np.clip(raw_predictions, 1.0, 5.0),
            training_seconds=training_seconds,
            inference_seconds=inference_seconds,
            evaluation_split="test",
            best_validation_metric=validation_metric,
        )
        row["selected"] = True
        results.append(row)

    payload = {
        "schema_version": "goodbooks-frozen-rating-test-v1",
        "phase": "frozen_rating_test",
        "dataset_version": manifest["version"],
        "seed": int(manifest["seed"]),
        "frozen_config_hash": artifact["config_hash"],
        "data_counts": manifest["counts"],
        "test_accessed": True,
        "ranking_metrics_status": "pending Ricky shared evaluation and candidate set",
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload


def run_frozen_unified_test(
    data_dir: Path,
    frozen_config_path: Path,
    *,
    output_path: Path,
    expected_manifest_path: Path | None = None,
    parquet_reader: Callable[[Path], pd.DataFrame] = pd.read_parquet,
) -> dict[str, Any]:
    """Evaluate frozen ALS/NMF configs with the shared rating/ranking API.

    Hyperparameters are read exclusively from the checksum-protected frozen
    validation artifact. The shared ``RankingEvaluationData`` is constructed
    once and reused by every model, ensuring identical users and candidates.
    """
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"unified test result already exists: {output_path}")
    artifact, manifest = _validated_frozen_artifact(
        Path(data_dir),
        Path(frozen_config_path),
        expected_manifest_path,
    )
    unsupported = set(artifact["best_configs"]).difference(
        {"als", "nmf", "bias_aware_als"}
    )
    if unsupported:
        raise ValueError(
            "Yutao unified test supports only ALS/NMF variants: "
            + ", ".join(sorted(unsupported))
        )

    data_dir = Path(data_dir)
    train = parquet_reader(data_dir / "train.parquet")
    test = parquet_reader(data_dir / "test.parquet")
    ranking_data = prepare_ranking_data(train, test)
    explicit_matrix = sparse.load_npz(data_dir / "train_explicit.npz").tocsr()
    n_users, n_items = explicit_matrix.shape

    results: list[dict[str, Any]] = []
    for model_name, selection in artifact["best_configs"].items():
        parameters = selection["hyperparameters"]
        model = _build_model(
            model_name,
            parameters,
            n_users=n_users,
            n_items=n_items,
            seed=int(artifact["seed"]),
        )
        started = time.perf_counter()
        model.fit(explicit_matrix)
        training_seconds = time.perf_counter() - started

        started = time.perf_counter()
        rating_metrics = evaluate_ratings(model, test)
        rating_inference_seconds = time.perf_counter() - started
        started = time.perf_counter()
        ranking_metrics = evaluate_ranking(model, ranking_data)
        ranking_inference_seconds = time.perf_counter() - started

        row = {
            "dataset_version": manifest["version"],
            "evaluation_split": "test",
            "model": model_name,
            "seed": int(manifest["seed"]),
            "n_factors": parameters.get("n_factors"),
            "learning_rate": parameters.get("learning_rate"),
            "reg_lambda": parameters.get("reg_lambda"),
            "iterations_or_epochs": _iterations(parameters),
            "selection_metric": artifact["selection_metric"],
            "best_validation_metric": float(selection["validation_metric"]),
            **rating_metrics,
            **ranking_metrics,
            "training_seconds": float(training_seconds),
            "inference_seconds": float(
                rating_inference_seconds + ranking_inference_seconds
            ),
            "rating_inference_seconds": float(rating_inference_seconds),
            "ranking_inference_seconds": float(ranking_inference_seconds),
            "hyperparameters": dict(parameters),
            "selected": True,
        }
        missing = set(RESULT_FIELDS).difference(row)
        if missing:
            raise AssertionError(f"incomplete result schema: {sorted(missing)}")
        results.append(row)

    payload = {
        "schema_version": "goodbooks-unified-test-results-v1",
        "phase": "frozen_unified_test",
        "dataset_version": manifest["version"],
        "seed": int(manifest["seed"]),
        "frozen_config_hash": artifact["config_hash"],
        "selection_source": "validation_only",
        "data_counts": manifest["counts"],
        "test_accessed": True,
        "ranking_protocol": {
            "candidate_policy": "full_train_catalog_excluding_seen",
            "relevance": "rating >= 4 OR (rating == 0 AND is_read)",
            "k_values": [5, 10, 20],
        },
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload
