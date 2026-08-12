import pytest

from movielens_cf.artifacts import validate_frontend_artifacts


def valid_payloads():
    metrics = {
        "version": "movielens-cf-v2",
        "generatedAtUtc": "2026-08-12T00:00:00+00:00",
        "experimentCodeVersion": "movielens-cf-v2",
        "dataset": "MovieLens 1M",
        "seed": 42,
        "split": "temporal",
        "relevance": "rating >= 4",
        "candidatePolicy": "full catalog minus seen",
        "models": [{"key": "user-cf"}, {"key": "item-cf"}],
    }
    samples = {
        "version": "movielens-samples-v2",
        "users": [{
            "userId": 1,
            "relevantTest": [{"movieId": 9}],
            "methods": {
                "popularity": [{"movieId": 9, "rankScore": 4.2, "hit": True}],
                "userCf": [{"movieId": 8, "rankScore": 5.3, "ratingEstimate": 5.0, "similarityWeight": 1.2, "hit": False}],
                "itemCf": [{"movieId": 7, "rankScore": 4.8, "ratingEstimate": 4.8, "similarityWeight": 0.9, "hit": False}],
            },
        }],
        "relatedItems": [],
    }
    return metrics, samples


def test_frontend_artifact_schema_accepts_provenance_and_three_methods():
    metrics, samples = valid_payloads()

    validate_frontend_artifacts(metrics, samples)


def test_frontend_artifact_schema_rejects_missing_method_or_provenance():
    metrics, samples = valid_payloads()
    del metrics["seed"]
    del samples["users"][0]["methods"]["popularity"]

    with pytest.raises(ValueError, match="seed"):
        validate_frontend_artifacts(metrics, samples)
