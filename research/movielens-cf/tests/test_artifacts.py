import pytest

from movielens_cf.artifacts import validate_frontend_artifacts


def valid_payloads():
    metrics = {
        "version": "movielens-cf-v4",
        "generatedAtUtc": "2026-08-12T00:00:00+00:00",
        "experimentCodeVersion": "movielens-cf-v3",
        "dataset": "MovieLens 1M",
        "seed": 42,
        "split": "temporal",
        "relevance": "rating >= 4",
        "candidatePolicy": "full catalog minus seen",
        "splitCounts": {"trainRatings": 70, "validationRatings": 10, "fittedRatings": 80, "testRatings": 20},
        "baselines": {"bayesianPopularity": {"test": {"hit_rate_at_10": 0.5}, "examples": [{}, {}]}},
        "models": [
            {
                "key": "user-cf",
                "algorithmVariant": "bias_aware",
                "selection": {"primaryMetric": "ndcg_at_10", "reason": "validation winner"},
                "test": {"hit_rate_at_10": 0.4},
                "system": {"rankingTieStats": {"fullyTiedListShare": 0.0}},
            },
            {
                "key": "item-cf",
                "algorithmVariant": "bias_aware",
                "selection": {"primaryMetric": "ndcg_at_10", "reason": "validation winner"},
                "test": {"hit_rate_at_10": 0.3},
                "system": {
                    "rankingTieStats": {"fullyTiedListShare": 0.0},
                    "top10FiveStarOnlyEvidenceShare": 0.1,
                },
            },
        ],
        "postHocLimitation": "Inspired by diagnostics already observed on the v3 test set.",
        "sensitivity": {"ratingAtLeast5": {"readOnly": True}},
    }
    samples = {
        "version": "movielens-samples-v4",
        "users": [{
            "userId": 1,
            "historyTotal": 12,
            "relevantTestTotal": 1,
            "relevantTest": [{"movieId": 9}],
            "methods": {
                "popularity": [{"movieId": 9, "rankScore": 4.2, "hit": True}],
                "userCf": [{"movieId": 8, "rankScore": 5.3, "ratingEstimate": 5.0, "similarityWeight": 1.2, "hit": False}],
                "itemCf": [{"movieId": 7, "rankScore": 4.8, "ratingEstimate": 4.8, "similarityWeight": 0.9, "hit": False}],
            },
        }],
        "relatedItems": [],
    }
    profile = {"fields": [
        {"name": name} for name in ("user_id", "movie_id", "rating", "timestamp", "title", "genres")
    ]}
    return metrics, samples, profile


def test_frontend_artifact_schema_accepts_provenance_and_three_methods():
    metrics, samples, profile = valid_payloads()

    validate_frontend_artifacts(metrics, samples, profile)


def test_frontend_artifact_schema_rejects_missing_method_or_provenance():
    metrics, samples, profile = valid_payloads()
    del metrics["seed"]
    del samples["users"][0]["methods"]["popularity"]

    with pytest.raises(ValueError, match="seed"):
        validate_frontend_artifacts(metrics, samples, profile)


def test_v4_schema_rejects_missing_item_cf_tie_diagnostics():
    metrics, samples, profile = valid_payloads()
    del metrics["models"][1]["system"]["rankingTieStats"]

    with pytest.raises(ValueError, match="rankingTieStats"):
        validate_frontend_artifacts(metrics, samples, profile)
