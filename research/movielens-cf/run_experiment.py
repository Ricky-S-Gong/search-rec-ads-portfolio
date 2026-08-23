"""Run the approved CPU MovieLens 1M CF benchmark and publish compact artifacts."""

from __future__ import annotations

import argparse
import itertools
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn

from movielens_cf.artifacts import validate_frontend_artifacts, write_json
from movielens_cf.baselines import BayesianPopularity, BiasBaseline
from movielens_cf.data import dataset_profile, load_movielens
from movielens_cf.evaluation import (
    paired_bootstrap_interval,
    ranking_metrics,
    ranking_tie_stats,
    rating_metrics,
    relevant_truth,
    recommendation_stats,
    user_bootstrap_interval,
)
from movielens_cf.neighborhood import NeighborhoodCF
from movielens_cf.split import per_user_temporal_split


SEED = 42
K = 10
RELEVANCE = 4.0
K_VALUES = (20, 40, 80)
MIN_SUPPORT_VALUES = (5, 10, 20)
SHRINKAGE_VALUES = (10.0, 25.0, 50.0)
MIN_NEIGHBOR_VALUES = (2, 5)
PRIOR_WEIGHTS = (10.0, 25.0, 50.0, 100.0)
LEGACY_V3_PARAMS = {
    "user": {"k": 80, "min_support": 10, "shrinkage": 50.0, "min_neighbors": 5},
    "item": {"k": 20, "min_support": 5, "shrinkage": 10.0, "min_neighbors": 2},
}


def recommendation_ids(values: dict) -> dict[int, list[int]]:
    return {
        int(user_id): [int(item.movie_id) for item in recommendations]
        for user_id, recommendations in values.items()
    }


def popularity_recommendations(
    model: BayesianPopularity, fitted: pd.DataFrame, users: list[int], n: int = K
) -> dict[int, list[int]]:
    seen = {
        int(user_id): set(group["movie_id"].astype(int))
        for user_id, group in fitted.groupby("user_id", observed=True)
    }
    return {
        user_id: [item for item in model.ranking if item not in seen.get(user_id, set())][:n]
        for user_id in users
    }


def evaluate_configuration(
    fitted: pd.DataFrame,
    holdout: pd.DataFrame,
    mode: str,
    params: dict,
    variant: str = "bias_aware",
    secondary_ordering: str = "bayesian",
) -> tuple[NeighborhoodCF, dict, pd.DataFrame, dict[int, list[int]], dict]:
    started = time.perf_counter()
    model = NeighborhoodCF(
        mode=mode,
        variant=variant,
        secondary_ordering=secondary_ordering,
        block_size=384,
        **params,
    ).fit(fitted)
    fit_seconds = time.perf_counter() - started
    truth = relevant_truth(holdout, set(model.movie_ids.astype(int)), threshold=RELEVANCE)
    started = time.perf_counter()
    detailed = model.recommend_many(list(truth), n=K)
    recommend_seconds = time.perf_counter() - started
    recommendations = recommendation_ids(detailed)
    metrics, per_user = ranking_metrics(truth, recommendations, k=K)
    fallback = [item.used_fallback for values in detailed.values() for item in values]
    personalized = [item for values in detailed.values() for item in values if not item.used_fallback]
    personalized_scores = {
        user_id: scores
        for user_id, values in detailed.items()
        if (scores := [item.ranking_score for item in values if not item.used_fallback])
    }
    system = {
        "fitSeconds": fit_seconds,
        "recommendSeconds": recommend_seconds,
        "usersPerSecond": len(truth) / recommend_seconds if recommend_seconds else 0.0,
        "fallbackShare": float(np.mean(fallback)) if fallback else 0.0,
        "top10RawAboveRatingMaxShare": (
            float(np.mean([item.ranking_score > 5 for item in personalized]))
            if personalized else 0.0
        ),
        "top10ClippedShare": (
            float(np.mean([item.ranking_score != item.rating_estimate for item in personalized]))
            if personalized else 0.0
        ),
        "rankingTieStats": ranking_tie_stats(personalized_scores),
        "retainedNeighbors": int(model.similarity.nnz),
        "neighborArtifactMiB": float(
            (model.similarity.data.nbytes + model.similarity.indices.nbytes + model.similarity.indptr.nbytes)
            / 2**20
        ),
    }
    if mode == "item" and personalized:
        five_star_only = []
        for user_id, values in detailed.items():
            user_index = model.user_index[user_id]
            for recommendation in values:
                if recommendation.used_fallback:
                    continue
                item_index = model.movie_index[recommendation.movie_id]
                neighbors = model.similarity.getrow(item_index).indices
                rated = model.binary[user_index, neighbors].toarray().ravel() > 0
                source_ratings = model.ratings[user_index, neighbors[rated]].toarray().ravel()
                five_star_only.append(bool(len(source_ratings) and np.all(source_ratings == 5.0)))
        system["top10FiveStarOnlyEvidenceShare"] = float(np.mean(five_star_only))
    return model, metrics, per_user, recommendations, system


def rating_evaluation(model: NeighborhoodCF, holdout: pd.DataFrame) -> dict:
    predictable = holdout.loc[
        holdout["user_id"].isin(model.user_index) & holdout["movie_id"].isin(model.movie_index)
    ]
    predictions = [
        model.predict(int(row.user_id), int(row.movie_id)).rating
        for row in predictable.itertuples(index=False)
    ]
    return {
        **rating_metrics(predictable["rating"], predictions),
        "ratings": int(len(predictable)),
        "coverage": float(len(predictable) / len(holdout)),
    }


def segment_metrics(
    per_user: pd.DataFrame, fitted: pd.DataFrame
) -> list[dict]:
    activity = fitted.groupby("user_id", observed=True).size().rename("trainingRatings")
    frame = per_user.merge(activity, left_on="user_id", right_index=True)
    frame["segment"] = pd.qcut(
        frame["trainingRatings"].rank(method="first"),
        4,
        labels=["sparse", "light", "active", "power"],
    )
    return [
        {
            "segment": str(segment),
            "users": int(len(group)),
            "trainingRatingsMedian": float(group["trainingRatings"].median()),
            "hitRateAt10": float(group["hit_at_10"].mean()),
            "ndcgAt10": float(group["ndcg_at_10"].mean()),
            "recallAt10": float(group["recall_at_10"].mean()),
        }
        for segment, group in frame.groupby("segment", observed=True)
    ]


def sample_payload(
    fitted: pd.DataFrame,
    test: pd.DataFrame,
    movies: pd.DataFrame,
    user_model: NeighborhoodCF,
    item_model: NeighborhoodCF,
    popularity: BayesianPopularity,
    popular_recs: dict[int, list[int]],
    evaluated_recommendations: dict[str, dict[int, list[int]]],
) -> dict:
    movie_rows = movies.set_index("movie_id")
    title = movie_rows["title"].astype(str).to_dict()
    genres = movie_rows["genres"].astype(str).to_dict()
    activity = fitted.groupby("user_id", observed=True).size().sort_values()
    truth = relevant_truth(test, set(item_model.movie_ids.astype(int)))
    eligible = sorted(truth)
    hit_users = [
        user_id for user_id in eligible
        if any(
            truth[user_id].intersection(recommendations.get(user_id, []))
            for recommendations in (popular_recs, *evaluated_recommendations.values())
        )
    ]
    pool = hit_users or eligible
    targets = activity.loc[pool].quantile([0.15, 0.5, 0.85]).to_numpy()
    users = []
    for target in targets:
        user_id = min(
            (candidate for candidate in pool if candidate not in users),
            key=lambda candidate: (abs(float(activity.loc[candidate]) - target), candidate),
        )
        users.append(int(user_id))
    user_recs = user_model.recommend_many(users, n=K)
    item_recs = item_model.recommend_many(users, n=K)
    item_counts = fitted.groupby("movie_id", observed=True).size().sort_values(ascending=False)
    head = set(item_counts.head(max(1, int(np.ceil(len(item_counts) * 0.2)))).index.astype(int))

    def movie_payload(movie_id: int, **extra: object) -> dict:
        return {
            "movieId": int(movie_id),
            "title": title.get(int(movie_id), str(movie_id)),
            "genres": genres.get(int(movie_id), "Unknown").split("|"),
            "popularityBand": "head" if int(movie_id) in head else "long-tail",
            **extra,
        }

    def evidence(model: NeighborhoodCF, user_id: int, movie_id: int) -> list[dict]:
        user = model.user_index[user_id]
        item = model.movie_index[movie_id]
        neighbors = model.similarity.getrow(user if model.mode == "user" else item)
        if model.mode == "user":
            mask = model.binary[neighbors.indices, item].toarray().ravel() > 0
            indices, weights = neighbors.indices[mask], neighbors.data[mask]
            residuals = model.residual_matrix[indices, item].toarray().ravel()
            labels = [f"Viewer {int(model.user_ids[index])}" for index in indices]
            source_ratings = model.ratings[indices, item].toarray().ravel()
        else:
            mask = model.binary[user, neighbors.indices].toarray().ravel() > 0
            indices, weights = neighbors.indices[mask], neighbors.data[mask]
            residuals = model.residual_matrix[user, indices].toarray().ravel()
            labels = [title.get(int(model.movie_ids[index]), str(model.movie_ids[index])) for index in indices]
            source_ratings = model.ratings[user, indices].toarray().ravel()
        denominator = float(np.abs(weights).sum())
        contributions = weights * residuals / denominator if denominator else np.zeros_like(weights)
        order = np.lexsort((np.asarray(labels), -np.abs(contributions)))[:3]
        return [
            {
                "source": labels[position],
                "similarity": round(float(weights[position]), 4),
                "residual": round(float(residuals[position]), 3),
                "sourceRating": round(float(source_ratings[position]), 1),
                "contribution": round(float(contributions[position]), 4),
            }
            for position in order
        ]

    def cf_payload(model: NeighborhoodCF, user_id: int, recommendation: object) -> dict:
        movie_id = int(recommendation.movie_id)
        return movie_payload(
            movie_id,
            rankScore=round(float(recommendation.ranking_score), 4),
            ratingEstimate=round(float(recommendation.rating_estimate), 3),
            similarityWeight=round(float(recommendation.similarity_weight_sum), 4),
            scoreWasClipped=bool(recommendation.ranking_score != recommendation.rating_estimate),
            neighbors=int(recommendation.neighbor_count),
            fallback=bool(recommendation.used_fallback),
            secondaryScore=round(float(recommendation.secondary_score), 4),
            secondaryScoreName=recommendation.secondary_score_name,
            hit=movie_id in truth[user_id],
            evidence=[] if recommendation.used_fallback else evidence(model, user_id, movie_id),
        )

    examples = []
    for user_id in users:
        history = fitted.loc[fitted["user_id"] == user_id].sort_values(
            ["rating", "timestamp"], ascending=[False, False]
        ).head(5)
        relevant = test.loc[
            (test["user_id"] == user_id) & (test["rating"] >= RELEVANCE)
        ].sort_values(["timestamp", "movie_id"])
        examples.append(
            {
                "userId": user_id,
                "user": f"Viewer {user_id}",
                "activity": int(activity.loc[user_id]),
                "historyTotal": int(activity.loc[user_id]),
                "history": [
                    movie_payload(int(row.movie_id), rating=float(row.rating))
                    for row in history.itertuples(index=False)
                ],
                "relevantTest": [
                    movie_payload(int(row.movie_id), rating=float(row.rating))
                    for row in relevant.itertuples(index=False)
                ],
                "relevantTestTotal": int(len(relevant)),
                "methods": {
                    "popularity": [
                        movie_payload(
                            movie_id,
                            rankScore=round(float(popularity.scores[movie_id]), 4),
                            ratingEstimate=None,
                            scoreWasClipped=False,
                            neighbors=0,
                            fallback=True,
                            hit=movie_id in truth[user_id],
                            evidence=[],
                        )
                        for movie_id in popular_recs[user_id]
                    ],
                    "userCf": [cf_payload(user_model, user_id, item) for item in user_recs[user_id]],
                    "itemCf": [cf_payload(item_model, user_id, item) for item in item_recs[user_id]],
                },
            }
        )
    seeds = [
        movie_id
        for query in ("Toy Story (1995)", "Matrix, The (1999)", "Fargo (1996)")
        for movie_id in movies.loc[movies["title"] == query, "movie_id"].astype(int).head(1)
    ]
    related = []
    for movie_id in seeds:
        index = item_model.movie_index.get(movie_id)
        if index is None:
            continue
        row = item_model.similarity.getrow(index)
        order = np.lexsort((item_model.movie_ids[row.indices], -row.data))[:5]
        related.append(
            {
                "seed": movie_payload(movie_id),
                "neighbors": [
                    movie_payload(
                        int(item_model.movie_ids[row.indices[position]]),
                        similarity=round(float(row.data[position]), 4),
                        support=int(
                            item_model.binary[:, index].multiply(
                                item_model.binary[:, row.indices[position]]
                            ).sum()
                        ),
                    )
                    for position in order
                ],
            }
        )
    return {"version": "movielens-samples-v4", "users": examples, "relatedItems": related}


def make_figures(profile: dict, models: list[dict], fitted: pd.DataFrame, output: Path) -> None:
    def save_svg(fig, path: Path) -> None:
        fig.savefig(path, facecolor=fig.get_facecolor())
        path.write_text(
            "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n",
            encoding="utf-8",
        )

    plt.style.use("dark_background")
    colors = ["#35d0e2", "#a78bfa", "#f7b955"]
    labels = [model["label"] for model in models]
    ndcg = [model["test"]["ndcg_at_10"] for model in models]
    recall = [model["test"]["recall_at_10"] for model in models]
    hit_rate = [model["test"]["hit_rate_at_10"] for model in models]
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#07111f")
    ax.set_facecolor("#07111f")
    x = np.arange(len(labels)); width = 0.24
    ax.bar(x - width, ndcg, width, label="NDCG@10", color=colors)
    ax.bar(x, recall, width, label="Recall@10", color=colors, alpha=0.55)
    ax.bar(x + width, hit_rate, width, label="Hit Rate@10", color=colors, alpha=0.3)
    ax.set_xticks(x, labels); ax.set_ylim(0, max(ndcg + recall + hit_rate) * 1.25)
    ax.set_ylabel("Higher is better"); ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False); fig.tight_layout()
    save_svg(fig, output / "model-ranking.svg"); plt.close(fig)

    counts = fitted.groupby("movie_id", observed=True).size().sort_values(ascending=False).to_numpy()
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#07111f")
    ax.set_facecolor("#07111f")
    ax.plot(np.arange(1, len(counts) + 1), counts, color="#a78bfa", linewidth=2)
    ax.fill_between(np.arange(1, len(counts) + 1), counts, color="#a78bfa", alpha=0.15)
    ax.set_yscale("log"); ax.set_xlabel("Popularity ranking of movies")
    ax.set_ylabel("Number of ratings"); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); save_svg(fig, output / "popularity-long-tail.svg"); plt.close(fig)

def run(data_dir: Path, output: Path, smoke: bool = False) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    ratings, movies = load_movielens(data_dir)
    if smoke:
        smoke_users = np.sort(ratings["user_id"].unique())[:250]
        ratings = ratings.loc[ratings["user_id"].isin(smoke_users)].copy()
    profile = dataset_profile(ratings, movies)
    train, validation, test = per_user_temporal_split(ratings)
    validation_results = []
    selected = {}
    selection_details = {}
    train_item_counts = train.groupby("movie_id", observed=True).size()

    def candidate_sort_key(value: dict) -> tuple:
        metrics = value["metrics"]
        ties = value["system"]["rankingTieStats"]
        return (
            -metrics["ndcg_at_10"],
            -metrics["recall_at_10"],
            -metrics["hit_rate_at_10"],
            ties["fullyTiedListShare"],
            -value["beyondAccuracy"]["catalog_coverage"],
            value["system"]["fitSeconds"],
        )

    for mode in ("user", "item"):
        stage_one_params = (
            [LEGACY_V3_PARAMS[mode]]
            if smoke
            else [
                {
                    "k": k,
                    "min_support": support,
                    "shrinkage": shrinkage,
                    "min_neighbors": 2,
                }
                for k, support, shrinkage in itertools.product(
                    K_VALUES, MIN_SUPPORT_VALUES, SHRINKAGE_VALUES
                )
            ]
        )
        evaluated: dict[tuple, dict] = {}

        def evaluate_candidate(params: dict, stage: str) -> dict:
            key = tuple(params[name] for name in ("k", "min_support", "shrinkage", "min_neighbors"))
            if key in evaluated:
                return evaluated[key]
            model, candidate_metrics, _, recommendations, system = evaluate_configuration(
                train, validation, mode, params, variant="bias_aware"
            )
            record = {
                "stage": stage,
                "params": params,
                "metrics": candidate_metrics,
                "system": system,
                "beyondAccuracy": recommendation_stats(
                    recommendations, train_item_counts, len(model.movie_ids)
                ),
            }
            evaluated[key] = record
            return record

        stage_one = [evaluate_candidate(params, "orthogonal-grid") for params in stage_one_params]
        stage_one.sort(key=candidate_sort_key)
        finalists = stage_one[:1] if smoke else stage_one[:3]
        for finalist in finalists:
            base = finalist["params"]
            for min_neighbors in MIN_NEIGHBOR_VALUES:
                evaluate_candidate(
                    {**base, "min_neighbors": min_neighbors}, "min-neighbors-finalist"
                )
        candidates = sorted(evaluated.values(), key=candidate_sort_key)
        selected[mode] = candidates[0]["params"]
        selection_details[mode] = {
            "primaryMetric": "ndcg_at_10",
            "tieBreakOrder": [
                "recall_at_10",
                "hit_rate_at_10",
                "fullyTiedListShare",
                "catalog_coverage",
            ],
            "reason": "Best validation result under the predeclared lexicographic rule.",
        }
        validation_results.append(
            {
                "mode": mode,
                "algorithmVariant": "bias_aware",
                "searchProtocol": (
                    "Smoke uses the legacy-sized configuration. Full run evaluates the 3x3x3 "
                    "orthogonal k/support/shrinkage grid at min_neighbors=2, then compares "
                    "min_neighbors={2,5} for the top three validation candidates."
                ),
                "candidates": candidates,
                "selected": selected[mode],
                "selection": selection_details[mode],
            }
        )

    validation_truth = relevant_truth(
        validation, set(train_item_counts.index.astype(int)), threshold=RELEVANCE
    )
    popularity_validation = []
    for prior_weight in (PRIOR_WEIGHTS[:1] if smoke else PRIOR_WEIGHTS):
        candidate = BayesianPopularity(prior_weight=prior_weight).fit(train)
        recs = popularity_recommendations(candidate, train, list(validation_truth))
        candidate_metrics, _ = ranking_metrics(validation_truth, recs, k=K)
        popularity_validation.append(
            {
                "priorWeight": prior_weight,
                "metrics": candidate_metrics,
                "beyondAccuracy": recommendation_stats(
                    recs, train_item_counts, len(train_item_counts)
                ),
            }
        )
    popularity_validation.sort(
        key=lambda value: (
            -value["metrics"]["ndcg_at_10"],
            -value["metrics"]["recall_at_10"],
            -value["metrics"]["hit_rate_at_10"],
            -value["beyondAccuracy"]["catalog_coverage"],
        )
    )
    selected_prior_weight = popularity_validation[0]["priorWeight"]

    secondary_ablation = []
    for ordering in ("none", "bayesian"):
        ablation_model, ablation_metrics, _, ablation_recs, ablation_system = evaluate_configuration(
            train,
            validation,
            "item",
            selected["item"],
            variant="bias_aware",
            secondary_ordering=ordering,
        )
        secondary_ablation.append(
            {
                "ordering": ordering,
                "metrics": ablation_metrics,
                "beyondAccuracy": recommendation_stats(
                    ablation_recs, train_item_counts, len(ablation_model.movie_ids)
                ),
                "rankingTieStats": ablation_system["rankingTieStats"],
            }
        )

    fitted = pd.concat([train, validation], ignore_index=True)
    final_models = []
    model_objects = {}
    final_per_user = {}
    final_recommendations = {}
    item_counts = fitted.groupby("movie_id", observed=True).size()
    for mode in ("user", "item"):
        model, rank, per_user, recommendations, system = evaluate_configuration(
            fitted, test, mode, selected[mode], variant="bias_aware"
        )
        model_objects[mode] = model
        final_per_user[mode] = per_user
        final_recommendations[mode] = recommendations
        beyond = recommendation_stats(recommendations, item_counts, len(model.movie_ids))
        final_models.append(
            {
                "key": f"{mode}-cf",
                "label": "User-CF" if mode == "user" else "Item-CF",
                "mode": mode,
                "algorithmVariant": "bias_aware",
                "similarity": "Baseline-residual cosine",
                "secondaryOrdering": "Bayesian popularity, then evidence strength, then movie ID",
                "hyperparameters": selected[mode],
                "selection": selection_details[mode],
                "test": rank,
                "rating": rating_evaluation(model, test),
                "beyondAccuracy": beyond,
                "system": system,
                "segments": segment_metrics(per_user, fitted),
                "confidence95": {
                    "hitRateAt10": user_bootstrap_interval(per_user, "hit_at_10"),
                    "ndcgAt10": user_bootstrap_interval(per_user, "ndcg_at_10"),
                    "recallAt10": user_bootstrap_interval(per_user, "recall_at_10"),
                },
            }
        )

    legacy_v3_models = []
    for mode in ("user", "item"):
        legacy_model, legacy_rank, _, legacy_recommendations, legacy_system = evaluate_configuration(
            fitted,
            test,
            mode,
            LEGACY_V3_PARAMS[mode],
            variant="mean_centered",
            secondary_ordering="legacy_id",
        )
        legacy_v3_models.append(
            {
                "key": f"{mode}-cf",
                "algorithmVariant": "mean_centered",
                "hyperparameters": LEGACY_V3_PARAMS[mode],
                "test": legacy_rank,
                "beyondAccuracy": recommendation_stats(
                    legacy_recommendations, item_counts, len(legacy_model.movie_ids)
                ),
                "system": legacy_system,
            }
        )

    truth = relevant_truth(test, set(item_counts.index.astype(int)), threshold=RELEVANCE)
    popularity = BayesianPopularity(prior_weight=selected_prior_weight).fit(fitted)
    popular_recs = popularity_recommendations(popularity, fitted, list(truth))
    popular_rank, popular_per_user = ranking_metrics(truth, popular_recs, k=K)
    truth_five = relevant_truth(test, set(item_counts.index.astype(int)), threshold=5.0)
    sensitivity = {
        "ratingAtLeast5": {
            "readOnly": True,
            "selectionUse": "None; this threshold never selects a model or replaces the main rating>=4 protocol.",
            "population": len(truth_five),
            "methods": {
                "popularity": ranking_metrics(truth_five, popular_recs, k=K)[0],
                "user-cf": ranking_metrics(
                    truth_five, final_recommendations["user"], k=K
                )[0],
                "item-cf": ranking_metrics(
                    truth_five, final_recommendations["item"], k=K
                )[0],
            },
        }
    }
    movie_titles = movies.set_index("movie_id")["title"].astype(str).to_dict()
    bayesian_rows = fitted.groupby("movie_id", observed=True)["rating"].agg(["count", "mean"])
    low_id = int(bayesian_rows.sort_values(["count", "mean"], ascending=[True, False]).index[0])
    high_id = int(bayesian_rows.sort_values(["count", "mean"], ascending=[False, False]).index[0])
    bayesian_examples = [
        {
            "support": label,
            "movieId": movie_id,
            "title": movie_titles.get(movie_id, str(movie_id)),
            "ratingCount": int(bayesian_rows.loc[movie_id, "count"]),
            "ratingMean": float(bayesian_rows.loc[movie_id, "mean"]),
            "score": float(popularity.scores[movie_id]),
        }
        for label, movie_id in (("low", low_id), ("high", high_id))
    ]
    bias = BiasBaseline().fit(fitted)
    predictable = test.loc[test["user_id"].isin(bias.user_bias) & test["movie_id"].isin(bias.item_bias)]
    bias_predictions = [bias.predict(int(row.user_id), int(row.movie_id)) for row in predictable.itertuples(index=False)]
    baselines = {
        "bayesianPopularity": {
            "priorWeight": popularity.prior_weight,
            "validationCandidates": popularity_validation,
            "selection": {
                "primaryMetric": "ndcg_at_10",
                "reason": "Selected on validation only from prior_weight={10,25,50,100}.",
            },
            "globalMean": popularity.global_mean,
            "examples": bayesian_examples,
            "test": popular_rank,
            "beyondAccuracy": recommendation_stats(popular_recs, item_counts, len(item_counts)),
            "confidence95": {
                "hitRateAt10": user_bootstrap_interval(popular_per_user, "hit_at_10"),
                "ndcgAt10": user_bootstrap_interval(popular_per_user, "ndcg_at_10"),
                "recallAt10": user_bootstrap_interval(popular_per_user, "recall_at_10"),
            },
        },
        "biasRating": {
            **rating_metrics(predictable["rating"], bias_predictions),
            "ratings": int(len(predictable)),
        },
    }
    runtime = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikitLearn": sklearn.__version__,
        "compute": "CPU",
        "platform": platform.platform(),
        "timingContext": "One single-process offline batch over the full ranking cohort; recommendSeconds includes candidate scoring, seen-item removal, and Top-10 selection, but excludes fitting, data loading, and artifact writing. BLAS thread count was not forced.",
    }
    metadata = {
        "version": "movielens-cf-v4",
        "experimentCodeVersion": "movielens-cf-v4",
        "generatedAtUtc": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": "MovieLens 1M smoke subset" if smoke else "MovieLens 1M",
        "datasetUrl": "https://grouplens.org/datasets/movielens/1m/",
        "archiveMd5": "c4d9eecfca2ab87c1945afe126590906",
        "seed": SEED,
        "split": "Per-user temporal 80/10/10; refit train+validation for final test",
        "relevance": "rating >= 4",
        "candidatePolicy": "Full fitted catalog minus seen movies",
        "k": K,
        "runtime": runtime,
        "splitCounts": {
            "trainRatings": int(len(train)),
            "validationRatings": int(len(validation)),
            "fittedRatings": int(len(fitted)),
            "testRatings": int(len(test)),
        },
    }
    metrics = {
        **metadata,
        "evaluationPopulation": {
            "rankingUsers": len(truth),
            "testRatings": int(len(test)),
            "trainingRatings": int(len(fitted)),
            "fittedCatalogMovies": int(len(item_counts)),
        },
        "baselines": baselines,
        "models": final_models,
        "postHocLimitation": (
            "This v4 model iteration was motivated by tie failures already observed in the v3 test "
            "results. Hyperparameters remain validation-only, but the iteration is post-hoc rather "
            "than a fully untouched confirmatory experiment."
        ),
        "sensitivity": sensitivity,
        "diagnostics": {
            "relevantItemHeadConcentration": {
                "definition": "Movies ranked by train+validation interaction count; head is the top 20% of fitted catalog items; denominator is relevant test ratings for the ranking cohort.",
                "relevantTestRatings": int(
                    test.loc[
                        (test["rating"] >= RELEVANCE)
                        & test["movie_id"].isin(item_counts.index)
                        & test["user_id"].isin(truth)
                    ].shape[0]
                ),
                "top10PercentShare": float(
                    test.loc[
                        (test["rating"] >= RELEVANCE)
                        & test["movie_id"].isin(item_counts.index)
                        & test["user_id"].isin(truth),
                        "movie_id",
                    ]
                    .map(item_counts.rank(method="average", ascending=False, pct=True))
                    .le(0.1).mean()
                ),
                "top20PercentShare": float(
                    test.loc[
                        (test["rating"] >= RELEVANCE)
                        & test["movie_id"].isin(item_counts.index)
                        & test["user_id"].isin(truth),
                        "movie_id",
                    ]
                    .map(item_counts.rank(method="average", ascending=False, pct=True))
                    .le(0.2).mean()
                ),
            },
            "pairedUserCfMinusPopularity": {
                "ndcgAt10": paired_bootstrap_interval(
                    final_per_user["user"], popular_per_user, "ndcg_at_10"
                ),
                "recallAt10": paired_bootstrap_interval(
                    final_per_user["user"], popular_per_user, "recall_at_10"
                ),
                "hitRateAt10": paired_bootstrap_interval(
                    final_per_user["user"], popular_per_user, "hit_at_10"
                ),
            },
            "rankingCorrection": "Top-N uses raw neighborhood estimates; rating RMSE/MAE uses estimates clipped to [1, 5].",
            "secondaryOrderingValidation": secondary_ablation,
            "legacyV3": legacy_v3_models,
        },
        "metricCaveat": "Offline explicit-rating results; they do not measure CTR, watch time, retention, revenue, or causal lift.",
    }
    samples = sample_payload(
        fitted,
        test,
        movies,
        model_objects["user"],
        model_objects["item"],
        popularity,
        popular_recs,
        final_recommendations,
    )
    validate_frontend_artifacts(metrics, samples, profile)
    write_json(output / "profile.json", {**metadata, "profile": profile})
    write_json(output / "metrics.json", metrics)
    write_json(
        output / "comparisons.json",
        {
            **metadata,
            "validation": validation_results,
            "popularityValidation": popularity_validation,
            "secondaryOrderingValidation": secondary_ablation,
            "legacyV3": legacy_v3_models,
            "models": final_models,
            "baselines": baselines,
            "postHocLimitation": metrics["postHocLimitation"],
        },
    )
    write_json(output / "samples.json", samples)
    make_figures(profile, [{"label": "Popularity", "test": popular_rank}, *final_models], fitted, output)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("data") / "ml-1m")
    parser.add_argument("--output", type=Path, default=Path(__file__).parents[2] / "public" / "artifacts" / "movielens")
    parser.add_argument("--smoke", action="store_true", help="Run a fast 250-user end-to-end verification")
    args = parser.parse_args()
    metrics = run(args.data, args.output, smoke=args.smoke)
    for model in metrics["models"]:
        print(model["label"], model["test"], model["rating"], model["system"])


if __name__ == "__main__":
    main()
