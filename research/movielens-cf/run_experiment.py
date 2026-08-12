"""Run the approved CPU MovieLens 1M CF benchmark and publish compact artifacts."""

from __future__ import annotations

import argparse
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn

from movielens_cf.artifacts import write_json
from movielens_cf.baselines import BayesianPopularity, BiasBaseline
from movielens_cf.data import dataset_profile, load_movielens
from movielens_cf.evaluation import (
    ranking_metrics,
    rating_metrics,
    recommendation_stats,
    user_bootstrap_interval,
)
from movielens_cf.neighborhood import NeighborhoodCF
from movielens_cf.split import per_user_temporal_split


SEED = 42
K = 10
RELEVANCE = 4.0
CANDIDATES = [
    {"k": 20, "min_support": 5, "shrinkage": 10.0, "min_neighbors": 2},
    {"k": 40, "min_support": 5, "shrinkage": 25.0, "min_neighbors": 2},
    {"k": 80, "min_support": 10, "shrinkage": 50.0, "min_neighbors": 5},
]


def relevant_truth(holdout: pd.DataFrame, catalog: set[int]) -> dict[int, set[int]]:
    relevant = holdout.loc[
        (holdout["rating"] >= RELEVANCE) & holdout["movie_id"].isin(catalog)
    ]
    return {
        int(user_id): set(group["movie_id"].astype(int))
        for user_id, group in relevant.groupby("user_id", observed=True)
        if len(group)
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
    fitted: pd.DataFrame, holdout: pd.DataFrame, mode: str, params: dict
) -> tuple[NeighborhoodCF, dict, pd.DataFrame, dict[int, list[int]], dict]:
    started = time.perf_counter()
    model = NeighborhoodCF(mode=mode, block_size=384, **params).fit(fitted)
    fit_seconds = time.perf_counter() - started
    truth = relevant_truth(holdout, set(model.movie_ids.astype(int)))
    started = time.perf_counter()
    detailed = model.recommend_many(list(truth), n=K)
    recommend_seconds = time.perf_counter() - started
    recommendations = recommendation_ids(detailed)
    metrics, per_user = ranking_metrics(truth, recommendations, k=K)
    fallback = [item.used_fallback for values in detailed.values() for item in values]
    system = {
        "fitSeconds": fit_seconds,
        "recommendSeconds": recommend_seconds,
        "usersPerSecond": len(truth) / recommend_seconds if recommend_seconds else 0.0,
        "fallbackShare": float(np.mean(fallback)) if fallback else 0.0,
        "retainedNeighbors": int(model.similarity.nnz),
        "neighborArtifactMiB": float(
            (model.similarity.data.nbytes + model.similarity.indices.nbytes + model.similarity.indptr.nbytes)
            / 2**20
        ),
    }
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
            "ndcgAt10": float(group["ndcg_at_10"].mean()),
            "recallAt10": float(group["recall_at_10"].mean()),
        }
        for segment, group in frame.groupby("segment", observed=True)
    ]


def sample_payload(
    fitted: pd.DataFrame,
    movies: pd.DataFrame,
    user_model: NeighborhoodCF,
    item_model: NeighborhoodCF,
) -> dict:
    title = movies.set_index("movie_id")["title"].astype(str).to_dict()
    activity = fitted.groupby("user_id", observed=True).size().sort_values()
    positions = np.linspace(0.1, 0.9, 5)
    users = list(dict.fromkeys(int(activity.index[min(len(activity) - 1, round(position * (len(activity) - 1)))]) for position in positions))
    user_recs = user_model.recommend_many(users, n=K)
    item_recs = item_model.recommend_many(users, n=K)
    examples = []
    for user_id in users:
        history = fitted.loc[fitted["user_id"] == user_id].sort_values(
            ["rating", "timestamp"], ascending=[False, False]
        ).head(5)
        examples.append(
            {
                "user": f"Viewer {user_id}",
                "activity": int(activity.loc[user_id]),
                "history": [
                    {"title": title.get(int(row.movie_id), str(row.movie_id)), "rating": float(row.rating)}
                    for row in history.itertuples(index=False)
                ],
                "userCf": [
                    {
                        "title": title.get(item.movie_id, str(item.movie_id)),
                        "score": round(item.score, 3),
                        "neighbors": item.neighbor_count,
                        "fallback": item.used_fallback,
                    }
                    for item in user_recs[user_id]
                ],
                "itemCf": [
                    {
                        "title": title.get(item.movie_id, str(item.movie_id)),
                        "score": round(item.score, 3),
                        "neighbors": item.neighbor_count,
                        "fallback": item.used_fallback,
                    }
                    for item in item_recs[user_id]
                ],
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
                "seed": title.get(movie_id, str(movie_id)),
                "neighbors": [
                    {
                        "title": title.get(int(item_model.movie_ids[row.indices[position]]), "Unknown"),
                        "similarity": round(float(row.data[position]), 4),
                    }
                    for position in order
                ],
            }
        )
    return {"version": "movielens-samples-v1", "users": examples, "relatedItems": related}


def make_figures(profile: dict, models: list[dict], fitted: pd.DataFrame, output: Path) -> None:
    plt.style.use("dark_background")
    colors = ["#35d0e2", "#a78bfa", "#f7b955"]
    labels = [model["label"] for model in models]
    ndcg = [model["test"]["ndcg_at_10"] for model in models]
    recall = [model["test"]["recall_at_10"] for model in models]
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#07111f")
    ax.set_facecolor("#07111f")
    x = np.arange(len(labels)); width = 0.34
    ax.bar(x - width / 2, ndcg, width, label="NDCG@10", color=colors)
    ax.bar(x + width / 2, recall, width, label="Recall@10", color=colors, alpha=0.48)
    ax.set_xticks(x, labels); ax.set_ylim(0, max(ndcg + recall) * 1.25)
    ax.set_ylabel("Higher is better"); ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False); fig.tight_layout()
    fig.savefig(output / "model-ranking.svg", facecolor=fig.get_facecolor()); plt.close(fig)

    counts = fitted.groupby("movie_id", observed=True).size().sort_values(ascending=False).to_numpy()
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#07111f")
    ax.set_facecolor("#07111f")
    ax.plot(np.arange(1, len(counts) + 1), counts, color="#a78bfa", linewidth=2)
    ax.fill_between(np.arange(1, len(counts) + 1), counts, color="#a78bfa", alpha=0.15)
    ax.set_yscale("log"); ax.set_xlabel("Movies ordered by training popularity")
    ax.set_ylabel("Ratings (log scale)"); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(output / "popularity-long-tail.svg", facecolor=fig.get_facecolor()); plt.close(fig)


def run(data_dir: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    ratings, movies = load_movielens(data_dir)
    profile = dataset_profile(ratings, movies)
    train, validation, test = per_user_temporal_split(ratings)
    validation_results = []
    selected = {}
    for mode in ("user", "item"):
        candidates = []
        for params in CANDIDATES:
            model, metrics, _, _, system = evaluate_configuration(train, validation, mode, params)
            candidates.append({"params": params, "metrics": metrics, "system": system})
        candidates.sort(key=lambda value: (-value["metrics"]["ndcg_at_10"], -value["metrics"]["recall_at_10"], value["system"]["fitSeconds"]))
        selected[mode] = candidates[0]["params"]
        validation_results.append({"mode": mode, "candidates": candidates, "selected": selected[mode]})

    fitted = pd.concat([train, validation], ignore_index=True)
    final_models = []
    model_objects = {}
    final_per_user = {}
    final_recommendations = {}
    item_counts = fitted.groupby("movie_id", observed=True).size()
    for mode in ("user", "item"):
        model, rank, per_user, recommendations, system = evaluate_configuration(
            fitted, test, mode, selected[mode]
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
                "similarity": "Mean-centered cosine" if mode == "user" else "Adjusted cosine",
                "hyperparameters": selected[mode],
                "test": rank,
                "rating": rating_evaluation(model, test),
                "beyondAccuracy": beyond,
                "system": system,
                "segments": segment_metrics(per_user, fitted),
                "confidence95": {
                    "ndcgAt10": user_bootstrap_interval(per_user, "ndcg_at_10"),
                    "recallAt10": user_bootstrap_interval(per_user, "recall_at_10"),
                },
            }
        )

    truth = relevant_truth(test, set(item_counts.index.astype(int)))
    popularity = BayesianPopularity().fit(fitted)
    popular_recs = popularity_recommendations(popularity, fitted, list(truth))
    popular_rank, _ = ranking_metrics(truth, popular_recs, k=K)
    bias = BiasBaseline().fit(fitted)
    predictable = test.loc[test["user_id"].isin(bias.user_bias) & test["movie_id"].isin(bias.item_bias)]
    bias_predictions = [bias.predict(int(row.user_id), int(row.movie_id)) for row in predictable.itertuples(index=False)]
    baselines = {
        "bayesianPopularity": {
            "test": popular_rank,
            "beyondAccuracy": recommendation_stats(popular_recs, item_counts, len(item_counts)),
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
    }
    metadata = {
        "version": "movielens-cf-v1",
        "experimentCodeVersion": "movielens-cf-v1",
        "generatedAtUtc": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": "MovieLens 1M",
        "datasetUrl": "https://grouplens.org/datasets/movielens/1m/",
        "archiveMd5": "c4d9eecfca2ab87c1945afe126590906",
        "seed": SEED,
        "split": "Per-user temporal 80/10/10; refit train+validation for final test",
        "relevance": "rating >= 4",
        "candidatePolicy": "Full fitted catalog minus seen movies",
        "k": K,
        "runtime": runtime,
    }
    metrics = {
        **metadata,
        "evaluationPopulation": {
            "rankingUsers": len(truth),
            "testRatings": int(len(test)),
            "trainingRatings": int(len(fitted)),
        },
        "baselines": baselines,
        "models": final_models,
        "metricCaveat": "Offline explicit-rating results; they do not measure CTR, watch time, retention, revenue, or causal lift.",
    }
    write_json(output / "profile.json", {**metadata, "profile": profile})
    write_json(output / "metrics.json", metrics)
    write_json(output / "comparisons.json", {**metadata, "validation": validation_results, "models": final_models, "baselines": baselines})
    write_json(output / "samples.json", sample_payload(fitted, movies, model_objects["user"], model_objects["item"]))
    make_figures(profile, [{"label": "Popularity", "test": popular_rank}, *final_models], fitted, output)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("data") / "ml-1m")
    parser.add_argument("--output", type=Path, default=Path(__file__).parents[2] / "public" / "artifacts" / "movielens")
    args = parser.parse_args()
    metrics = run(args.data, args.output)
    for model in metrics["models"]:
        print(model["label"], model["test"], model["rating"], model["system"])


if __name__ == "__main__":
    main()
