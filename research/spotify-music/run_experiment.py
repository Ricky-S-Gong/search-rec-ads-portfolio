"""Reproduce the Spotify content-retrieval experiment and web artifacts."""

from __future__ import annotations

import argparse
import ast
import json
import platform
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize

RANDOM_SEED = 42
FEATURES = [
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "liveness",
    "loudness",
    "speechiness",
    "tempo",
    "valence",
]


def artist_name(raw: object) -> str:
    """Return a readable primary artist from the dataset's list-like field."""
    text = str(raw)
    try:
        values = ast.literal_eval(text)
        if isinstance(values, list) and values:
            return str(values[0])
    except (SyntaxError, ValueError):
        pass
    return text.strip("[]'\"")


def prepare_tracks(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"id", "name", "artists", "year", "popularity", *FEATURES}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    tracks = frame.drop_duplicates("id", keep="first").copy()
    for column in [*FEATURES, "year", "popularity"]:
        tracks[column] = pd.to_numeric(tracks[column], errors="coerce")
        tracks[column] = tracks[column].replace([np.inf, -np.inf], np.nan)
        tracks[column] = tracks[column].fillna(tracks[column].median())
    tracks["artist"] = tracks["artists"].map(artist_name)
    return tracks.reset_index(drop=True)


def stratified_demo(tracks: pd.DataFrame, size: int, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Sample across decade and popularity bands, then put recognizable items first."""
    if len(tracks) <= size:
        return tracks.sort_values(["popularity", "year"], ascending=False).reset_index(drop=True)
    work = tracks.copy()
    work["decade"] = (work["year"] // 10 * 10).astype(int)
    work["pop_band"] = pd.qcut(work["popularity"].rank(method="first"), 5, labels=False)
    groups = list(work.groupby(["decade", "pop_band"], observed=True))
    per_group = max(1, size // len(groups))
    chosen = []
    for _, group in groups:
        chosen.extend(group.sample(min(len(group), per_group), random_state=seed).index.tolist())
    chosen = list(dict.fromkeys(chosen))
    remaining = size - len(chosen)
    if remaining > 0:
        pool = work.drop(index=chosen)
        chosen.extend(pool.sample(remaining, random_state=seed).index.tolist())
    demo = tracks.loc[chosen[:size]].copy()
    return demo.sort_values(["popularity", "year"], ascending=False).reset_index(drop=True)


def exact_top_k(matrix: np.ndarray, query_index: int, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
    scores = matrix @ matrix[query_index]
    scores[query_index] = -np.inf
    indices = np.argpartition(-scores, k)[:k]
    order = np.argsort(-scores[indices])
    return indices[order], scores[indices][order]


def cluster_top_k(
    matrix: np.ndarray, labels: np.ndarray, query_index: int, k: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    candidates = np.flatnonzero(labels == labels[query_index])
    scores = matrix[candidates] @ matrix[query_index]
    scores[candidates == query_index] = -np.inf
    take = min(k, max(1, len(candidates) - 1))
    local = np.argpartition(-scores, take - 1)[:take]
    order = np.argsort(-scores[local])
    return candidates[local][order], scores[local][order]


def intra_list_diversity(matrix: np.ndarray, indices: np.ndarray) -> float:
    if len(indices) < 2:
        return 0.0
    similarities = matrix[indices] @ matrix[indices].T
    triangle = similarities[np.triu_indices(len(indices), k=1)]
    return float(1 - triangle.mean())


def genre_label_coverage(data_path: Path, tracks: pd.DataFrame) -> float:
    genre_path = data_path.with_name("data_w_genres.csv")
    if not genre_path.exists():
        return 0.0
    genre_frame = pd.read_csv(genre_path, usecols=["artists", "genres"])
    known = set(genre_frame.loc[genre_frame["genres"] != "[]", "artists"].astype(str).str.casefold())
    return float(tracks["artist"].astype(str).str.casefold().isin(known).mean())


def run(data_path: Path, output_dir: Path, demo_size: int = 5000) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(data_path)
    tracks = prepare_tracks(raw)
    values = tracks[FEATURES].to_numpy(dtype=np.float32)

    minmax = MinMaxScaler()
    minmax_values = minmax.fit_transform(values).astype(np.float32)
    cosine_matrix = normalize(minmax_values).astype(np.float32)

    standard_values = StandardScaler().fit_transform(values).astype(np.float32)
    knn = NearestNeighbors(n_neighbors=11, metric="euclidean", algorithm="auto").fit(standard_values)

    clusters = 64 if len(tracks) >= 6400 else max(4, len(tracks) // 100)
    kmeans = MiniBatchKMeans(
        n_clusters=clusters,
        random_state=RANDOM_SEED,
        batch_size=4096,
        n_init=3,
    ).fit(minmax_values)
    labels = kmeans.labels_

    rng = np.random.default_rng(RANDOM_SEED)
    query_count = min(40, len(tracks))
    query_indices = rng.choice(len(tracks), size=query_count, replace=False)
    exact_times, cluster_times, recalls, diversities, recommended = [], [], [], [], set()
    neighborhood_distances = []
    rec_popularity = []
    for query_index in query_indices:
        started = time.perf_counter()
        exact_indices, exact_scores = exact_top_k(cosine_matrix, int(query_index))
        exact_times.append((time.perf_counter() - started) * 1000)

        started = time.perf_counter()
        candidate_indices, _ = cluster_top_k(cosine_matrix, labels, int(query_index))
        cluster_times.append((time.perf_counter() - started) * 1000)

        recalls.append(len(set(exact_indices) & set(candidate_indices)) / len(exact_indices))
        diversities.append(intra_list_diversity(cosine_matrix, exact_indices))
        neighborhood_distances.append(float(1 - exact_scores.mean()))
        recommended.update(exact_indices.tolist())
        rec_popularity.extend(tracks.iloc[exact_indices]["popularity"].tolist())

    silhouette_rows = min(2000, len(tracks))
    silhouette_indices = rng.choice(len(tracks), size=silhouette_rows, replace=False)
    cluster_silhouette = float(
        silhouette_score(minmax_values[silhouette_indices], labels[silhouette_indices])
    )

    knn_started = time.perf_counter()
    knn.kneighbors(standard_values[query_indices[: min(10, query_count)]])
    knn_latency = (time.perf_counter() - knn_started) * 1000 / min(10, query_count)

    demo = stratified_demo(tracks, min(demo_size, len(tracks)))
    demo_features = minmax.transform(demo[FEATURES].to_numpy(dtype=np.float32))
    demo_tracks = [
        {
            "id": str(row.id),
            "name": str(row.name),
            "artist": str(row.artist),
            "year": int(row.year),
            "popularity": int(round(row.popularity)),
            "features": [round(float(value), 4) for value in vector],
        }
        for row, vector in zip(demo.itertuples(index=False), demo_features, strict=True)
    ]
    demo_artifact = {
        "version": "spotify-demo-v1",
        "featureNames": FEATURES,
        "tracks": demo_tracks,
    }
    (output_dir / "demo-tracks.json").write_text(
        json.dumps(demo_artifact, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )

    metrics = {
        "version": "spotify-experiment-v1",
        "randomSeed": RANDOM_SEED,
        "runtime": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikitLearn": sklearn.__version__,
            "compute": "CPU",
        },
        "dataset": {
            "source": "vatsalmavani/spotify-dataset",
            "rawRows": int(len(raw)),
            "deduplicatedRows": int(len(tracks)),
            "demoRows": int(len(demo)),
            "genreLabelCoverage": round(genre_label_coverage(data_path, tracks), 6),
        },
        "evaluation": {
            "queryCount": query_count,
            "exactLatencyMs": round(float(np.median(exact_times)), 4),
            "clusterLatencyMs": round(float(np.median(cluster_times)), 4),
            "euclideanKnnLatencyMs": round(float(knn_latency), 4),
            "clusterRecallAt10": round(float(np.mean(recalls)), 6),
            "diversityAt10": round(float(np.mean(diversities)), 6),
            "coverageAt10": round(len(recommended) / len(tracks), 6),
            "neighborhoodDistanceAt10": round(float(np.mean(neighborhood_distances)), 6),
            "recommendedPopularityMean": round(float(np.mean(rec_popularity)), 4),
            "catalogPopularityMean": round(float(tracks["popularity"].mean()), 4),
            "clusterSilhouette": round(cluster_silhouette, 6),
        },
        "metricCaveat": "Proxy and systems metrics only; the dataset has no user relevance labels.",
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    plot_rows = min(2500, len(tracks))
    plot_indices = rng.choice(len(tracks), size=plot_rows, replace=False)
    projection = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(
        minmax_values[plot_indices]
    )
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#07111f")
    ax.set_facecolor("#07111f")
    scatter = ax.scatter(
        projection[:, 0], projection[:, 1], c=labels[plot_indices], cmap="twilight",
        s=7, alpha=0.65, linewidths=0,
    )
    scatter.set_rasterized(True)
    ax.set_title("Spotify audio-feature space · PCA for visualization only", color="#edf5ff", loc="left")
    ax.set_xlabel("Principal component 1", color="#9cb0c8")
    ax.set_ylabel("Principal component 2", color="#9cb0c8")
    ax.tick_params(colors="#9cb0c8")
    for spine in ax.spines.values():
        spine.set_color("#29435f")
    fig.tight_layout()
    fig.savefig(output_dir / "feature-space.svg", format="svg", transparent=True)
    plt.close(fig)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Path to Kaggle data.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("public/artifacts/spotify"),
        help="Directory for generated web artifacts",
    )
    parser.add_argument("--demo-size", type=int, default=5000)
    args = parser.parse_args()
    metrics = run(args.data, args.output, args.demo_size)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
