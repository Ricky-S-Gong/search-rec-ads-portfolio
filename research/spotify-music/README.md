# Spotify content recommender

This project runs on CPU. It does not require a Spotify API key.

```bash
uv sync
uv run python research/spotify-music/download_data.py
uv run python research/spotify-music/run_experiment.py \
  --data research/spotify-music/data/data.csv
uv run pytest
```

The experiment writes versioned metrics, a PCA visualization, and the compact
browser-demo dataset to `public/artifacts/spotify/`. Raw Kaggle files remain
gitignored and are never redistributed.

The metrics are content-coherence and systems proxies. They are not user
relevance metrics because the source dataset has no user interactions.
