from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = {"user_id", "video_id", "date", "time_ms", "tab", "long_view", "is_rand"}


def build_histories(events: pd.DataFrame, limit: int = 50) -> pd.DataFrame:
    """Create a bounded, strictly-prior positive-view history for each request."""
    required = {"request_id", "user_id", "item_id", "request_time_ms", "long_view", "is_random"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"missing required event columns: {sorted(missing)}")
    if limit < 1:
        raise ValueError("history limit must be positive")

    positives = events.loc[(events["long_view"]) & (~events["is_random"])].sort_values(
        ["user_id", "request_time_ms"], kind="stable"
    )
    output: list[pd.DataFrame] = []
    for _, user_events in positives.groupby("user_id", sort=False):
        times = user_events["request_time_ms"].to_numpy()
        item_ids = user_events["item_id"].astype("int64").tolist()
        previous_counts = times.searchsorted(times, side="left")
        histories = [item_ids[max(0, count - limit):count] for count in previous_counts]
        output.append(
            pd.DataFrame(
                {
                    "request_id": user_events["request_id"].to_numpy(),
                    "user_id": user_events["user_id"].to_numpy(),
                    "history_item_ids": histories,
                }
            )
        )
    return pd.concat(output, ignore_index=True) if output else pd.DataFrame(
        columns=["request_id", "user_id", "history_item_ids"]
    )


def normalize_events(raw: pd.DataFrame, source: str) -> pd.DataFrame:
    """Map one official KuaiRand log file to the shared training contract."""
    missing = REQUIRED_COLUMNS.difference(raw.columns)
    if missing:
        raise ValueError(f"missing required KuaiRand columns: {sorted(missing)}")
    events = pd.DataFrame(
        {
            "request_id": source + ":" + raw.index.astype(str),
            "user_id": raw["user_id"].astype("int64"),
            "item_id": raw["video_id"].astype("int64"),
            "request_time_ms": raw["time_ms"].astype("int64"),
            "tab": raw["tab"].astype("int64"),
            "long_view": raw["long_view"].astype(bool),
            "is_random": raw["is_rand"].astype(bool),
        }
    )
    if not events["request_id"].is_unique:
        raise ValueError(f"duplicate request IDs in {source}")
    return events


def split_events(events: pd.DataFrame) -> pd.DataFrame:
    """Apply the frozen time protocol; random exposure remains diagnostic-only."""
    result = events.copy()
    time = pd.to_datetime(result["request_time_ms"], unit="ms", utc=True)
    result["split"] = "test"
    result.loc[time < pd.Timestamp("2022-04-15", tz="UTC"), "split"] = "warmup"
    result.loc[(time >= pd.Timestamp("2022-04-15", tz="UTC")) & (time < pd.Timestamp("2022-04-29", tz="UTC")), "split"] = "train"
    result.loc[(time >= pd.Timestamp("2022-04-29", tz="UTC")) & (time < pd.Timestamp("2022-05-04", tz="UTC")), "split"] = "validation"
    result.loc[result["is_random"], "split"] = "random"
    return result
