from __future__ import annotations

import gzip
import json
import sqlite3
from collections.abc import Iterable, Iterator
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_COLUMNS = [
    "user_id",
    "item_id",
    "rating",
    "is_read",
    "is_reviewed",
    "event_time",
]


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes"}
    return bool(value)


def _clean_record(record: dict) -> dict | None:
    user_id = str(record.get("user_id", "")).strip()
    item_id = str(record.get("book_id", record.get("item_id", ""))).strip()
    try:
        rating = int(record.get("rating", 0))
    except (TypeError, ValueError):
        return None
    if not user_id or not item_id or not 0 <= rating <= 5:
        return None
    raw_time = (
        record.get("date_updated")
        or record.get("read_at")
        or record.get("date_added")
        or record.get("event_time")
    )
    try:
        event_time = pd.Timestamp(datetime.strptime(str(raw_time), "%a %b %d %H:%M:%S %z %Y"))
    except (TypeError, ValueError):
        event_time = pd.to_datetime(raw_time, utc=True, errors="coerce")
    if pd.isna(event_time):
        return None
    event_time = pd.Timestamp(event_time).tz_convert("UTC")
    is_read = _as_bool(record.get("is_read", False))
    is_reviewed = _as_bool(record.get("is_reviewed", False))
    return {
        "user_id": user_id,
        "item_id": item_id,
        "rating": rating,
        "is_read": is_read,
        "is_reviewed": is_reviewed,
        "event_time": event_time,
        "_information_score": int(rating > 0) + int(is_read) + int(is_reviewed),
    }


def normalize_interactions(records: Iterable[dict]) -> pd.DataFrame:
    """Clean Goodreads records and deterministically deduplicate user-book pairs."""
    rows = []
    for record in records:
        cleaned = _clean_record(record)
        if cleaned is not None:
            rows.append(cleaned)
    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(
        ["user_id", "item_id", "_information_score", "event_time"],
        ascending=[True, True, False, False],
        kind="stable",
    )
    frame = frame.drop_duplicates(["user_id", "item_id"], keep="first")
    return frame[OUTPUT_COLUMNS].sort_values(["user_id", "item_id"], kind="stable").reset_index(
        drop=True
    )


def iterative_k_core(
    interactions: pd.DataFrame,
    *,
    min_user_interactions: int,
    min_user_ratings: int,
    min_item_interactions: int,
    min_item_ratings: int,
) -> pd.DataFrame:
    """Repeat user/item support filtering until the graph reaches a fixed point."""
    frame = interactions.copy()
    while not frame.empty:
        before = len(frame)
        explicit = frame["rating"].gt(0)
        user_total = frame.groupby("user_id", observed=True).size()
        user_rated = explicit.groupby(frame["user_id"], observed=True).sum()
        valid_users = user_total.index[
            (user_total >= min_user_interactions)
            & (user_rated.reindex(user_total.index, fill_value=0) >= min_user_ratings)
        ]
        frame = frame[frame["user_id"].isin(valid_users)]
        if frame.empty:
            break
        explicit = frame["rating"].gt(0)
        item_total = frame.groupby("item_id", observed=True).size()
        item_rated = explicit.groupby(frame["item_id"], observed=True).sum()
        valid_items = item_total.index[
            (item_total >= min_item_interactions)
            & (item_rated.reindex(item_total.index, fill_value=0) >= min_item_ratings)
        ]
        frame = frame[frame["item_id"].isin(valid_items)]
        if len(frame) == before:
            break
    return frame.reset_index(drop=True)


def sample_users(
    interactions: pd.DataFrame, max_users: int | None, seed: int
) -> pd.DataFrame:
    """Select at most max_users from a sorted population using a fixed RNG seed."""
    users = np.array(sorted(interactions["user_id"].unique()), dtype=object)
    if max_users is None or len(users) <= max_users:
        return interactions.copy().reset_index(drop=True)
    selected = np.random.default_rng(seed).choice(users, size=max_users, replace=False)
    return interactions[interactions["user_id"].isin(set(selected))].reset_index(drop=True)


def encode_ids(interactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Map stable sorted string identifiers to contiguous integer indices."""
    user_ids = sorted(interactions["user_id"].astype(str).unique())
    item_ids = sorted(interactions["item_id"].astype(str).unique())
    users = pd.DataFrame({"user_id": user_ids, "user_idx": np.arange(len(user_ids), dtype=np.int32)})
    items = pd.DataFrame({"item_id": item_ids, "item_idx": np.arange(len(item_ids), dtype=np.int32)})
    user_map = dict(zip(users["user_id"], users["user_idx"], strict=True))
    item_map = dict(zip(items["item_id"], items["item_idx"], strict=True))
    encoded = interactions.copy()
    encoded["user_idx"] = encoded["user_id"].map(user_map).astype("int32")
    encoded["item_idx"] = encoded["item_id"].map(item_map).astype("int32")
    return encoded, users, items


def iter_json_gzip(path: Path) -> Iterator[dict]:
    """Yield one JSON object at a time without expanding the gzip file on disk."""
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on line {line_number}") from error


def stage_to_sqlite(raw_path: Path, database_path: Path, batch_size: int = 10_000) -> int:
    """Stream, clean, and globally deduplicate the large source into SQLite."""
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    connection.execute("DROP TABLE IF EXISTS interactions")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS interactions (
            user_id TEXT NOT NULL,
            item_id TEXT NOT NULL,
            rating INTEGER NOT NULL,
            is_read INTEGER NOT NULL,
            is_reviewed INTEGER NOT NULL,
            event_time TEXT NOT NULL,
            information_score INTEGER NOT NULL,
            PRIMARY KEY (user_id, item_id)
        ) WITHOUT ROWID
        """
    )
    upsert = """
        INSERT INTO interactions VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(user_id, item_id) DO UPDATE SET
            rating=excluded.rating,
            is_read=excluded.is_read,
            is_reviewed=excluded.is_reviewed,
            event_time=excluded.event_time,
            information_score=excluded.information_score
        WHERE excluded.information_score > interactions.information_score
           OR (excluded.information_score = interactions.information_score
               AND excluded.event_time > interactions.event_time)
    """
    batch: list[tuple] = []
    for record in iter_json_gzip(raw_path):
        cleaned = _clean_record(record)
        if cleaned is None:
            continue
        batch.append(
            (
                cleaned["user_id"],
                cleaned["item_id"],
                cleaned["rating"],
                int(cleaned["is_read"]),
                int(cleaned["is_reviewed"]),
                cleaned["event_time"].isoformat(),
                cleaned["_information_score"],
            )
        )
        if len(batch) >= batch_size:
            connection.executemany(upsert, batch)
            connection.commit()
            batch.clear()
    if batch:
        connection.executemany(upsert, batch)
        connection.commit()
    count = int(connection.execute("SELECT COUNT(*) FROM interactions").fetchone()[0])
    connection.close()
    return count


def load_staged(database_path: Path) -> pd.DataFrame:
    """Load the deduplicated staging table after disk-backed ingestion."""
    with sqlite3.connect(database_path) as connection:
        frame = pd.read_sql_query(
            "SELECT user_id, item_id, rating, is_read, is_reviewed, event_time FROM interactions",
            connection,
        )
    frame["is_read"] = frame["is_read"].astype(bool)
    frame["is_reviewed"] = frame["is_reviewed"].astype(bool)
    frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True)
    return frame
