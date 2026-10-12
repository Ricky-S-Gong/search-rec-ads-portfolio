"""Bounded exact retrieval with deterministic original-ID tie breaking."""

import numpy as np


def exact_topk(users, items, item_ids, request_times, eligible_after, k=100, item_block=2048):
    """Return aligned [B,K] IDs/scores; -1/-inf explicitly mark unavailable slots."""
    users, items = np.asarray(users), np.asarray(items)
    ids, times, after = map(np.asarray, (item_ids, request_times, eligible_after))
    if users.ndim != 2 or items.ndim != 2 or users.shape[1] != items.shape[1]:
        raise ValueError("Vectors must have matching dimensions")
    if len(ids) != len(items) or after.shape != ids.shape or times.shape != (len(users),):
        raise ValueError("Misaligned IDs, vectors or timeline")
    if ids.dtype.kind not in "iu" or np.any(ids < 0) or len(np.unique(ids)) != len(ids):
        raise ValueError("Candidate IDs must be unique nonnegative integers")
    if not np.isfinite(users).all() or not np.isfinite(items).all() or k < 1 or item_block < 1:
        raise ValueError("Finite vectors and positive block/K required")
    best_ids = np.full((len(users), k), -1, dtype=np.int64)
    best_scores = np.full((len(users), k), -np.inf)
    for start in range(0, len(ids), item_block):
        stop = start + item_block
        scores = users @ items[start:stop].T
        scores = np.where(after[None, start:stop] <= times[:, None], scores, -np.inf)
        block_ids = np.broadcast_to(ids[None, start:stop], scores.shape)
        candidates = np.concatenate([best_ids, block_ids], axis=1)
        scores = np.concatenate([best_scores, scores], axis=1)
        # A stable full block sort makes the boundary tie rule exact.
        order = np.lexsort((candidates, -scores), axis=1)[:, :k]
        best_scores = np.take_along_axis(scores, order, axis=1)
        best_ids = np.take_along_axis(candidates, order, axis=1)
        best_ids[~np.isfinite(best_scores)] = -1
    return best_ids, best_scores


def ranking_metrics(predictions, targets, covered):
    """Single-positive per-request Recall/HitRate and NDCG; uncovered targets miss."""
    predictions, targets, covered = map(np.asarray, (predictions, targets, covered))
    if (
        predictions.ndim != 2
        or targets.shape != covered.shape
        or targets.shape != (len(predictions),)
    ):
        raise ValueError("Misaligned metric inputs")
    if not len(targets):
        return {
            "requests": 0,
            "recall_at_100": None,
            "ndcg_at_100": None,
            "target_coverage": None,
            "covered_recall_at_100": None,
            "covered_ndcg_at_100": None,
            "insufficient_k_rate": None,
        }
    matches = (predictions == targets[:, None]) & covered[:, None]
    hit = matches.any(axis=1)
    ranks = matches.argmax(axis=1) + 1
    gain = np.where(hit, 1 / np.log2(ranks + 1), 0)
    return {
        "requests": len(targets),
        "recall_at_100": float(hit.mean()),
        "ndcg_at_100": float(gain.mean()),
        "target_coverage": float(covered.mean()),
        "covered_recall_at_100": float(hit[covered].mean()) if covered.any() else None,
        "covered_ndcg_at_100": float(gain[covered].mean()) if covered.any() else None,
        "insufficient_k_rate": float(
            ((predictions >= 0).sum(axis=1) < predictions.shape[1]).mean()
        ),
    }
