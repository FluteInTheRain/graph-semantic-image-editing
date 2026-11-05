"""Recall@K and MRR for text-to-image retrieval (PDF section 7, ADR-0043).

A query's result is reduced to one number: the 1-based rank of the first
relevant image, or None when no relevant image is among the candidates.
"""

from collections.abc import Sequence


def first_relevant_rank(retrieved_ids: Sequence[int], relevant_ids: set[int]) -> int | None:
    """Return the 1-based position of the first relevant id, or None if there is none."""
    for position, image_id in enumerate(retrieved_ids, start=1):
        if image_id in relevant_ids:
            return position
    return None


def recall_at_k(ranks: Sequence[int | None], k: int) -> float:
    """Fraction of queries with a relevant image in the first k results."""
    if not ranks:
        raise ValueError("recall needs at least one query")
    hits = 0
    for rank in ranks:
        if rank is not None and rank <= k:
            hits += 1
    return hits / len(ranks)


def mean_reciprocal_rank(ranks: Sequence[int | None]) -> float:
    """Mean of 1/rank; a query with no relevant candidate adds 0."""
    if not ranks:
        raise ValueError("MRR needs at least one query")
    total = 0.0
    for rank in ranks:
        if rank is not None:
            total += 1.0 / rank
    return total / len(ranks)


def compute_metrics(ranks: Sequence[int | None], recall_ks: Sequence[int], top_k: int) -> dict:
    """Return every reported metric. MRR is named @top_k because ranks stop there."""
    for k in recall_ks:
        if k > top_k:
            raise ValueError(f"cannot measure R@{k} from top_k={top_k} candidates")

    metrics = {}
    for k in recall_ks:
        metrics[f"R@{k}"] = recall_at_k(ranks, k)
    metrics[f"MRR@{top_k}"] = mean_reciprocal_rank(ranks)
    metrics["num_queries"] = len(ranks)
    return metrics
