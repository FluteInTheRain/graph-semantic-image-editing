"""Recall@K and MRR for text-to-image retrieval (PDF section 7, ADR-0043).

A query's result is reduced to one number: the 1-based rank of the first
relevant image, or None when no relevant image is among the candidates.

Decision (ADR-0043): with several relevant images, the first one found decides the rank.
  Why: the user is served as soon as one correct image appears.
  Decided by: FluteInTheRain, 2026-11-05 (proposed by Claude Code in ADR-0043).

Decision (ADR-0043): ranks stop at top_k (100); report MRR@100, a query with no
relevant image in the top 100 adds 0.
  Why: CLIP + Graph only re-ranks those same 100 candidates, so both models are
  scored under the same cut.
  Decided by: FluteInTheRain, 2026-11-05 (proposed by Claude Code in ADR-0043;
  top_k = 100 taken from FluteInTheRain's template configs/clip_graph_vg.yaml).

Decision (spec 2026-11-13): nDCG@10 with binary relevance (an image is relevant or not).
  Why: Recall and MRR only look at the first relevant image; nDCG also rewards
  putting every relevant image high in the list.
  Decided by: FluteInTheRain, 2026-11-13 (proposed by Claude Code in the level-1
  evaluation extensions spec).
"""

import math
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


def ndcg_at_k(retrieved_ids: Sequence[int], relevant_ids: set[int], k: int) -> float:
    """nDCG@k with binary relevance: DCG of the list divided by the best possible DCG."""
    dcg = 0.0
    for position, image_id in enumerate(retrieved_ids[:k], start=1):
        if image_id in relevant_ids:
            dcg += 1.0 / math.log2(position + 1)

    # Ideal list: relevant ids fill the first positions. Relevant ids missing from
    # the retrieved list still count, so a poor list cannot score 1.
    ideal_hits = min(len(relevant_ids), k)
    idcg = 0.0
    for position in range(1, ideal_hits + 1):
        idcg += 1.0 / math.log2(position + 1)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def ranked_metrics(
    rankings: Sequence[Sequence[int]],
    relevant: Sequence[set[int]],
    recall_ks: Sequence[int],
    top_k: int,
    ndcg_k: int,
) -> dict:
    """Metrics from one ranked list per query: compute_metrics keys plus nDCG@ndcg_k."""
    if len(rankings) != len(relevant):
        raise ValueError("need one relevant set per ranked list")

    ranks = []
    ndcg_total = 0.0
    for retrieved_ids, relevant_ids in zip(rankings, relevant, strict=True):
        ranks.append(first_relevant_rank(retrieved_ids, relevant_ids))
        ndcg_total += ndcg_at_k(retrieved_ids, relevant_ids, ndcg_k)

    metrics = compute_metrics(ranks, recall_ks, top_k)
    metrics[f"nDCG@{ndcg_k}"] = ndcg_total / len(rankings)
    return metrics
