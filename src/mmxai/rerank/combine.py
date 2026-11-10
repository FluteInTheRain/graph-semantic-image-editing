"""Combine the CLIP cosine and the graph score, and choose alpha on val (ADR-0016, ADR-0046).

    final = alpha * cos(q, x_i) + (1 - alpha) * graph_score(q, i)

Both scores are cosines, so they share one scale and can be added directly.
Numpy only.
"""

from collections.abc import Sequence

import numpy as np

from mmxai.eval.metrics import compute_metrics, first_relevant_rank


def final_scores(cosine: np.ndarray, graph: np.ndarray, alpha: float) -> np.ndarray:
    """Weighted sum of the two scores; any shape, as long as both match."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if cosine.shape != graph.shape:
        raise ValueError(f"cosine has shape {cosine.shape} but graph has {graph.shape}")
    return alpha * cosine + (1 - alpha) * graph


def rerank(candidate_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Sort each query's candidates by score, best first.

    A stable sort on the negated score keeps CLIP's order between equal scores, so the
    result is the same on every run.
    """
    order = np.argsort(-scores, axis=1, kind="stable")
    return np.take_along_axis(candidate_ids, order, axis=1)


def rerank_metrics(
    candidate_ids: np.ndarray,
    cosine: np.ndarray,
    graph: np.ndarray,
    alpha: float,
    relevant: Sequence[set[int]],
    recall_ks: Sequence[int],
) -> tuple[dict, list[int | None]]:
    """Metrics and per-query ranks after re-ranking every query's candidates with alpha."""
    reranked = rerank(candidate_ids, final_scores(cosine, graph, alpha))
    ranks = []
    for row, relevant_ids in zip(reranked, relevant, strict=True):
        ranks.append(first_relevant_rank(row.tolist(), relevant_ids))
    top_k = candidate_ids.shape[1]
    return compute_metrics(ranks, recall_ks, top_k), ranks


def choose_alpha(
    candidate_ids: np.ndarray,
    cosine: np.ndarray,
    graph: np.ndarray,
    relevant: Sequence[set[int]],
    alpha_grid: Sequence[float],
    recall_ks: Sequence[int],
) -> tuple[float, dict[float, dict]]:
    """Return the alpha with the best MRR, and the metrics of every alpha in the grid.

    Decision (ADR-0016, ADR-0046): alpha is chosen on VAL only, by MRR@top_k.
      Why: MRR uses the whole rank, while one Recall@K ignores moves below K.
      Proposed by Claude Code, 2026-11-10; not yet decided by the team.
    Decision: on a tie, keep the LARGER alpha.
      Why: it trusts CLIP more, so the graph is only credited with a gain it really gave.
      Proposed by Claude Code, 2026-11-10; not yet decided by the team.
    """
    if not alpha_grid:
        raise ValueError("alpha_grid is empty")
    top_k = candidate_ids.shape[1]
    mrr_name = f"MRR@{top_k}"
    metrics_by_alpha = {}
    best_alpha = None
    for alpha in sorted(alpha_grid):
        metrics, _ = rerank_metrics(candidate_ids, cosine, graph, alpha, relevant, recall_ks)
        metrics_by_alpha[float(alpha)] = metrics
        # >= with the grid sorted ascending: a later (larger) alpha wins ties.
        if best_alpha is None or metrics[mrr_name] >= metrics_by_alpha[best_alpha][mrr_name]:
            best_alpha = float(alpha)
    return best_alpha, metrics_by_alpha
