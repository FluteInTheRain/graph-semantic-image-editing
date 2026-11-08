"""Fidelity: does removing or keeping only the explanation change the score as expected?"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from mmxai.explain.intervention import mark_remaining_edges


def fidelity_plus(full_score: float, score_without_explanation: float) -> float:
    """How much the score drops when the explanation edges are removed.

    Larger is better: a faithful explanation is necessary for the score.
    """
    return full_score - score_without_explanation


def fidelity_minus(full_score: float, score_with_only_explanation: float) -> float:
    """How much the score drops when only the explanation edges are kept.

    Closer to 0 is better: a faithful explanation is enough to keep the score.
    """
    return full_score - score_with_only_explanation


def rank_drop(rank_before: int, rank_after: int) -> int:
    """How many places a candidate falls after an intervention (1 = top rank).

    Positive means it fell, negative means it rose.
    """
    if rank_before < 1 or rank_after < 1:
        raise ValueError(f"ranks start at 1, got {rank_before} and {rank_after}")
    return rank_after - rank_before


def rank_of(scores: np.ndarray, candidate: int) -> int:
    """Rank of one candidate among all candidates (1 = highest score).

    Ties share the better rank, so a tie never makes a candidate look like it fell.
    """
    return 1 + int((scores > scores[candidate]).sum())


@dataclass(frozen=True)
class FidelityResult:
    """Fidelity of one explanation for one candidate."""

    fidelity_plus: float
    fidelity_minus: float
    rank_drop: int


def evaluate_fidelity(
    edge_index: np.ndarray,
    explanation_edges: Iterable[tuple[int, int]],
    both_directions: bool,
    graph_scores: Callable[[np.ndarray], np.ndarray],
    cosine: np.ndarray,
    alpha: float,
    candidate: int,
) -> FidelityResult:
    """Score the graph three times (full, without the explanation, only the explanation)
    and report how the candidate's graph score and rank change.

    graph_scores takes the edges that remain and returns one graph score per candidate.
    The rank uses the final score alpha * cosine + (1 - alpha) * graph (ADR-0016).
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    all_edges = np.ones(edge_index.shape[1], dtype=bool)
    without_explanation = mark_remaining_edges(edge_index, explanation_edges, both_directions)
    # Keeping only the explanation is exactly the set of edges the removal took away.
    only_explanation = ~without_explanation

    full = graph_scores(all_edges)
    without = graph_scores(without_explanation)
    only = graph_scores(only_explanation)

    final_before = alpha * cosine + (1 - alpha) * full
    final_after = alpha * cosine + (1 - alpha) * without

    return FidelityResult(
        fidelity_plus=fidelity_plus(full[candidate], without[candidate]),
        fidelity_minus=fidelity_minus(full[candidate], only[candidate]),
        rank_drop=rank_drop(rank_of(final_before, candidate), rank_of(final_after, candidate)),
    )
