"""Fidelity: does removing or keeping only the explanation change the score as expected?"""


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
