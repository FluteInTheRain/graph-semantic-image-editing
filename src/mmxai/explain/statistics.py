"""Statistics for comparing explanations, in plain numpy (no extra dependency)."""

from collections.abc import Mapping, Sequence

import numpy as np


def average_ranks(values: Sequence[float]) -> np.ndarray:
    """Rank values from 1 (smallest); tied values share the average of their ranks."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values))
    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and values[order[end + 1]] == values[order[start]]:
            end += 1
        ranks[order[start : end + 1]] = (start + end) / 2 + 1
        start = end + 1
    return ranks


def spearman(first: Sequence[float], second: Sequence[float]) -> float:
    """Rank correlation between two scorings of the same items (1 = same order)."""
    if len(first) != len(second) or len(first) < 2:
        raise ValueError("need two lists of the same length, with at least 2 items")
    first_ranks = average_ranks(first)
    second_ranks = average_ranks(second)
    if first_ranks.std() == 0 or second_ranks.std() == 0:
        raise ValueError("cannot correlate a list whose values are all equal")
    return float(np.corrcoef(first_ranks, second_ranks)[0, 1])


def wilcoxon_greater(differences: Sequence[float]) -> float:
    """One-sided Wilcoxon signed-rank p-value for "the differences tend to be positive".

    Zero differences are dropped. The p-value is exact: it counts, over every way of
    flipping signs, how often the positive rank sum is at least the observed one.
    """
    differences = np.asarray(differences, dtype=float)
    differences = differences[differences != 0]
    if len(differences) == 0:
        return 1.0

    # Ranks of tied values end in .5, so doubling them gives whole numbers to count with.
    doubled_ranks = (2 * average_ranks(np.abs(differences))).astype(int)
    observed = int(doubled_ranks[differences > 0].sum())

    # ways[s] = number of sign choices whose positive doubled-rank sum is s.
    ways = [1] + [0] * int(doubled_ranks.sum())
    for rank in doubled_ranks:
        for total in range(len(ways) - 1, rank - 1, -1):
            ways[total] += ways[total - rank]

    return sum(ways[observed:]) / 2 ** len(differences)


def mean_and_std(values: Sequence[float]) -> tuple[float, float]:
    """Mean and sample standard deviation (ddof=1, ADR-0011)."""
    if len(values) < 2:
        raise ValueError("need at least 2 values for a sample standard deviation")
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1))


def choose_explainer(validation_scores: Mapping[str, Sequence[float]]) -> str:
    """Pick the explainer with the highest mean score on validation queries.

    Only validation scores may be passed here (PDF section 6): the choice is then run once
    on test. Ties go to the first name in alphabetical order, so the choice is repeatable.
    """
    if not validation_scores:
        raise ValueError("no explainers to choose from")
    best_name = None
    best_mean = -np.inf
    for name in sorted(validation_scores):
        mean = float(np.mean(validation_scores[name]))
        if mean > best_mean:
            best_name, best_mean = name, mean
    return best_name
