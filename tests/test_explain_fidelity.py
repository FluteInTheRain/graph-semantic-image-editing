import numpy as np
import pytest

from mmxai.explain.fidelity import fidelity_minus, fidelity_plus, rank_drop
from mmxai.explain.intervention import mark_remaining_edges

# Small graph: image 0 contains objects 1 and 2; object 1 relates to object 3.
EDGE_INDEX = np.array(
    [
        [0, 0, 1],
        [1, 2, 3],
    ]
)
EDGE_WEIGHT = np.array([0.5, 0.3, 0.2])


def toy_score(remaining: np.ndarray) -> float:
    """Stand-in for the graph model: the image's score is the sum of the edges it still has."""
    return float(EDGE_WEIGHT[remaining].sum())


def test_fidelity_plus_on_a_hand_computed_example():
    assert fidelity_plus(full_score=1.0, score_without_explanation=0.2) == pytest.approx(0.8)


def test_fidelity_minus_on_a_hand_computed_example():
    assert fidelity_minus(full_score=1.0, score_with_only_explanation=0.9) == pytest.approx(0.1)


def test_fidelity_works_with_negative_scores():
    assert fidelity_plus(full_score=-0.1, score_without_explanation=-0.4) == pytest.approx(0.3)


def test_removing_explanation_edges_drops_the_score():
    full = toy_score(mark_remaining_edges(EDGE_INDEX, [], both_directions=False))
    without = toy_score(mark_remaining_edges(EDGE_INDEX, [(0, 1)], both_directions=False))

    assert fidelity_plus(full, without) == pytest.approx(0.5)


def test_removing_more_important_edges_gives_higher_fidelity_plus():
    full = toy_score(mark_remaining_edges(EDGE_INDEX, [], both_directions=False))
    without_strong = toy_score(mark_remaining_edges(EDGE_INDEX, [(0, 1)], both_directions=False))
    without_weak = toy_score(mark_remaining_edges(EDGE_INDEX, [(1, 3)], both_directions=False))

    assert fidelity_plus(full, without_strong) > fidelity_plus(full, without_weak)


def test_keeping_only_a_complete_explanation_gives_zero_fidelity_minus():
    full = toy_score(mark_remaining_edges(EDGE_INDEX, [], both_directions=False))
    only_explanation = toy_score(np.array([True, True, True]))

    assert fidelity_minus(full, only_explanation) == pytest.approx(0.0)


def test_rank_drop_is_positive_when_the_candidate_falls():
    assert rank_drop(rank_before=2, rank_after=5) == 3


def test_rank_drop_is_negative_when_the_candidate_rises():
    assert rank_drop(rank_before=4, rank_after=1) == -3


def test_rank_drop_rejects_ranks_below_one():
    with pytest.raises(ValueError, match="start at 1"):
        rank_drop(rank_before=0, rank_after=3)
