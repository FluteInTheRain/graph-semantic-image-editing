import numpy as np
import pytest

from mmxai.explain.fidelity import (
    evaluate_fidelity,
    fidelity_minus,
    fidelity_plus,
    rank_drop,
    rank_of,
)
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


# Three candidate images (nodes 0, 1, 2) and their objects (nodes 3, 4, 5).
CANDIDATE_EDGES = np.array(
    [
        [0, 0, 1, 2],
        [3, 4, 4, 5],
    ]
)
CANDIDATE_WEIGHTS = np.array([0.5, 0.3, 0.4, 0.2])
COSINE = np.array([0.2, 0.3, 0.25])


def toy_graph_scores(remaining: np.ndarray) -> np.ndarray:
    """Stand-in for the graph model: each image scores the sum of its remaining edges."""
    scores = np.zeros(3)
    for column in range(CANDIDATE_EDGES.shape[1]):
        if remaining[column]:
            image = CANDIDATE_EDGES[0, column]
            scores[image] += CANDIDATE_WEIGHTS[column]
    return scores


def test_rank_of_counts_strictly_higher_scores():
    assert rank_of(np.array([0.5, 0.9, 0.1]), candidate=0) == 2


def test_rank_of_gives_tied_candidates_the_better_rank():
    assert rank_of(np.array([0.5, 0.5, 0.1]), candidate=1) == 1


def test_evaluate_fidelity_on_a_hand_computed_example():
    # Full: image 0 has graph 0.8 and final 0.5, ahead of image 1 (0.35).
    # Without edge (0, 3): graph 0.3, final 0.25, so image 1 overtakes it.
    result = evaluate_fidelity(
        CANDIDATE_EDGES,
        [(0, 3)],
        both_directions=False,
        graph_scores=toy_graph_scores,
        cosine=COSINE,
        alpha=0.5,
        candidate=0,
    )

    assert result.fidelity_plus == pytest.approx(0.5)
    assert result.fidelity_minus == pytest.approx(0.3)
    assert result.rank_drop == 1


def test_removing_an_unrelated_edge_changes_nothing_for_the_candidate():
    result = evaluate_fidelity(
        CANDIDATE_EDGES,
        [(2, 5)],
        both_directions=False,
        graph_scores=toy_graph_scores,
        cosine=COSINE,
        alpha=0.5,
        candidate=0,
    )

    assert result.fidelity_plus == pytest.approx(0.0)
    assert result.rank_drop == 0


def test_alpha_one_ignores_the_graph_so_rank_never_changes():
    result = evaluate_fidelity(
        CANDIDATE_EDGES,
        [(0, 3), (0, 4)],
        both_directions=False,
        graph_scores=toy_graph_scores,
        cosine=COSINE,
        alpha=1.0,
        candidate=0,
    )

    assert result.rank_drop == 0


def test_evaluate_fidelity_rejects_alpha_outside_zero_one():
    with pytest.raises(ValueError, match="alpha"):
        evaluate_fidelity(
            CANDIDATE_EDGES,
            [(0, 3)],
            both_directions=False,
            graph_scores=toy_graph_scores,
            cosine=COSINE,
            alpha=1.5,
            candidate=0,
        )
