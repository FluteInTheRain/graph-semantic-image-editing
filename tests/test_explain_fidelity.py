import dataclasses

import numpy as np
import pytest

from mmxai.explain.config import load_explain_config
from mmxai.explain.fidelity import (
    characterization,
    evaluate_explanation,
    evaluate_fidelity,
    fidelity_minus,
    fidelity_plus,
    normalized_fidelity,
    random_control,
    rank_drop,
    rank_of,
    robust_fidelity,
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


# Image 0 with four incoming relations of different strengths; edge 0 -> 5 leaves the image.
STAR = np.array([[1, 2, 3, 4, 0], [0, 0, 0, 0, 5]])
STAR_WEIGHT = np.array([0.4, 0.3, 0.2, 0.1, 0.9])
REACHES_IMAGE = np.array([True, True, True, True, False])


def star_score(remaining: np.ndarray) -> float:
    return float(STAR_WEIGHT[remaining & REACHES_IMAGE].sum())


def test_random_control_removes_the_requested_number_of_relations():
    values = random_control(STAR, 0, 1, False, 4, star_score, repeats=5, seed=0)

    assert values == pytest.approx([1.0] * 5)


def test_random_control_is_repeatable_with_the_same_seed():
    first = random_control(STAR, 0, 1, False, 2, star_score, repeats=10, seed=3)
    second = random_control(STAR, 0, 1, False, 2, star_score, repeats=10, seed=3)

    assert first.tolist() == second.tolist()


def test_random_control_never_touches_edges_out_of_reach():
    values = random_control(STAR, 0, 1, False, 1, star_score, repeats=50, seed=0)

    assert set(np.round(values, 6)) <= {0.4, 0.3, 0.2, 0.1}


def test_random_control_rejects_too_many_relations():
    with pytest.raises(ValueError, match="cannot remove"):
        random_control(STAR, 0, 1, False, 5, star_score, repeats=1, seed=0)


def test_robust_fidelity_with_probabilities_one_and_zero_is_plain_fidelity():
    plus, minus = robust_fidelity(
        STAR,
        0,
        1,
        [(1, 0)],
        False,
        star_score,
        remove_prob=1.0,
        add_back_prob=0.0,
        repeats=3,
        seed=0,
    )

    assert plus == pytest.approx(0.4)
    assert minus == pytest.approx(1.0 - 0.4)


def test_robust_fidelity_removes_only_part_of_the_explanation_on_average():
    plus, _ = robust_fidelity(
        STAR,
        0,
        1,
        [(1, 0), (2, 0)],
        False,
        star_score,
        remove_prob=0.5,
        add_back_prob=0.0,
        repeats=2000,
        seed=0,
    )

    assert plus == pytest.approx(0.5 * 0.7, abs=0.02)


def test_robust_fidelity_minus_adds_back_other_relations_on_average():
    _, minus = robust_fidelity(
        STAR,
        0,
        1,
        [(1, 0)],
        False,
        star_score,
        remove_prob=1.0,
        add_back_prob=0.5,
        repeats=2000,
        seed=0,
    )

    assert minus == pytest.approx(0.5 * 0.6, abs=0.02)


def test_robust_fidelity_rejects_probabilities_outside_zero_one():
    with pytest.raises(ValueError, match="remove_prob"):
        robust_fidelity(STAR, 0, 1, [(1, 0)], False, star_score, 1.5, 0.0, 1, 0)


def test_normalized_fidelity_is_a_share_of_the_graph_effect():
    plus, minus = normalized_fidelity(full=1.0, without=0.6, only=0.9, empty=0.2)

    assert plus == pytest.approx(0.5)
    assert minus == pytest.approx(0.125)


def test_normalized_fidelity_is_clipped_to_zero_one():
    plus, minus = normalized_fidelity(full=1.0, without=1.5, only=-1.0, empty=0.0)

    assert (plus, minus) == (0.0, 1.0)


def test_normalized_fidelity_is_none_when_edges_do_not_matter():
    assert normalized_fidelity(full=0.3, without=0.3, only=0.3, empty=0.3) is None


def test_characterization_is_one_for_a_perfect_explanation():
    assert characterization(fid_plus=1.0, fid_minus=0.0, plus_weight=0.5) == pytest.approx(1.0)


def test_characterization_on_a_hand_computed_example():
    # Harmonic mean of 0.5 and 1 - 0.5 = 0.5 is 0.5.
    assert characterization(0.5, 0.5, plus_weight=0.5) == pytest.approx(0.5)


def test_characterization_is_zero_when_the_explanation_is_not_necessary():
    assert characterization(fid_plus=0.0, fid_minus=0.0, plus_weight=0.5) == 0.0


def test_evaluate_explanation_on_the_star_graph():
    config = dataclasses.replace(
        load_explain_config("configs/explain_vg.yaml"), num_layers=1, both_directions=False
    )

    report = evaluate_explanation(STAR, 0, [(1, 0)], star_score, config, seed=0)

    # Full 1.0, without (1, 0): 0.6, only (1, 0): 0.4, no edges: 0.0.
    assert report.fidelity_plus == pytest.approx(0.4)
    assert report.fidelity_minus == pytest.approx(0.6)
    assert report.normalized_plus == pytest.approx(0.4)
    assert report.characterization == pytest.approx(0.4)
    assert report.random_plus_mean == pytest.approx(0.25, abs=0.1)
    assert report.sparsity == pytest.approx(0.75)
    assert report.all_in_reach
    assert report.connected_share == pytest.approx(1.0)
