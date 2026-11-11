import dataclasses

import numpy as np
import pytest

from mmxai.explain.config import load_explain_config
from mmxai.explain.fidelity import (
    characterization,
    evaluate_explanation,
    fidelity_minus,
    fidelity_plus,
    final_scores,
    normalized_fidelity,
    rank_drop,
    rank_of,
    robust_fidelity,
    sample_relation_sets,
)
from mmxai.explain.occlusion import occlusion
from toy_gnn import toy_gcn

CONFIG = load_explain_config("configs/explain_vg.yaml")


def test_fidelity_plus_on_a_hand_computed_example():
    assert fidelity_plus(full_score=1.0, score_without_explanation=0.2) == pytest.approx(0.8)


def test_fidelity_minus_on_a_hand_computed_example():
    assert fidelity_minus(full_score=1.0, score_with_only_explanation=0.9) == pytest.approx(0.1)


def test_fidelity_works_with_negative_scores():
    assert fidelity_plus(full_score=-0.1, score_without_explanation=-0.4) == pytest.approx(0.3)


def test_rank_drop_is_positive_when_the_candidate_falls():
    assert rank_drop(rank_before=2, rank_after=5) == 3


def test_rank_drop_is_negative_when_the_candidate_rises():
    assert rank_drop(rank_before=4, rank_after=1) == -3


def test_rank_drop_rejects_ranks_below_one():
    with pytest.raises(ValueError, match="start at 1"):
        rank_drop(rank_before=0, rank_after=3)


def test_rank_of_counts_strictly_higher_scores():
    assert rank_of(np.array([0.5, 0.9, 0.1]), candidate=0) == 2


def test_rank_of_gives_tied_candidates_the_better_rank():
    assert rank_of(np.array([0.5, 0.5, 0.1]), candidate=1) == 1


def test_final_scores_mix_cosine_and_graph():
    scores = final_scores(np.array([0.2, 0.4]), np.array([1.0, 0.0]), alpha=0.5)

    assert scores == pytest.approx([0.6, 0.2])


def test_final_scores_reject_alpha_outside_zero_one():
    with pytest.raises(ValueError, match="alpha"):
        final_scores(np.array([0.2]), np.array([1.0]), alpha=1.5)


def test_sample_relation_sets_have_the_requested_size_and_no_repeats():
    relations = [(1, 0), (2, 0), (3, 0), (4, 0)]

    sets = sample_relation_sets(relations, size=2, repeats=20, seed=0)

    assert len(sets) == 20
    assert all(len(set(picked)) == 2 for picked in sets)


def test_sample_relation_sets_are_repeatable_with_the_same_seed():
    relations = [(1, 0), (2, 0), (3, 0), (4, 0)]

    first = sample_relation_sets(relations, 2, repeats=10, seed=3)
    second = sample_relation_sets(relations, 2, repeats=10, seed=3)

    assert first == second


def test_sample_relation_sets_reject_too_many_relations():
    with pytest.raises(ValueError, match="cannot pick"):
        sample_relation_sets([(1, 0)], size=2, repeats=1, seed=0)


# Image 0 with four incoming relations of different strengths; edge 0 -> 5 leaves the image.
STAR = np.array([[1, 2, 3, 4, 0], [0, 0, 0, 0, 5]])
STAR_WEIGHT = np.array([0.4, 0.3, 0.2, 0.1, 0.9])
REACHES_IMAGE = np.array([True, True, True, True, False])


def star_score(remaining: np.ndarray) -> float:
    return float(STAR_WEIGHT[remaining & REACHES_IMAGE].sum())


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
    result = normalized_fidelity(full=1.0, without=0.6, only=0.9, empty=0.2)

    assert result.plus == pytest.approx(0.5)
    assert result.minus == pytest.approx(0.125)
    assert not result.clipped


def test_normalized_fidelity_reports_clipping():
    result = normalized_fidelity(full=1.0, without=1.5, only=-1.0, empty=0.0)

    assert (result.plus, result.minus) == (0.0, 1.0)
    assert result.clipped


def test_normalized_fidelity_is_none_when_edges_do_not_matter():
    assert normalized_fidelity(full=0.3, without=0.3, only=0.3, empty=0.3) is None


def test_characterization_is_one_for_a_perfect_explanation():
    assert characterization(fid_plus=1.0, fid_minus=0.0, plus_weight=0.5) == pytest.approx(1.0)


def test_characterization_on_a_hand_computed_example():
    # Harmonic mean of 0.5 and 1 - 0.5 = 0.5 is 0.5.
    assert characterization(0.5, 0.5, plus_weight=0.5) == pytest.approx(0.5)


def test_characterization_is_zero_when_the_explanation_is_not_necessary():
    assert characterization(fid_plus=0.0, fid_minus=0.0, plus_weight=0.5) == 0.0


# Two candidate images: image 0 (node 0) is the star above, image 1 (node 6) has one edge.
TWO_IMAGES = np.array([[1, 2, 3, 4, 7], [0, 0, 0, 0, 6]])
TWO_WEIGHTS = np.array([0.4, 0.3, 0.2, 0.1, 0.65])


def two_image_scores(remaining: np.ndarray) -> np.ndarray:
    into_first = np.array([True, True, True, True, False])
    first = float(TWO_WEIGHTS[remaining & into_first].sum())
    second = float(TWO_WEIGHTS[remaining & ~into_first].sum())
    return np.array([first, second])


STAR_CONFIG = dataclasses.replace(CONFIG, num_layers=1, both_directions=False)
RANKED = two_image_scores(np.ones(5, dtype=bool))
COSINE = np.array([0.5, 0.5])


def evaluate(explanation, alpha=0.5, ranked=RANKED, config=STAR_CONFIG):
    return evaluate_explanation(
        TWO_IMAGES, 0, 0, explanation, two_image_scores, ranked, COSINE, alpha, config, seed=0
    )


def test_evaluate_explanation_on_a_hand_computed_example():
    # Image 0: full 1.0, without (1, 0) 0.6, only (1, 0) 0.4, no edges 0.0.
    # Final scores 0.75 vs 0.575; without (1, 0): 0.55, so image 1 overtakes it.
    report = evaluate([(1, 0)])

    assert report.fidelity_plus == pytest.approx(0.4)
    assert report.fidelity_minus == pytest.approx(0.6)
    assert report.rank_drop == 1
    assert report.normalized_plus == pytest.approx(0.4)
    assert report.characterization == pytest.approx(0.4)
    assert report.random_plus_mean == pytest.approx(0.25, abs=0.1)
    assert report.sparsity == pytest.approx(0.75)
    assert report.all_in_reach
    assert report.connected_share == pytest.approx(1.0)


def test_evaluate_explanation_is_not_applicable_when_alpha_is_one():
    assert evaluate([(1, 0)], alpha=1.0) is None


def test_evaluate_explanation_rejects_scores_unlike_the_ranked_ones():
    with pytest.raises(ValueError, match="used for ranking"):
        evaluate([(1, 0)], ranked=np.array([0.9, 0.65]))


def test_evaluate_explanation_counts_undirected_relations_once():
    undirected = np.array([[1, 0, 2, 0], [0, 1, 0, 2]])

    def scores(remaining: np.ndarray) -> np.ndarray:
        return np.array([float(remaining.sum())])

    config = dataclasses.replace(CONFIG, num_layers=1, both_directions=True)
    report = evaluate_explanation(
        undirected,
        0,
        0,
        [(1, 0)],
        scores,
        scores(np.ones(4, dtype=bool)),
        np.array([0.0]),
        0.5,
        config,
        seed=0,
    )

    assert report.fidelity_plus == pytest.approx(2.0)
    assert report.sparsity == pytest.approx(0.5)


def test_occlusion_explanation_beats_random_removal_on_a_toy_gcn():
    # Image 0 has one strong neighbour (node 1) and four weak ones.
    graph = np.array([[1, 2, 3, 4, 5], [0, 0, 0, 0, 0]])
    features = np.array([1.0, 3.0, 0.2, 0.2, 0.2, 0.2])

    def scores(remaining: np.ndarray) -> np.ndarray:
        return np.array([float(toy_gcn(graph[:, remaining], features, num_layers=1)[0])])

    ranked = scores(np.ones(5, dtype=bool))
    top = occlusion(graph, 0, 1, False, lambda m: float(scores(m)[0]), float(ranked[0]))[0]

    config = dataclasses.replace(CONFIG, num_layers=1, both_directions=False)
    report = evaluate_explanation(
        graph,
        0,
        0,
        [(top.source, top.target)],
        scores,
        ranked,
        np.array([0.0]),
        0.5,
        config,
        seed=0,
    )

    assert (top.source, top.target) == (1, 0)
    assert report.fidelity_plus > report.random_plus_mean
