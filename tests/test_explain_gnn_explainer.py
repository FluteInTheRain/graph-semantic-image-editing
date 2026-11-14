import numpy as np
import pytest

from mmxai.explain.gnn_explainer import gnn_explainer, mask_loss, strongest_relations
from mmxai.explain.integrated_gradients import relation_columns
from mmxai.explain.intervention import mark_remaining_edges
from toy_gnn import finite_difference_gradient

# Image 0 receives edges from objects 1, 2 and 3; only the first one matters.
GRAPH = np.array([[1, 2, 3], [0, 0, 0]])
VALUES = np.array([1.0, 0.0, 0.0])

SETTINGS = {
    "epochs": 300,
    "learning_rate": 1.0,
    "size_weight": 0.05,
    "entropy_weight": 0.1,
}


def linear_score(weights: np.ndarray) -> tuple[float, np.ndarray]:
    return float((weights * VALUES).sum()), VALUES.copy()


def linear_mask_score(remaining: np.ndarray) -> float:
    return linear_score(remaining.astype(float))[0]


def run(seed: int, **settings):
    return gnn_explainer(
        GRAPH, 0, 1, False, linear_mask_score, linear_score, 1.0, seed=seed, **settings
    )


def test_the_relation_that_carries_the_score_keeps_the_highest_mask():
    result = run(0, **SETTINGS)

    assert (result[0].source, result[0].target) == (1, 0)
    assert result[0].mask > 0.5
    assert all(edge.mask < 0.5 for edge in result[1:])


def test_mask_values_stay_between_zero_and_one():
    assert all(0.0 <= edge.mask <= 1.0 for edge in run(0, **SETTINGS))


def test_same_seed_gives_the_same_mask():
    assert [e.mask for e in run(7, **SETTINGS)] == [e.mask for e in run(7, **SETTINGS)]


def test_result_is_a_mask_not_a_score_contribution():
    result = run(0, **SETTINGS)

    assert not hasattr(result[0], "contribution")


def test_loss_gradient_matches_finite_differences():
    columns = relation_columns(GRAPH, [(1, 0), (2, 0), (3, 0)], both_directions=False)
    logits = np.array([0.3, -0.2, 1.1])

    def loss_only(point: np.ndarray) -> float:
        return mask_loss(point, columns, linear_score, 1.0, 0.05, 0.1)[0]

    _, gradient = mask_loss(logits, columns, linear_score, 1.0, 0.05, 0.1)

    assert gradient == pytest.approx(finite_difference_gradient(loss_only, logits), abs=1e-6)


def test_rejects_zero_epochs():
    with pytest.raises(ValueError, match="at least 1"):
        run(0, epochs=0, learning_rate=1.0, size_weight=0.0, entropy_weight=0.0)


def test_shown_effects_come_from_removing_the_picked_relations():
    picked = strongest_relations(run(0, **SETTINGS), top_m=1)

    full = linear_mask_score(np.ones(GRAPH.shape[1], dtype=bool))
    without = linear_mask_score(mark_remaining_edges(GRAPH, picked, False))

    assert picked == [(1, 0)]
    assert full - without == pytest.approx(1.0)
