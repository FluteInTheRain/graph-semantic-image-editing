import numpy as np
import pytest

from mmxai.explain.gnn_explainer import gnn_explainer

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


def test_the_relation_that_carries_the_score_keeps_the_highest_mask():
    result = gnn_explainer(GRAPH, 0, 1, False, linear_score, seed=0, **SETTINGS)

    assert (result[0].source, result[0].target) == (1, 0)
    assert result[0].contribution > 0.5
    assert all(edge.contribution < 0.5 for edge in result[1:])


def test_mask_values_stay_between_zero_and_one():
    result = gnn_explainer(GRAPH, 0, 1, False, linear_score, seed=0, **SETTINGS)

    assert all(0.0 <= edge.contribution <= 1.0 for edge in result)


def test_same_seed_gives_the_same_mask():
    first = gnn_explainer(GRAPH, 0, 1, False, linear_score, seed=7, **SETTINGS)
    second = gnn_explainer(GRAPH, 0, 1, False, linear_score, seed=7, **SETTINGS)

    assert [e.contribution for e in first] == [e.contribution for e in second]


def test_rejects_zero_epochs():
    with pytest.raises(ValueError, match="at least 1"):
        gnn_explainer(
            GRAPH,
            0,
            1,
            False,
            linear_score,
            seed=0,
            epochs=0,
            learning_rate=1.0,
            size_weight=0.0,
            entropy_weight=0.0,
        )
