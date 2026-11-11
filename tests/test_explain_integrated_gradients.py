import numpy as np
import pytest

from mmxai.explain.integrated_gradients import integrated_gradients

# Image 0 receives edges from objects 1, 2 and 3; edge 0 -> 4 leaves the image.
GRAPH = np.array([[1, 2, 3, 0], [0, 0, 0, 4]])
VALUES = np.array([0.5, 0.3, 0.2, 0.9])
INTO_IMAGE = np.array([1.0, 1.0, 1.0, 0.0])


def linear_score(weights: np.ndarray) -> tuple[float, np.ndarray]:
    score = float((weights * VALUES * INTO_IMAGE).sum())
    return score, VALUES * INTO_IMAGE


def squared_score(weights: np.ndarray) -> tuple[float, np.ndarray]:
    total = float((weights * VALUES * INTO_IMAGE).sum())
    return total**2, 2 * total * VALUES * INTO_IMAGE


def test_linear_model_gives_each_relation_its_own_value():
    result, gap = integrated_gradients(GRAPH, 0, 1, False, linear_score, steps=10)

    found = {(edge.source, edge.target): edge.contribution for edge in result}
    assert found == pytest.approx({(1, 0): 0.5, (2, 0): 0.3, (3, 0): 0.2})
    assert gap == pytest.approx(0.0)


def test_contributions_add_up_to_the_score_change_for_a_nonlinear_model():
    result, gap = integrated_gradients(GRAPH, 0, 1, False, squared_score, steps=50)

    assert sum(edge.contribution for edge in result) == pytest.approx(1.0, abs=1e-3)
    assert gap == pytest.approx(0.0, abs=1e-3)


def test_interacting_edges_share_the_score_instead_of_each_taking_all():
    # The score needs both edges ("and"). Occlusion credits each with 1;
    # Integrated Gradients splits the score, 0.5 each.
    graph = np.array([[1, 2], [0, 0]])

    def needs_both(weights: np.ndarray) -> tuple[float, np.ndarray]:
        return float(weights[0] * weights[1]), np.array([weights[1], weights[0]])

    result, gap = integrated_gradients(graph, 0, 1, False, needs_both, steps=4)

    assert [edge.contribution for edge in result] == pytest.approx([0.5, 0.5])
    assert gap == pytest.approx(0.0)


def test_both_copies_of_an_undirected_relation_count_as_one():
    undirected = np.array([[1, 0], [0, 1]])

    def score(weights: np.ndarray) -> tuple[float, np.ndarray]:
        return float(weights.sum()), np.ones(2)

    result, _ = integrated_gradients(undirected, 0, 2, True, score, steps=5)

    assert len(result) == 1
    assert result[0].contribution == pytest.approx(2.0)


def test_rejects_zero_steps():
    with pytest.raises(ValueError, match="at least 1"):
        integrated_gradients(GRAPH, 0, 1, False, linear_score, steps=0)
