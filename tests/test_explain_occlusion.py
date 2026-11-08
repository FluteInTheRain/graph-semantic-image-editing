import numpy as np
import pytest

from mmxai.explain.occlusion import occlusion

# Image 0 receives edges from objects 1 and 2; object 3 feeds object 1.
# Edge 0 -> 4 leaves the image.
EDGE_INDEX = np.array(
    [
        [1, 2, 3, 0],
        [0, 0, 1, 4],
    ]
)
EDGE_WEIGHT = np.array([0.5, 0.2, 0.4, 0.9])


def additive_score(remaining: np.ndarray) -> float:
    """Stand-in for the graph model: the image scores the sum of the edges that reach it."""
    reaches_image = np.array([True, True, True, False])
    return float(EDGE_WEIGHT[remaining & reaches_image].sum())


def test_each_edge_gets_its_own_weight_in_an_additive_model():
    result = occlusion(EDGE_INDEX, 0, 2, both_directions=False, score_image=additive_score)

    found = {(edge.source, edge.target): edge.contribution for edge in result}
    assert found == pytest.approx({(1, 0): 0.5, (3, 1): 0.4, (2, 0): 0.2})


def test_edges_are_sorted_from_largest_drop_to_smallest():
    result = occlusion(EDGE_INDEX, 0, 2, both_directions=False, score_image=additive_score)

    contributions = [edge.contribution for edge in result]
    assert contributions == sorted(contributions, reverse=True)


def test_edges_out_of_reach_are_not_tried():
    result = occlusion(EDGE_INDEX, 0, 1, both_directions=False, score_image=additive_score)

    found = {(edge.source, edge.target) for edge in result}
    assert (0, 4) not in found


def test_each_undirected_relation_is_tried_once():
    undirected = np.array([[0, 1, 0, 2], [1, 0, 2, 0]])

    def score(remaining: np.ndarray) -> float:
        return float(remaining.sum())

    result = occlusion(undirected, 0, 2, both_directions=True, score_image=score)

    assert len(result) == 2
    assert all(edge.contribution == pytest.approx(2.0) for edge in result)


def test_interacting_edges_each_get_the_full_drop():
    # The score needs both edges ("and"), so removing either one loses everything.
    # Occlusion then credits each edge with the full score: contributions need not add up.
    def needs_both(remaining: np.ndarray) -> float:
        return max(0.0, float(remaining[0]) + float(remaining[1]) - 1.0)

    graph = np.array([[1, 2], [0, 0]])

    result = occlusion(graph, 0, 1, both_directions=False, score_image=needs_both)

    assert [edge.contribution for edge in result] == pytest.approx([1.0, 1.0])
