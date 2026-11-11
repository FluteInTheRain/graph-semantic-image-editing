import numpy as np
import pytest

from mmxai.explain.occlusion import measure_drops, occlusion
from toy_gnn import toy_gcn

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


def run_occlusion(graph, image_node, num_layers, both_directions, score_image):
    """Occlusion with the ranked score taken from the same model, as in a real run."""
    ranked = score_image(np.ones(graph.shape[1], dtype=bool))
    return occlusion(graph, image_node, num_layers, both_directions, score_image, ranked)


def test_each_edge_gets_its_own_weight_in_an_additive_model():
    result = run_occlusion(EDGE_INDEX, 0, 2, both_directions=False, score_image=additive_score)

    found = {(edge.source, edge.target): edge.contribution for edge in result}
    assert found == pytest.approx({(1, 0): 0.5, (3, 1): 0.4, (2, 0): 0.2})


def test_edges_are_sorted_from_largest_drop_to_smallest():
    result = run_occlusion(EDGE_INDEX, 0, 2, both_directions=False, score_image=additive_score)

    contributions = [edge.contribution for edge in result]
    assert contributions == sorted(contributions, reverse=True)


def test_edges_out_of_reach_are_not_tried():
    result = run_occlusion(EDGE_INDEX, 0, 1, both_directions=False, score_image=additive_score)

    found = {(edge.source, edge.target) for edge in result}
    assert (0, 4) not in found


def test_each_undirected_relation_is_tried_once():
    undirected = np.array([[0, 1, 0, 2], [1, 0, 2, 0]])

    def score(remaining: np.ndarray) -> float:
        return float(remaining.sum())

    result = run_occlusion(undirected, 0, 2, both_directions=True, score_image=score)

    assert len(result) == 2
    assert all(edge.contribution == pytest.approx(2.0) for edge in result)


def test_interacting_edges_each_get_the_full_drop():
    # The score needs both edges ("and"), so removing either one loses everything.
    # Occlusion then credits each edge with the full score: contributions need not add up.
    def needs_both(remaining: np.ndarray) -> float:
        return max(0.0, float(remaining[0]) + float(remaining[1]) - 1.0)

    graph = np.array([[1, 2], [0, 0]])

    result = run_occlusion(graph, 0, 1, both_directions=False, score_image=needs_both)

    assert [edge.contribution for edge in result] == pytest.approx([1.0, 1.0])


def test_undirected_relation_points_at_the_image():
    undirected = np.array([[0, 1], [1, 0]])

    def score(remaining: np.ndarray) -> float:
        return float(remaining.sum())

    result = run_occlusion(undirected, 0, 1, both_directions=True, score_image=score)

    assert [(edge.source, edge.target) for edge in result] == [(1, 0)]


def test_reported_direction_does_not_depend_on_edge_order():
    forward = np.array([[3, 1, 1, 0], [1, 3, 0, 1]])
    backward = forward[:, ::-1]

    def score(remaining: np.ndarray) -> float:
        return float(remaining.sum())

    first = run_occlusion(forward, 0, 2, both_directions=True, score_image=score)
    second = run_occlusion(backward, 0, 2, both_directions=True, score_image=score)

    assert {(e.source, e.target) for e in first} == {(e.source, e.target) for e in second}
    assert {(e.source, e.target) for e in first} == {(1, 0), (1, 3)}


def test_rejects_a_score_that_changes_between_calls():
    calls = []

    def unstable(remaining: np.ndarray) -> float:
        calls.append(1)
        return float(len(calls))

    with pytest.raises(ValueError, match="different results"):
        run_occlusion(EDGE_INDEX, 0, 2, both_directions=False, score_image=unstable)


def test_rejects_a_score_that_depends_on_edges_out_of_reach():
    # Edge 0 -> 4 leaves the image, but this score counts it: num_layers does not match the model.
    def counts_everything(remaining: np.ndarray) -> float:
        return float(EDGE_WEIGHT[remaining].sum())

    with pytest.raises(ValueError, match="out of reach"):
        run_occlusion(EDGE_INDEX, 0, 2, both_directions=False, score_image=counts_everything)


def test_on_a_toy_gcn_each_contribution_matches_removing_that_edge():
    graph = np.array(
        [
            [1, 2, 3, 0, 4, 2, 5, 6],
            [0, 1, 2, 4, 3, 0, 6, 3],
        ]
    )
    features = np.random.default_rng(0).uniform(1.0, 2.0, size=7)

    def gcn_score(remaining: np.ndarray) -> float:
        return float(toy_gcn(graph[:, remaining], features, num_layers=2)[0])

    result = run_occlusion(graph, 0, 2, both_directions=False, score_image=gcn_score)
    full = gcn_score(np.ones(graph.shape[1], dtype=bool))
    tried = {(edge.source, edge.target) for edge in result}

    # Recompute every drop directly, without the explain code.
    for column in range(graph.shape[1]):
        keep = np.ones(graph.shape[1], dtype=bool)
        keep[column] = False
        drop = full - gcn_score(keep)
        edge = (int(graph[0, column]), int(graph[1, column]))
        if edge in tried:
            reported = next(e.contribution for e in result if (e.source, e.target) == edge)
            assert reported == pytest.approx(drop)
        else:
            assert drop == pytest.approx(0.0), f"untried edge {edge} changes the score"

    assert result[0].contribution != pytest.approx(0.0)


def test_rejects_a_score_unlike_the_one_used_for_ranking():
    with pytest.raises(ValueError, match="ranking used"):
        occlusion(EDGE_INDEX, 0, 2, False, additive_score, ranked_graph_score=0.123)


def test_measure_drops_only_tries_the_given_relations():
    result = measure_drops(EDGE_INDEX, [(2, 0)], False, additive_score)

    assert [(e.source, e.target, round(e.contribution, 6)) for e in result] == [(2, 0, 0.2)]
