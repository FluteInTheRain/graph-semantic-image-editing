import numpy as np
import pytest

from mmxai.explain.intervention import mark_remaining_edges
from mmxai.explain.subgraph import edges_reaching_node

# Chain 3 -> 2 -> 1 -> 0, plus an edge leaving node 0 and an edge into an unrelated node.
DIRECTED = np.array(
    [
        [3, 2, 1, 0, 5],
        [2, 1, 0, 4, 6],
    ]
)


def test_one_layer_keeps_only_edges_into_the_node():
    reach = edges_reaching_node(DIRECTED, node=0, num_layers=1)

    assert reach.tolist() == [False, False, True, False, False]


def test_two_layers_add_edges_into_the_neighbours():
    reach = edges_reaching_node(DIRECTED, node=0, num_layers=2)

    assert reach.tolist() == [False, True, True, False, False]


def test_three_layers_reach_the_end_of_the_chain():
    reach = edges_reaching_node(DIRECTED, node=0, num_layers=3)

    assert reach.tolist() == [True, True, True, False, False]


def test_edges_leaving_the_node_cannot_reach_it():
    reach = edges_reaching_node(DIRECTED, node=0, num_layers=3)

    assert not reach[3]


def test_message_can_go_out_and_come_back_in_an_undirected_graph():
    # 0 <-> 1 stored as two opposite edges: 0 -> 1 -> 0 takes two rounds.
    undirected = np.array([[0, 1], [1, 0]])

    assert edges_reaching_node(undirected, node=0, num_layers=1).tolist() == [False, True]
    assert edges_reaching_node(undirected, node=0, num_layers=2).tolist() == [True, True]


def test_rejects_zero_layers():
    with pytest.raises(ValueError, match="at least 1"):
        edges_reaching_node(DIRECTED, node=0, num_layers=0)


def toy_message_passing(edge_index: np.ndarray, features: np.ndarray, num_layers: int):
    """Stand-in for a graph model: each round, a node adds the values sent along its edges."""
    state = features.copy()
    for _ in range(num_layers):
        new_state = state.copy()
        for source, target in edge_index.T:
            new_state[target] += state[source]
        state = new_state
    return state


def test_only_edges_in_reach_change_the_node_score():
    rng = np.random.default_rng(0)
    graph = np.array(
        [
            [1, 2, 3, 0, 4, 2, 5],
            [0, 1, 2, 4, 3, 0, 6],
        ]
    )
    features = rng.uniform(1.0, 2.0, size=7)
    num_layers = 2
    full_score = toy_message_passing(graph, features, num_layers)[0]

    reach = edges_reaching_node(graph, node=0, num_layers=num_layers)

    for column in range(graph.shape[1]):
        edge = (int(graph[0, column]), int(graph[1, column]))
        remaining = mark_remaining_edges(graph, [edge], both_directions=False)
        score = toy_message_passing(graph[:, remaining], features, num_layers)[0]
        assert (score != pytest.approx(full_score)) == reach[column]
