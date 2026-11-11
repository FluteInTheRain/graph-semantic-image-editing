import numpy as np
import pytest

from mmxai.explain.intervention import mark_remaining_edges
from mmxai.explain.subgraph import edges_reaching_node, relations_in_reach
from toy_gnn import toy_gcn, toy_sum_model

# Chain 3 -> 2 -> 1 -> 0, plus an edge leaving node 0 and an edge into an unrelated node.
DIRECTED = np.array(
    [
        [3, 2, 1, 0, 5],
        [2, 1, 0, 4, 6],
    ]
)

# Mixed graph: some edges can reach node 0, some cannot.
GRAPH = np.array(
    [
        [1, 2, 3, 0, 4, 2, 5, 6],
        [0, 1, 2, 4, 3, 0, 6, 3],
    ]
)


def test_one_layer_reaches_one_step_further_than_the_direct_edges():
    reach = edges_reaching_node(DIRECTED, node=0, num_layers=1)

    assert reach.tolist() == [False, True, True, False, False]


def test_two_layers_reach_the_end_of_the_chain():
    reach = edges_reaching_node(DIRECTED, node=0, num_layers=2)

    assert reach.tolist() == [True, True, True, False, False]


def test_edges_leaving_the_node_cannot_reach_it():
    reach = edges_reaching_node(DIRECTED, node=0, num_layers=3)

    assert not reach[3]


def test_unrelated_edges_are_never_included():
    reach = edges_reaching_node(DIRECTED, node=0, num_layers=3)

    assert not reach[4]


def test_node_without_incoming_edges_has_nothing_in_reach():
    reach = edges_reaching_node(DIRECTED, node=6, num_layers=2)

    assert reach.tolist() == [False, False, False, False, True]


def test_rejects_zero_layers():
    with pytest.raises(ValueError, match="at least 1"):
        edges_reaching_node(DIRECTED, node=0, num_layers=0)


def test_rejects_edges_with_wrong_shape():
    with pytest.raises(ValueError, match="shape"):
        edges_reaching_node(np.array([0, 1, 2]), node=0, num_layers=1)


def test_gcn_degree_makes_an_edge_into_the_sender_matter():
    # Chain 2 -> 1 -> 0 with one layer: edge 2 -> 1 changes the degree of node 1,
    # which scales the message 1 -> 0, so it changes node 0's score.
    chain = np.array([[2, 1], [1, 0]])
    features = np.array([1.0, 2.0, 3.0])
    full = toy_gcn(chain, features, num_layers=1)[0]
    without = toy_gcn(chain[:, [1]], features, num_layers=1)[0]

    assert without != pytest.approx(full)
    assert edges_reaching_node(chain, node=0, num_layers=1)[0]


@pytest.mark.parametrize("model", [toy_sum_model, toy_gcn])
@pytest.mark.parametrize("num_layers", [1, 2])
def test_every_edge_that_changes_the_score_is_in_reach(model, num_layers):
    features = np.random.default_rng(0).uniform(1.0, 2.0, size=7)
    full = model(GRAPH, features, num_layers)[0]
    reach = edges_reaching_node(GRAPH, node=0, num_layers=num_layers)

    for column in range(GRAPH.shape[1]):
        edge = (int(GRAPH[0, column]), int(GRAPH[1, column]))
        remaining = mark_remaining_edges(GRAPH, [edge], both_directions=False)
        score = model(GRAPH[:, remaining], features, num_layers)[0]
        if score != pytest.approx(full):
            assert reach[column], f"edge {edge} changes the score but is not in reach"


@pytest.mark.parametrize("model", [toy_sum_model, toy_gcn])
def test_keeping_only_edges_in_reach_keeps_the_score(model):
    features = np.random.default_rng(1).uniform(1.0, 2.0, size=7)
    reach = edges_reaching_node(GRAPH, node=0, num_layers=2)

    full = model(GRAPH, features, 2)[0]
    only_reach = model(GRAPH[:, reach], features, 2)[0]

    assert only_reach == pytest.approx(full)


def test_relations_in_reach_lists_each_undirected_relation_once():
    undirected = np.array([[0, 1, 1, 2], [1, 0, 2, 1]])

    relations = relations_in_reach(undirected, node=0, num_layers=2, both_directions=True)

    assert sorted(relations) == [(1, 0), (1, 2)]


def test_relations_in_reach_keeps_directed_edges_as_they_are():
    relations = relations_in_reach(DIRECTED, node=0, num_layers=1, both_directions=False)

    assert relations == [(2, 1), (1, 0)]
