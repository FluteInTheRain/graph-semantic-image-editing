import numpy as np
import pytest

from mmxai.explain.intervention import mark_remaining_edges

# Small graph: an image contains two objects, and one object relates to a third.
# Each relation is stored as two opposite edges.
EDGE_INDEX = np.array(
    [
        [0, 1, 0, 2, 1, 3],
        [1, 0, 2, 0, 3, 1],
    ]
)


def test_removes_only_the_named_edge_when_one_direction():
    remaining = mark_remaining_edges(EDGE_INDEX, [(0, 1)], both_directions=False)

    assert remaining.tolist() == [False, True, True, True, True, True]


def test_removes_the_reverse_edge_too_when_both_directions():
    remaining = mark_remaining_edges(EDGE_INDEX, [(0, 1)], both_directions=True)

    assert remaining.tolist() == [False, False, True, True, True, True]


def test_keeps_every_edge_when_nothing_is_removed():
    remaining = mark_remaining_edges(EDGE_INDEX, [], both_directions=True)

    assert remaining.all()


def test_same_result_filters_edge_types_in_step():
    edge_type = np.array([10, 10, 10, 10, 20, 20])

    remaining = mark_remaining_edges(EDGE_INDEX, [(1, 3)], both_directions=True)

    assert EDGE_INDEX[:, remaining].shape == (2, 4)
    assert edge_type[remaining].tolist() == [10, 10, 10, 10]


def test_missing_edge_raises_instead_of_being_skipped():
    with pytest.raises(ValueError, match="not in the graph"):
        mark_remaining_edges(EDGE_INDEX, [(2, 3)], both_directions=True)


def test_missing_reverse_edge_is_not_an_error_in_a_directed_graph():
    directed = np.array([[0, 1], [1, 3]])

    remaining = mark_remaining_edges(directed, [(0, 1)], both_directions=True)

    assert remaining.tolist() == [False, True]


def test_input_graph_is_not_modified():
    original = EDGE_INDEX.copy()

    mark_remaining_edges(EDGE_INDEX, [(0, 1)], both_directions=True)

    assert np.array_equal(EDGE_INDEX, original)


def test_rejects_edges_with_wrong_shape():
    with pytest.raises(ValueError, match="shape"):
        mark_remaining_edges(np.array([0, 1, 2]), [], both_directions=True)
