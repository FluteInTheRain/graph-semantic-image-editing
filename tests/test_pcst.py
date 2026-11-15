"""PCST subgraphs of a query on a tiny synthetic concept graph (ADR-0051)."""

import numpy as np
import pytest

pytest.importorskip("pcst_fast")

from mmxai.graph.pcst import (  # noqa: E402
    no_tree_from_prizes,
    no_tree_subgraph,
    prizes_from_top,
    query_subgraph,
    query_subgraph_from_prizes,
    rank_prizes,
    top_concepts,
    undirected_pairs,
    unit_rows,
)

# Concepts 0..3. Only 1 links 0 and 2; 3 hangs off 2.
PAIRS = np.array([[0, 1], [1, 2], [2, 3]])
# Unit vectors in 2-D: the query points between concept 0 and concept 2.
CONCEPTS = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
QUERY = np.array([0.6, 0.8])


def test_rank_prizes_give_k_to_the_best_and_zero_outside_top_k():
    similarity = np.array([0.1, 0.9, 0.5, 0.7])

    assert rank_prizes(similarity, 2).tolist() == [0.0, 2.0, 0.0, 1.0]


def test_rank_prizes_break_ties_by_concept_row():
    assert rank_prizes(np.array([0.5, 0.5, 0.1]), 1).tolist() == [1.0, 0.0, 0.0]


def test_undirected_pairs_drop_reverse_duplicates_and_self_loops():
    source = np.array([0, 1, 2, 2])
    target = np.array([1, 0, 2, 3])

    assert undirected_pairs(source, target).tolist() == [[0, 1], [2, 3]]


def test_cheap_edges_keep_the_zero_prize_bridge():
    subgraph = query_subgraph(QUERY, CONCEPTS, PAIRS, k=2, edge_cost=0.1)

    # Concepts 2 (best) and 0 get prizes 2 and 1; concept 1 is the only way to join them.
    assert subgraph.concepts.tolist() == [0, 1, 2]
    assert subgraph.prizes[1] == 0.0


def test_expensive_edges_keep_only_the_best_concept():
    subgraph = query_subgraph(QUERY, CONCEPTS, PAIRS, k=2, edge_cost=10.0)

    assert subgraph.concepts.tolist() == [2]


def test_no_tree_subgraph_is_the_prized_concepts_only():
    subgraph = no_tree_subgraph(QUERY, CONCEPTS, k=2)

    assert subgraph.concepts.tolist() == [0, 2]


def test_unit_rows_have_norm_one():
    rows = unit_rows(np.array([[3.0, 4.0], [0.0, 2.0]]))

    assert np.allclose(np.linalg.norm(rows, axis=1), 1.0)


def test_unit_rows_reject_a_zero_vector():
    with pytest.raises(ValueError, match="zero"):
        unit_rows(np.array([[1.0, 0.0], [0.0, 0.0]]))


def test_prizes_from_top_rows_equal_rank_prizes():
    similarity = np.array([0.1, 0.9, 0.5, 0.7, 0.9])
    top = top_concepts(similarity[None, :], 4)[0]

    for k in (1, 2, 4):
        assert (
            prizes_from_top(top, k, len(similarity)).tolist() == rank_prizes(similarity, k).tolist()
        )


def test_fast_path_gives_the_same_subgraphs_as_query_subgraph():
    rng = np.random.default_rng(0)
    concepts = rng.normal(size=(60, 8))
    queries = rng.normal(size=(15, 8))
    source = rng.integers(0, 60, size=150)
    target = rng.integers(0, 60, size=150)
    pairs = undirected_pairs(source, target)
    max_k = 6
    similarity = unit_rows(queries) @ unit_rows(concepts).T
    top = top_concepts(similarity, max_k)

    for k in (3, 6):
        costs = np.full(len(pairs), 0.2)
        for row, query in enumerate(queries):
            slow = query_subgraph(query, concepts, pairs, k, 0.2)
            prizes = prizes_from_top(top[row], k, len(concepts))
            fast = query_subgraph_from_prizes(prizes, pairs, costs)
            assert fast.concepts.tolist() == slow.concepts.tolist()
            assert fast.tree_pairs.tolist() == slow.tree_pairs.tolist()
            assert fast.prizes.tolist() == slow.prizes.tolist()
            assert no_tree_from_prizes(prizes).concepts.tolist() == (
                no_tree_subgraph(query, concepts, k).concepts.tolist()
            )
