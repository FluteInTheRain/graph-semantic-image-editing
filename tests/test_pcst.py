"""PCST subgraphs of a query on a tiny synthetic concept graph (ADR-0051)."""

import numpy as np
import pytest

pytest.importorskip("pcst_fast")

from mmxai.graph.pcst import no_tree_subgraph, query_subgraph, rank_prizes, undirected_pairs  # noqa: E402

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
