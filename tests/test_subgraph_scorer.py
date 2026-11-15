"""Stage-1 G-Retriever score: share of the query subgraph an image touches (ADR-0051)."""

import numpy as np

from mmxai.graph.contains import ContainsGraph
from mmxai.graph.pcst import QuerySubgraph
from mmxai.models.subgraph_scorer import (
    explain_overlap,
    image_concept_sets,
    overlap_scores,
)


def contains_graph() -> ContainsGraph:
    # Images 10, 11, 12; concepts 0..3. 10 has {0, 1}, 11 has {2}, 12 has nothing.
    return ContainsGraph(
        image_ids=np.array([10, 11, 12]),
        image_split=np.array(["test", "test", "test"]),
        concept_keys=np.array(["a.n.01", "b.n.01", "c.n.01", "d.n.01"]),
        edge_image=np.array([0, 0, 1]),
        edge_concept=np.array([0, 1, 2]),
    )


def subgraph(concepts: list[int], prizes: list[float]) -> QuerySubgraph:
    return QuerySubgraph(np.array(concepts), np.zeros((0, 2), dtype=np.int64), np.array(prizes))


def test_image_concept_sets():
    sets = image_concept_sets(contains_graph())

    assert sets == {10: {0, 1}, 11: {2}, 12: set()}


def test_overlap_counts_bridges_like_prized_concepts():
    sets = image_concept_sets(contains_graph())
    # S_q = {0, 1, 2}; concept 1 is a bridge (prize 0) but still counts.
    subgraphs = [subgraph([0, 1, 2], [2.0, 0.0, 1.0, 0.0])]
    candidate_ids = np.array([[10, 11, 12]])

    scores = overlap_scores(candidate_ids, subgraphs, sets)

    assert np.allclose(scores, [[2 / 3, 1 / 3, 0.0]])


def test_explanation_is_exactly_what_the_score_counts():
    sets = image_concept_sets(contains_graph())
    query_subgraph = subgraph([0, 1, 2], [2.0, 0.0, 1.0, 0.0])
    graph = contains_graph()

    record = explain_overlap(10, query_subgraph, sets, graph.concept_keys)

    assert record["matched"] == [
        {"concept": "a.n.01", "bridge": False},
        {"concept": "b.n.01", "bridge": True},
    ]
    assert record["score"] == 2 / 3
    # Fidelity by construction: dropping one matched concept lowers the score by 1/|S_q|.
    assert record["score_without_each"] == [1 / 3, 1 / 3]
