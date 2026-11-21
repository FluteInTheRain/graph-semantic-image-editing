"""The G-Retriever demo tables show S_q and every concept each image's score uses."""

from mmxai.explain.retriever_path import MatchedConcept, RetrieverAnswer, RetrieverResult
from mmxai.explain.retriever_view import format_retriever_result, format_subgraph


def result(matched: list[MatchedConcept]) -> RetrieverResult:
    return RetrieverResult(1, 3, 200, 0.30, 0.40, 0.35, matched)


def test_subgraph_lists_prized_concepts_bridges_and_tree_edges():
    answer = RetrieverAnswer(["a.n.01", "c.n.01"], ["b.n.01"], [("a.n.01", "b.n.01")], [])
    text = format_subgraph(answer)
    assert "prized (closest to the query): a.n.01, c.n.01" in text
    assert "bridges (added by PCST to connect them): b.n.01" in text
    assert "a.n.01 — b.n.01" in text


def test_every_matched_concept_is_shown_with_its_kind_and_drop():
    matched = [MatchedConcept("a.n.01", False, 0.05), MatchedConcept("b.n.01", True, -0.02)]
    text = format_retriever_result(result(matched), alpha=0.5)
    assert "| a.n.01 | prized | +0.0500 |" in text
    assert "| b.n.01 | bridge | -0.0200 |" in text


def test_image_without_matched_concept_says_its_score_is_its_own_vector():
    text = format_retriever_result(result([]), alpha=0.5)
    assert "No concept of this image lies in S_q" in text


def test_alpha_one_says_the_graph_takes_no_part():
    text = format_retriever_result(result([]), alpha=1.0)
    assert "alpha = 1: the graph plays no part" in text
