"""Combining CLIP and graph scores, and choosing alpha (ADR-0016, ADR-0046)."""

import numpy as np
import pytest

from mmxai.rerank.combine import choose_alpha, final_scores, rerank, rerank_metrics

# Two queries, three candidates each. Image ids are what the metrics compare.
CANDIDATES = np.array([[10, 11, 12], [20, 21, 22]])
COSINE = np.array([[0.9, 0.8, 0.7], [0.9, 0.8, 0.7]])
# The graph prefers the last candidate of each query.
GRAPH = np.array([[0.1, 0.2, 0.9], [0.1, 0.2, 0.9]])
RELEVANT = [{12}, {20}]


def test_final_scores_is_the_weighted_sum():
    assert final_scores(COSINE, GRAPH, 0.25) == pytest.approx(0.25 * COSINE + 0.75 * GRAPH)


def test_final_scores_rejects_alpha_outside_unit_interval():
    with pytest.raises(ValueError):
        final_scores(COSINE, GRAPH, 1.5)


def test_rerank_keeps_clip_order_between_equal_scores():
    scores = np.array([[0.5, 0.5, 0.9]])
    assert rerank(np.array([[1, 2, 3]]), scores).tolist() == [[3, 1, 2]]


def test_alpha_one_gives_plain_clip_ranks():
    _, ranks = rerank_metrics(CANDIDATES, COSINE, GRAPH, 1.0, RELEVANT, [1])
    assert ranks == [3, 1]


def test_alpha_zero_ranks_by_graph_only():
    _, ranks = rerank_metrics(CANDIDATES, COSINE, GRAPH, 0.0, RELEVANT, [1])
    assert ranks == [1, 3]


def test_choose_alpha_reports_every_alpha_and_keeps_the_larger_on_a_tie():
    # alpha = 0 and alpha = 1 give the same MRR here (ranks [1, 3] vs [3, 1]).
    alpha, by_alpha = choose_alpha(CANDIDATES, COSINE, GRAPH, RELEVANT, [1.0, 0.0], [1])
    assert set(by_alpha) == {0.0, 1.0}
    assert by_alpha[0.0]["MRR@3"] == pytest.approx(by_alpha[1.0]["MRR@3"])
    assert alpha == 1.0


def test_choose_alpha_picks_the_best_mrr():
    # Only the first query: the graph moves the answer from rank 3 to rank 1.
    alpha, _ = choose_alpha(CANDIDATES[:1], COSINE[:1], GRAPH[:1], RELEVANT[:1], [0.0, 1.0], [1])
    assert alpha == 0.0
