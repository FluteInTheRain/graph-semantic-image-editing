"""Retrieval metrics checked against small examples computed by hand (ADR-0043)."""

import math

import pytest

from mmxai.eval.metrics import (
    compute_metrics,
    first_relevant_rank,
    mean_reciprocal_rank,
    ndcg_at_k,
    ranked_metrics,
    recall_at_k,
)


def test_first_relevant_rank_is_one_based():
    assert first_relevant_rank([7, 3, 9], {3}) == 2


def test_first_relevant_rank_uses_the_earliest_of_several_relevant():
    assert first_relevant_rank([7, 3, 9], {9, 3}) == 2


def test_first_relevant_rank_is_none_when_missing():
    assert first_relevant_rank([7, 3, 9], {4}) is None


def test_recall_at_k_by_hand():
    # Ranks 1, 3, 6 and one miss: 1 of 4 within top 1, 2 of 4 within top 5.
    ranks = [1, 3, 6, None]

    assert recall_at_k(ranks, 1) == pytest.approx(0.25)
    assert recall_at_k(ranks, 5) == pytest.approx(0.5)
    assert recall_at_k(ranks, 10) == pytest.approx(0.75)


def test_mrr_by_hand():
    # (1/1 + 1/4 + 0) / 3 = 0.41666...
    assert mean_reciprocal_rank([1, 4, None]) == pytest.approx(1.25 / 3)


def test_metrics_reject_empty_input():
    with pytest.raises(ValueError):
        recall_at_k([], 1)
    with pytest.raises(ValueError):
        mean_reciprocal_rank([])


def test_compute_metrics_names_and_values():
    metrics = compute_metrics([1, 2, None, 10], recall_ks=[1, 5, 10], top_k=100)

    assert metrics == {
        "R@1": pytest.approx(0.25),
        "R@5": pytest.approx(0.5),
        "R@10": pytest.approx(0.75),
        "MRR@100": pytest.approx((1 + 0.5 + 0 + 0.1) / 4),
        "num_queries": 4,
    }


def test_compute_metrics_rejects_k_above_top_k():
    # Recall@200 cannot be measured from a list of 100 candidates.
    with pytest.raises(ValueError, match="top_k"):
        compute_metrics([1], recall_ks=[200], top_k=100)


def test_ndcg_one_relevant_at_first_position_is_one():
    assert ndcg_at_k([7, 3, 5], {7}, 10) == pytest.approx(1.0)


def test_ndcg_two_relevant_at_positions_two_and_five():
    dcg = 1 / math.log2(3) + 1 / math.log2(6)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k([9, 1, 8, 6, 2], {1, 2}, 10) == pytest.approx(dcg / idcg)


def test_ndcg_counts_relevant_items_outside_the_list_in_the_ideal():
    # 12 relevant ids, one found at position 1: IDCG uses min(12, 10) = 10 terms.
    idcg = sum(1 / math.log2(j + 1) for j in range(1, 11))
    assert ndcg_at_k([0], set(range(12)), 10) == pytest.approx(1 / idcg)


def test_ndcg_is_zero_without_relevant_in_the_first_k():
    assert ndcg_at_k(list(range(20)), {15}, 10) == 0.0


def test_ranked_metrics_match_compute_metrics_on_the_same_lists():
    rankings = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    relevant = [{2}, {9}, {7, 8}]
    result = ranked_metrics(rankings, relevant, [1, 2], top_k=3, ndcg_k=3)
    expected = compute_metrics([2, None, 1], [1, 2], 3)
    for name, value in expected.items():
        assert result[name] == value
    assert "nDCG@3" in result


def test_ndcg_at_k_rejects_non_positive_k():
    with pytest.raises(ValueError, match="k >= 1"):
        ndcg_at_k([1, 2], {1}, 0)


def test_ranked_metrics_rejects_invalid_ndcg_k():
    with pytest.raises(ValueError, match="top_k"):
        ranked_metrics([[1, 2]], [{1}], [1], top_k=2, ndcg_k=0)
    with pytest.raises(ValueError, match="top_k"):
        ranked_metrics([[1, 2]], [{1}], [1], top_k=2, ndcg_k=10)
