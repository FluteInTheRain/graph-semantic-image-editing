"""Retrieval metrics checked against small examples computed by hand (ADR-0043)."""

import pytest
from mmxai.eval.metrics import (
    compute_metrics,
    first_relevant_rank,
    mean_reciprocal_rank,
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
