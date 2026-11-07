"""Direct vs indirect benchmark metrics on synthetic rankings (ADR-0043, ADR-0044).

No CLIP, no embeddings: rankings are written by hand so the numbers can be checked.
"""

import numpy as np
import pytest

from mmxai.eval.query_benchmark import (
    evaluate_benchmark,
    load_verified_benchmark,
    score_rankings,
)
from mmxai.queries.schema import write_jsonl


def query(qid: str, query_type: str, relevant: list[str]) -> dict:
    return {
        "qid": qid,
        "text": f"query {qid}",
        "type": query_type,
        "dataset": "visual_genome",
        "relevant": relevant,
        "verification_status": "verified",
    }


QUERIES = [
    query("d1", "direct", ["10"]),
    query("d2", "direct", ["20"]),
    query("i1", "indirect", ["30"]),
    query("i2", "indirect", ["40", "50"]),
]
# Ranks by hand: d1 -> 1, d2 -> 3, i1 -> miss, i2 -> 2 (first of its two answers).
RETRIEVED = np.array(
    [
        [10, 11, 12],
        [21, 22, 20],
        [31, 32, 33],
        [41, 50, 40],
    ]
)


def test_direct_and_indirect_are_scored_separately():
    metrics = score_rankings(QUERIES, RETRIEVED, recall_ks=[1, 3], top_k=3)

    assert metrics["direct"]["R@1"] == pytest.approx(0.5)
    assert metrics["direct"]["MRR@3"] == pytest.approx((1 + 1 / 3) / 2)
    assert metrics["indirect"]["R@1"] == pytest.approx(0.0)
    assert metrics["indirect"]["R@3"] == pytest.approx(0.5)
    assert metrics["indirect"]["MRR@3"] == pytest.approx(0.5 / 2)
    assert metrics["all"]["num_queries"] == 4


def test_per_query_ranks_are_kept_for_paired_comparison():
    metrics = score_rankings(QUERIES, RETRIEVED, recall_ks=[1], top_k=3)

    assert [item["rank"] for item in metrics["per_query"]] == [1, 3, None, 2]


def test_missing_group_is_reported_as_none():
    metrics = score_rankings(QUERIES[:2], RETRIEVED[:2], recall_ks=[1], top_k=3)

    assert metrics["indirect"] is None


class FixedRanker:
    """Stands in for CLIP or CLIP + Graph: any object with a rank method plugs in."""

    def __init__(self):
        self.seen_texts = None

    def rank(self, texts, query_embeddings, top_k):
        self.seen_texts = texts
        return RETRIEVED[:, :top_k]


def test_any_ranker_with_a_rank_method_can_be_evaluated():
    embeddings = np.zeros((4, 2), dtype=np.float32)

    metrics = evaluate_benchmark(QUERIES, embeddings, FixedRanker(), recall_ks=[1], top_k=3)

    assert metrics["all"]["R@1"] == pytest.approx(0.25)


def test_ranker_sees_only_texts_never_the_answers():
    ranker = FixedRanker()

    evaluate_benchmark(QUERIES, np.zeros((4, 2)), ranker, recall_ks=[1], top_k=3)

    assert ranker.seen_texts == [query["text"] for query in QUERIES]


def test_missing_benchmark_fails_with_instructions(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_query_benchmark.py"):
        load_verified_benchmark(tmp_path / "visual_genome_test.jsonl", {"10"})


def test_unverified_benchmark_is_refused(tmp_path):
    path = tmp_path / "benchmark.jsonl"
    candidate = query("d1", "direct", ["10"])
    candidate["verification_status"] = "candidate"
    write_jsonl([candidate], path)

    with pytest.raises(ValueError, match="not verified"):
        load_verified_benchmark(path, {"10"})


def test_clip_index_ranker_returns_image_ids():
    pytest.importorskip("faiss")
    from mmxai.eval.query_benchmark import ClipIndexRanker
    from mmxai.index.image_index import build_image_index

    images = np.eye(3, dtype=np.float32)
    ranker = ClipIndexRanker(build_image_index(images, np.array([7, 8, 9])))

    retrieved = ranker.rank(["a query"], np.array([[0, 1, 0]], dtype=np.float32), top_k=2)

    assert retrieved[0, 0] == 8
