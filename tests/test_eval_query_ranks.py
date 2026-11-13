import numpy as np
import pytest

pytest.importorskip("faiss")  # mmxai.rerank.fusion builds FAISS indexes

from mmxai.eval.query_ranks import (  # noqa: E402
    clip_ranks,
    fused_ranks,
    rank_of_column,
    select_queries,
)
from mmxai.rerank.fusion import Candidates  # noqa: E402

# Two queries, three candidates each, in CLIP's order (clip scores decrease along a row).
# Query 0: relevant image 30 is CLIP's 3rd, but has the best graph score.
# Query 1: relevant image 40 is CLIP's 1st, but has the worst graph score.
CANDIDATES = Candidates(
    query_embeddings=np.zeros((2, 4), dtype=np.float32),
    candidate_ids=np.array([[10, 20, 30], [40, 50, 60]]),
    clip_scores=np.array([[0.9, 0.8, 0.7], [0.9, 0.8, 0.7]]),
    relevant=[{30}, {40}],
)
GRAPH_SCORES = np.array([[0.0, 0.0, 1.0], [0.0, 0.5, 0.5]])


def test_clip_ranks_follow_clip_order():
    assert clip_ranks(CANDIDATES) == [3, 1]


def test_fused_ranks_on_a_hand_computed_example():
    # alpha = 0.5. Query 0 final: [0.45, 0.40, 0.85] -> image 30 first.
    # Query 1 final: [0.45, 0.65, 0.60] -> order 50, 60, 40 -> image 40 third.
    assert fused_ranks(CANDIDATES, GRAPH_SCORES, alpha=0.5) == [1, 3]


def test_alpha_one_gives_exactly_the_clip_ranks():
    assert fused_ranks(CANDIDATES, GRAPH_SCORES, alpha=1.0) == clip_ranks(CANDIDATES)


def test_relevant_image_outside_the_candidates_has_no_rank():
    candidates = Candidates(
        query_embeddings=np.zeros((1, 4), dtype=np.float32),
        candidate_ids=np.array([[10, 20, 30]]),
        clip_scores=np.array([[0.9, 0.8, 0.7]]),
        relevant=[{99}],
    )

    assert clip_ranks(candidates) == [None]


def test_rank_of_column_uses_the_final_score_order():
    final = np.array([0.45, 0.65, 0.60])

    assert [rank_of_column(final, column) for column in range(3)] == [3, 1, 2]


def test_equal_final_scores_keep_clip_order():
    final = np.array([0.5, 0.5, 0.5])

    assert [rank_of_column(final, column) for column in range(3)] == [1, 2, 3]


def test_success_is_the_largest_rank_improvement():
    clip = [5, 10, 3, 8]
    fused = [4, 1, 3, 2]  # changes: +1, +9, 0, +6

    success, _ = select_queries(clip, fused, count=2)

    assert success == [1, 3]


def test_failure_is_the_largest_rank_drop():
    clip = [1, 2, 3, 4]
    fused = [9, 3, 3, 10]  # changes: -8, -1, 0, -6

    _, failure = select_queries(clip, fused, count=2)

    assert failure == [0, 3]


def test_ties_go_to_the_smaller_query_index():
    clip = [4, 6, 4, 6]
    fused = [2, 4, 6, 8]  # changes: +2, +2, -2, -2

    success, failure = select_queries(clip, fused, count=1)

    assert success == [0]
    assert failure == [2]


def test_unchanged_queries_and_queries_without_rank_are_never_picked():
    clip = [None, 3, None, 7]
    fused = [None, 3, None, 7]

    success, failure = select_queries(clip, fused, count=5)

    assert success == []
    assert failure == []


def test_fewer_than_count_are_returned_when_not_enough_exist():
    clip = [3, 1]
    fused = [1, 2]

    success, failure = select_queries(clip, fused, count=5)

    assert success == [0]
    assert failure == [1]


def test_rejects_rank_lists_of_different_length():
    with pytest.raises(ValueError, match="CLIP\\+Graph ranks"):
        select_queries([1, 2], [1], count=1)
