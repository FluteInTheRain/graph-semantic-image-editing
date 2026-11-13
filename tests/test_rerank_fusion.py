"""Score fusion, re-ranking and the choice of alpha (PDF section 4.2, ADR-0046)."""

import numpy as np
import pytest

pytest.importorskip("faiss")

from mmxai.eval.clip_baseline import evaluate_clip  # noqa: E402
from mmxai.rerank.fusion import (  # noqa: E402
    Candidates,
    candidate_graph_scores,
    choose_alpha,
    fuse_scores,
    rerank_ranks,
    retrieve_candidates,
    sweep_alpha,
)


def unit_rows(matrix: np.ndarray) -> np.ndarray:
    return (matrix / np.linalg.norm(matrix, axis=1, keepdims=True)).astype(np.float32)


def random_split_data():
    """8 images (ids 10..17), 2 texts each, near their image so CLIP is right often."""
    generator = np.random.default_rng(0)
    image_ids = np.arange(10, 18)
    images = unit_rows(generator.normal(size=(8, 6)))
    text_image_ids = np.repeat(image_ids, 2)
    texts = unit_rows(np.repeat(images, 2, axis=0) + generator.normal(scale=0.8, size=(16, 6)))
    return images, image_ids, texts, text_image_ids


# One query, three candidates in CLIP order; image 3 is the correct one.
ONE_QUERY = Candidates(
    query_embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
    candidate_ids=np.array([[1, 2, 3]]),
    clip_scores=np.array([[0.9, 0.8, 0.7]]),
    relevant=[{3}],
)


def test_fusion_matches_the_hand_computed_example():
    # README section 3.11: 0.7 * 0.30 + 0.3 * 0.80 = 0.45.
    final = fuse_scores(np.array([0.30]), np.array([0.80]), alpha=0.7)

    assert final[0] == pytest.approx(0.45)


def test_alpha_outside_zero_one_is_an_error():
    with pytest.raises(ValueError, match="alpha"):
        fuse_scores(np.array([0.3]), np.array([0.8]), alpha=1.5)


def test_graph_score_can_move_the_correct_image_up():
    graph = np.array([[0.0, 0.0, 1.0]])

    clip_only = rerank_ranks(ONE_QUERY, fuse_scores(ONE_QUERY.clip_scores, graph, 1.0))
    graph_only = rerank_ranks(ONE_QUERY, fuse_scores(ONE_QUERY.clip_scores, graph, 0.0))

    assert clip_only == [3]
    assert graph_only == [1]


def test_equal_final_scores_keep_clip_order():
    flat = np.array([[0.5, 0.5, 0.5]])

    assert rerank_ranks(ONE_QUERY, flat) == [3]


def test_alpha_one_gives_exactly_the_plain_clip_metrics():
    # The key fairness check: with the graph switched off, re-ranking changes nothing.
    images, image_ids, texts, text_image_ids = random_split_data()
    eval_ids = image_ids[:6]
    top_k = 4

    clip_metrics = evaluate_clip(
        images, image_ids, texts, text_image_ids, eval_ids, top_k, [1, 4], 0.9999
    )
    candidates = retrieve_candidates(
        images, image_ids, texts, text_image_ids, eval_ids, top_k, 0.9999
    )
    any_graph_scores = np.random.default_rng(1).normal(size=candidates.clip_scores.shape)
    sweep = sweep_alpha(candidates, any_graph_scores, [1.0], [1, 4])

    for name in ("R@1", "R@4", "MRR@4"):
        assert sweep[1.0][name] == clip_metrics[name]


def test_candidates_come_from_the_evaluated_split_only():
    images, image_ids, texts, text_image_ids = random_split_data()
    eval_ids = np.array([10, 12, 14])

    candidates = retrieve_candidates(images, image_ids, texts, text_image_ids, eval_ids, 3, 0.9999)

    assert set(candidates.candidate_ids.ravel().tolist()) <= {10, 12, 14}
    assert len(candidates.query_embeddings) == 6  # 2 texts per evaluated image


def test_candidate_graph_scores_look_up_vectors_by_image_id():
    # h rows are in a different order than the candidates on purpose.
    h_image_ids = np.array([3, 1, 2])
    image_h = np.array([[0.0, 1.0], [2.0, 0.0], [1.0, 1.0]], dtype=np.float32)

    scores = candidate_graph_scores(ONE_QUERY, image_h, h_image_ids)

    assert scores[0].tolist() == pytest.approx([1.0, 2**-0.5, 0.0])


def test_candidate_without_graph_vector_is_an_error():
    with pytest.raises(ValueError, match="3"):
        candidate_graph_scores(ONE_QUERY, np.ones((2, 2), dtype=np.float32), np.array([1, 2]))


def test_choose_alpha_picks_the_best_metric():
    sweep = {0.0: {"MRR@100": 0.2}, 0.5: {"MRR@100": 0.4}, 1.0: {"MRR@100": 0.3}}

    assert choose_alpha(sweep, "MRR@100") == 0.5


def test_choose_alpha_breaks_ties_towards_plain_clip():
    sweep = {0.0: {"MRR@100": 0.4}, 0.5: {"MRR@100": 0.4}, 1.0: {"MRR@100": 0.3}}

    assert choose_alpha(sweep, "MRR@100") == 0.5
