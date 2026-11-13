"""Tests for image -> text retrieval on 3 images with 2 texts each (spec 5.2, ADR-0043)."""

import numpy as np
import pytest

from mmxai.eval.image_to_text import (
    ranked_text_rows,
    retrieve_text_candidates,
    text_graph_scores,
)
from mmxai.rerank.fusion import fuse_scores

IMAGE_IDS = np.array([10, 11, 12])
TEXT_IMAGE_IDS = np.array([10, 10, 11, 11, 12, 12])


def unit(vector):
    array = np.array(vector, dtype=np.float32)
    return array / np.linalg.norm(array)


def one_hot(position, size=5):
    vector = np.zeros(size, dtype=np.float32)
    vector[position] = 1.0
    return vector


# Text rows 0..5. Row 3 repeats the sentence of row 1 (same vector, other image).
TEXTS = np.stack([one_hot(0), one_hot(1), one_hot(2), one_hot(1), one_hot(3), one_hot(4)])
# Each image points mostly at the first text of its own pair.
IMAGES = np.stack(
    [
        unit([1.0, 0.2, 0, 0, 0]),
        unit([0, 0.3, 1.0, 0, 0]),
        unit([0, 0, 0, 1.0, 0.1]),
    ]
)


def make_candidates(eval_image_ids=IMAGE_IDS, top_k=3):
    return retrieve_text_candidates(
        IMAGES, IMAGE_IDS, TEXTS, TEXT_IMAGE_IDS, eval_image_ids, top_k, 0.9999
    )


def test_relevant_texts_are_the_image_own_texts_and_identical_sentences():
    candidates = make_candidates()
    # Image 10 owns rows 0 and 1; row 3 has the same sentence as row 1, so it counts too.
    assert candidates.relevant[0] == {0, 1, 3}
    # Image 11 owns rows 2 and 3; row 1 has the same sentence as row 3.
    assert candidates.relevant[1] == {1, 2, 3}
    assert candidates.relevant[2] == {4, 5}


def test_candidates_are_ordered_by_clip_cosine():
    candidates = make_candidates()
    assert candidates.candidate_rows[0, 0] == 0  # image 10 is closest to e0
    assert candidates.candidate_rows[1, 0] == 2
    assert candidates.candidate_rows[2, 0] == 4
    assert np.all(np.diff(candidates.clip_scores, axis=1) <= 0)
    assert candidates.query_image_ids.tolist() == [10, 11, 12]


def test_alpha_one_gives_exactly_the_clip_order():
    candidates = make_candidates()
    # Graph scores that would reverse the order if they mattered.
    graph = candidates.clip_scores[:, ::-1].copy()
    final = fuse_scores(candidates.clip_scores, graph, 1.0)
    final[0, 1] = final[0, 2] = 0.5  # equal scores must keep CLIP's order
    order = ranked_text_rows(candidates, final)
    assert order[1] == candidates.candidate_rows[1].tolist()
    assert order[0][1:] == candidates.candidate_rows[0, 1:].tolist()


def test_text_graph_scores_are_cosines_with_h():
    candidates = make_candidates(top_k=1)
    image_h = np.array([[3.0, 4.0, 0, 0, 0], [0, 0, 2.0, 0, 0], [0, 0, 0, 0, 5.0]])
    scores = text_graph_scores(candidates, image_h, IMAGE_IDS)
    # Image 10 -> text row 0 = e0: cos([3,4,..], e0) = 3/5. Image 11 -> row 2 = e2: 1.
    # Image 12 -> row 4 = e3: h has no e3 part, so 0.
    assert scores.shape == (3, 1)
    assert scores[:, 0] == pytest.approx([0.6, 1.0, 0.0])


def test_gallery_holds_only_texts_of_the_evaluated_images():
    candidates = make_candidates(eval_image_ids=np.array([10, 11]), top_k=4)
    assert candidates.text_embeddings.shape == (4, 5)
    assert candidates.query_image_ids.tolist() == [10, 11]
    assert candidates.candidate_rows.max() <= 3
    assert candidates.relevant[0] == {0, 1, 3}


def test_retrieve_text_candidates_rejects_missing_eval_image_id():
    with pytest.raises(ValueError, match="99"):
        retrieve_text_candidates(
            np.eye(3, dtype=np.float32),
            IMAGE_IDS,
            TEXTS,
            TEXT_IMAGE_IDS,
            np.array([10, 99]),
            top_k=2,
            same_text_threshold=0.9999,
        )


def test_text_graph_scores_rejects_missing_query_image_id():
    candidates = make_candidates(top_k=1)
    image_h = np.eye(3, 5, dtype=np.float32)
    with pytest.raises(ValueError, match="12"):
        text_graph_scores(candidates, image_h, np.array([10, 11]))


def test_text_graph_scores_works_across_multiple_chunks():
    candidates = make_candidates(top_k=2)
    image_h = np.array([[3.0, 4.0, 0, 0, 0], [0, 0, 2.0, 0, 0], [0, 0, 0, 0, 5.0]])
    scores = text_graph_scores(candidates, image_h, IMAGE_IDS, chunk_size=2)
    full = text_graph_scores(candidates, image_h, IMAGE_IDS, chunk_size=100)
    assert scores.shape == (3, 2)
    np.testing.assert_allclose(scores, full)
