"""Contains edges from CLIP image-concept similarity, on tiny made-up vectors (ADR-0046).

Three concepts point along the first three axes. Each image vector is chosen so its
nearest concepts are obvious by eye.
"""

import numpy as np
import pytest

from mmxai.graph.clip_edges import clip_topk_concepts

CONCEPTS = ["cat.n.01", "dog.n.01", "sofa.n.01"]
CONCEPT_FEATURES = np.eye(3, dtype=np.float32)
IMAGE_IDS = np.array([10, 20, 30])
IMAGE_FEATURES = np.array(
    [
        [0.9, 0.1, 0.0],  # 10: cat, then dog
        [0.0, 0.2, 0.8],  # 20: sofa, then dog
        [0.1, 0.9, 0.3],  # 30: dog, then sofa
    ],
    dtype=np.float32,
)


def test_each_image_gets_its_k_most_similar_concepts():
    edges = clip_topk_concepts(IMAGE_IDS, IMAGE_FEATURES, CONCEPTS, CONCEPT_FEATURES, k=2)

    assert edges == {
        10: {"cat.n.01", "dog.n.01"},
        20: {"sofa.n.01", "dog.n.01"},
        30: {"dog.n.01", "sofa.n.01"},
    }


def test_similarity_is_cosine_so_vector_length_does_not_matter():
    longer = IMAGE_FEATURES * np.array([[10.0], [1.0], [0.1]], dtype=np.float32)

    edges = clip_topk_concepts(IMAGE_IDS, longer, CONCEPTS, CONCEPT_FEATURES, k=1)

    assert edges == {10: {"cat.n.01"}, 20: {"sofa.n.01"}, 30: {"dog.n.01"}}


def test_an_image_edges_depend_only_on_its_own_vector():
    # No statistic across images: changing image 20 must not move any other edge.
    changed = IMAGE_FEATURES.copy()
    changed[1] = [1.0, 0.0, 0.0]

    before = clip_topk_concepts(IMAGE_IDS, IMAGE_FEATURES, CONCEPTS, CONCEPT_FEATURES, k=1)
    after = clip_topk_concepts(IMAGE_IDS, changed, CONCEPTS, CONCEPT_FEATURES, k=1)

    assert after[20] == {"cat.n.01"}
    assert after[10] == before[10]
    assert after[30] == before[30]


def test_ties_go_to_the_concept_listed_first():
    # Same input must always give the same graph.
    tied = np.array([[1.0, 1.0, 0.0]], dtype=np.float32)

    edges = clip_topk_concepts(np.array([10]), tied, CONCEPTS, CONCEPT_FEATURES, k=1)

    assert edges == {10: {"cat.n.01"}}


def test_small_chunks_give_the_same_edges():
    whole = clip_topk_concepts(IMAGE_IDS, IMAGE_FEATURES, CONCEPTS, CONCEPT_FEATURES, k=2)
    chunked = clip_topk_concepts(
        IMAGE_IDS, IMAGE_FEATURES, CONCEPTS, CONCEPT_FEATURES, k=2, chunk_size=1
    )

    assert chunked == whole


def test_k_larger_than_the_vocabulary_is_an_error():
    with pytest.raises(ValueError, match="k"):
        clip_topk_concepts(IMAGE_IDS, IMAGE_FEATURES, CONCEPTS, CONCEPT_FEATURES, k=4)


def test_k_below_one_is_an_error():
    with pytest.raises(ValueError, match="k"):
        clip_topk_concepts(IMAGE_IDS, IMAGE_FEATURES, CONCEPTS, CONCEPT_FEATURES, k=0)
