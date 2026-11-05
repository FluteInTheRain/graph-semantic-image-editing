"""Which images count as correct answers for a query (ADR-0043). Numpy only."""

import numpy as np
from mmxai.eval.relevance import same_text_relevant_images


def unit(vector: list[float]) -> np.ndarray:
    array = np.array(vector, dtype=np.float32)
    return array / np.linalg.norm(array)


def test_query_with_unique_text_has_only_its_own_image():
    embeddings = np.stack([unit([1, 0, 0]), unit([0, 1, 0])])
    image_ids = np.array([10, 20])

    relevant = same_text_relevant_images(embeddings, image_ids, threshold=0.9999)

    assert relevant == [{10}, {20}]


def test_same_text_in_two_images_makes_both_relevant():
    # Texts 0 and 2 are the same sentence, found in images 10 and 30.
    embeddings = np.stack([unit([1, 0, 0]), unit([0, 1, 0]), unit([1, 0, 0])])
    image_ids = np.array([10, 20, 30])

    relevant = same_text_relevant_images(embeddings, image_ids, threshold=0.9999)

    assert relevant == [{10, 30}, {20}, {10, 30}]


def test_tiny_numeric_noise_still_counts_as_same_text():
    # The same sentence encoded in two GPU batches can differ around 1e-6.
    embeddings = np.stack([unit([1, 0, 0]), unit([1, 1e-6, 0])])
    image_ids = np.array([10, 20])

    relevant = same_text_relevant_images(embeddings, image_ids, threshold=0.9999)

    assert relevant == [{10, 20}, {10, 20}]


def test_similar_but_different_text_is_not_relevant():
    # Cosine about 0.995: close in meaning, but not the same sentence.
    embeddings = np.stack([unit([1, 0, 0]), unit([1, 0.1, 0])])
    image_ids = np.array([10, 20])

    relevant = same_text_relevant_images(embeddings, image_ids, threshold=0.9999)

    assert relevant == [{10}, {20}]


def test_result_does_not_depend_on_chunk_size():
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(9, 4)).astype(np.float32)
    embeddings[5] = embeddings[1]
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    image_ids = np.arange(9) // 3  # three texts per image

    small_chunks = same_text_relevant_images(embeddings, image_ids, 0.9999, chunk_size=2)
    one_chunk = same_text_relevant_images(embeddings, image_ids, 0.9999, chunk_size=100)

    assert small_chunks == one_chunk
    assert small_chunks[1] == {0, 1}
