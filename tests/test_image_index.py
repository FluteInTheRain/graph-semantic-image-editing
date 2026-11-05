"""FAISS image index: exact cosine search that returns image ids (ADR-0043)."""

import numpy as np
import pytest

pytest.importorskip("faiss")

from mmxai.index.image_index import build_image_index, search_images  # noqa: E402


def unit_rows(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    return array / np.linalg.norm(array, axis=1, keepdims=True)


def test_search_returns_image_ids_in_order_of_cosine():
    images = unit_rows([[1, 0], [0, 1], [1, 1]])
    index = build_image_index(images, np.array([10, 20, 30]))

    scores, ids = search_images(index, unit_rows([[1, 0.1]]), top_k=3)

    assert ids.tolist() == [[10, 30, 20]]
    assert scores[0, 0] == pytest.approx(float(unit_rows([[1, 0.1]])[0] @ images[0]), abs=1e-6)


def test_search_matches_brute_force_numpy():
    rng = np.random.default_rng(0)
    images = unit_rows(rng.normal(size=(50, 16)))
    queries = unit_rows(rng.normal(size=(7, 16)))
    image_ids = np.arange(1000, 1050)
    index = build_image_index(images, image_ids)

    _, ids = search_images(index, queries, top_k=5)

    expected = image_ids[np.argsort(-(queries @ images.T), axis=1)[:, :5]]
    assert ids.tolist() == expected.tolist()


def test_top_k_larger_than_the_gallery_is_an_error():
    index = build_image_index(unit_rows([[1, 0], [0, 1]]), np.array([1, 2]))

    with pytest.raises(ValueError, match="only 2 images"):
        search_images(index, unit_rows([[1, 0]]), top_k=3)


def test_ids_and_embeddings_must_have_the_same_length():
    with pytest.raises(ValueError):
        build_image_index(unit_rows([[1, 0], [0, 1]]), np.array([1]))
