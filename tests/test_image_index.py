"""FAISS image index: exact cosine search that returns image ids (ADR-0043)."""

import numpy as np
import pytest

pytest.importorskip("faiss")

from mmxai.index.image_index import (  # noqa: E402
    build_image_index,
    load_image_index,
    save_image_index,
    search_images,
)


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


def test_saved_index_gives_the_same_results(tmp_path):
    rng = np.random.default_rng(1)
    images = unit_rows(rng.normal(size=(20, 8)))
    queries = unit_rows(rng.normal(size=(3, 8)))
    original = build_image_index(images, np.arange(500, 520))

    save_image_index(original, tmp_path / "store", meta={"split": "test"})
    loaded = load_image_index(tmp_path / "store")

    assert (
        search_images(loaded, queries, 5)[1].tolist()
        == search_images(original, queries, 5)[1].tolist()
    )


def test_save_refuses_to_overwrite(tmp_path):
    index = build_image_index(unit_rows([[1, 0], [0, 1]]), np.array([1, 2]))
    save_image_index(index, tmp_path / "store", meta={})

    with pytest.raises(FileExistsError):
        save_image_index(index, tmp_path / "store", meta={})


def test_load_rejects_ids_that_do_not_match_the_index(tmp_path):
    index = build_image_index(unit_rows([[1, 0], [0, 1]]), np.array([1, 2]))
    save_image_index(index, tmp_path / "store", meta={})
    np.save(tmp_path / "store" / "image_ids.npy", np.array([1, 2, 3]))

    with pytest.raises(ValueError, match="rows"):
        load_image_index(tmp_path / "store")
