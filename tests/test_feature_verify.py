"""Tests for the checks that run on extracted embeddings. Numpy only, no torch."""

from pathlib import Path

import numpy as np
import pytest

from mmxai.features.cache import save_features
from mmxai.features.verify import FeatureVerificationError, verify_features

NUM_IMAGES = 20
DIM = 32


def normalize(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def make_arrays(texts_per_image: int) -> dict:
    """Build aligned data: each text is its image's vector plus a little noise."""
    rng = np.random.default_rng(0)
    image_embeddings = normalize(rng.normal(size=(NUM_IMAGES, DIM)))
    image_ids = np.arange(100, 100 + NUM_IMAGES)
    text_image_ids = np.repeat(image_ids, texts_per_image)
    rows = np.repeat(np.arange(NUM_IMAGES), texts_per_image)
    noise = 0.05 * rng.normal(size=(len(rows), DIM))
    return {
        "image_embeddings": image_embeddings,
        "image_ids": image_ids,
        "text_embeddings": normalize(image_embeddings[rows] + noise),
        "text_image_ids": text_image_ids,
    }


def write_flickr(output_dir: Path, arrays: dict, is_partial: bool = False) -> Path:
    num_texts = len(arrays["text_image_ids"])
    save_features(
        output_dir,
        **arrays,
        text_id_name="caption_index",
        text_id_values=np.zeros(num_texts, dtype=np.int64),
        meta={
            "dataset": "flickr30k",
            "num_images": len(arrays["image_ids"]),
            "num_texts": num_texts,
            "regions_per_image": None,
            "is_partial": is_partial,
        },
    )
    return output_dir


def write_visual_genome(output_dir: Path, arrays: dict, is_partial: bool = False) -> Path:
    num_texts = len(arrays["text_image_ids"])
    save_features(
        output_dir,
        **arrays,
        text_id_name="text_ids",
        text_id_values=np.arange(num_texts),
        meta={
            "dataset": "visual_genome",
            "num_images": len(arrays["image_ids"]),
            "num_texts": num_texts,
            "regions_per_image": 5,
            "is_partial": is_partial,
        },
    )
    return output_dir


def test_valid_flickr_output_passes(tmp_path):
    verify_features(write_flickr(tmp_path / "out", make_arrays(5)))


def test_valid_visual_genome_output_passes(tmp_path):
    verify_features(write_visual_genome(tmp_path / "out", make_arrays(3)))


def test_row_count_mismatch_fails(tmp_path):
    arrays = make_arrays(5)
    arrays["image_ids"] = arrays["image_ids"][:-1]
    with pytest.raises(FeatureVerificationError, match="image rows"):
        verify_features(write_flickr(tmp_path / "out", arrays))


def test_count_different_from_meta_fails(tmp_path):
    output_dir = tmp_path / "out"
    arrays = make_arrays(5)
    num_texts = len(arrays["text_image_ids"])
    save_features(
        output_dir,
        **arrays,
        text_id_name="caption_index",
        text_id_values=np.zeros(num_texts, dtype=np.int64),
        meta={
            "dataset": "flickr30k",
            "num_images": NUM_IMAGES + 1,
            "num_texts": num_texts,
            "regions_per_image": None,
            "is_partial": False,
        },
    )
    with pytest.raises(FeatureVerificationError, match="num_images"):
        verify_features(output_dir)


def test_nan_fails(tmp_path):
    arrays = make_arrays(5)
    arrays["image_embeddings"][0, 0] = np.nan
    with pytest.raises(FeatureVerificationError, match="NaN"):
        verify_features(write_flickr(tmp_path / "out", arrays))


def test_unnormalized_vector_fails(tmp_path):
    arrays = make_arrays(5)
    arrays["text_embeddings"][0] *= 2
    with pytest.raises(FeatureVerificationError, match="unit length"):
        verify_features(write_flickr(tmp_path / "out", arrays))


def test_duplicate_image_id_fails(tmp_path):
    arrays = make_arrays(5)
    arrays["image_ids"][1] = arrays["image_ids"][0]
    with pytest.raises(FeatureVerificationError, match="duplicate"):
        verify_features(write_flickr(tmp_path / "out", arrays))


def test_text_pointing_to_unknown_image_fails(tmp_path):
    arrays = make_arrays(5)
    arrays["text_image_ids"][0] = 999999
    with pytest.raises(FeatureVerificationError, match="unknown image"):
        verify_features(write_flickr(tmp_path / "out", arrays))


def test_more_than_five_regions_fails(tmp_path):
    arrays = make_arrays(6)
    with pytest.raises(FeatureVerificationError, match="more than 5"):
        verify_features(write_visual_genome(tmp_path / "out", arrays))


def test_flickr_requires_exactly_five_captions_when_complete(tmp_path):
    arrays = make_arrays(4)
    with pytest.raises(FeatureVerificationError, match="exactly 5"):
        verify_features(write_flickr(tmp_path / "out", arrays))


def test_partial_flickr_allows_fewer_captions(tmp_path):
    arrays = make_arrays(4)
    verify_features(write_flickr(tmp_path / "out", arrays, is_partial=True))


def test_misaligned_rows_fail_sanity_check(tmp_path):
    arrays = make_arrays(5)
    arrays["text_embeddings"] = arrays["text_embeddings"][::-1]
    with pytest.raises(FeatureVerificationError, match="misaligned"):
        verify_features(write_flickr(tmp_path / "out", arrays))


def test_shuffle_that_changes_nothing_is_not_called_misaligned(tmp_path):
    # With few rows a shuffle can leave every pair unchanged; that proves nothing.
    arrays = make_arrays(5)
    keep = arrays["text_image_ids"] == arrays["image_ids"][0]
    arrays["text_embeddings"] = arrays["text_embeddings"][keep]
    arrays["text_image_ids"] = arrays["text_image_ids"][keep]
    verify_features(write_flickr(tmp_path / "out", arrays, is_partial=True))
