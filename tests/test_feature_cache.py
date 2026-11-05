"""Tests for saving and loading the embedding cache. Numpy only, no torch."""

import json
from pathlib import Path

import numpy as np
import pytest

from mmxai.features.cache import (
    check_meta_matches_config,
    compute_code_hash,
    feature_dir,
    load_image_features,
    load_meta,
    load_text_features,
    save_features,
)
from mmxai.utils.config import FeatureConfig


def make_config(**changes) -> FeatureConfig:
    values = {
        "dataset": "flickr30k",
        "dataset_version": "karpathy",
        "data_dir": "data/flickr30k",
        "output_root": "artifacts/features",
        "model": "ViT-B-32",
        "pretrained": "laion2b_s34b_b79k",
        "device": "cpu",
        "image_batch_size": 2,
        "text_batch_size": 2,
        "num_workers": 0,
        "max_failed_image_fraction": 0.001,
    }
    values.update(changes)
    return FeatureConfig(**values)


def save_small_features(output_dir: Path) -> None:
    save_features(
        output_dir,
        image_embeddings=np.eye(2, dtype=np.float64),
        image_ids=np.array([10, 20]),
        text_embeddings=np.eye(2, dtype=np.float64),
        text_image_ids=np.array([10, 20]),
        text_id_name="caption_index",
        text_id_values=np.array([0, 0]),
        meta={"dataset": "flickr30k"},
    )


def test_save_and_load_round_trip(tmp_path):
    output_dir = tmp_path / "features"
    save_small_features(output_dir)

    image_embeddings, image_ids = load_image_features(output_dir)
    texts = load_text_features(output_dir)

    assert image_embeddings.dtype == np.float32
    assert image_ids.dtype == np.int64
    assert image_ids.tolist() == [10, 20]
    assert np.array_equal(image_embeddings, np.eye(2))
    assert texts["embeddings"].dtype == np.float32
    assert texts["image_ids"].tolist() == [10, 20]
    assert texts["caption_index"].tolist() == [0, 0]
    assert load_meta(output_dir) == {"dataset": "flickr30k"}


def test_saved_arrays_load_without_pickle(tmp_path):
    output_dir = tmp_path / "features"
    save_small_features(output_dir)
    with np.load(output_dir / "text_features.npz", allow_pickle=False) as data:
        assert data["image_ids"].shape == (2,)


def test_save_features_refuses_existing_output(tmp_path):
    output_dir = tmp_path / "features"
    save_small_features(output_dir)
    with pytest.raises(FileExistsError):
        save_small_features(output_dir)


def test_no_temporary_folder_left_after_save(tmp_path):
    output_dir = tmp_path / "features"
    save_small_features(output_dir)
    assert [path.name for path in tmp_path.iterdir()] == ["features"]


def test_load_features_requires_meta(tmp_path):
    output_dir = tmp_path / "features"
    save_small_features(output_dir)
    (output_dir / "meta.json").unlink()
    with pytest.raises(FileNotFoundError):
        load_image_features(output_dir)


def test_feature_dir_separates_partial_runs():
    config = make_config()
    assert feature_dir(config, is_partial=False) == Path(
        "artifacts/features/flickr30k/ViT-B-32__laion2b_s34b_b79k"
    )
    assert feature_dir(config, is_partial=True) == Path(
        "artifacts/features/flickr30k_partial/ViT-B-32__laion2b_s34b_b79k"
    )


def test_check_meta_accepts_same_settings():
    config = make_config()
    meta = {
        "dataset": "flickr30k",
        "dataset_version": "karpathy",
        "model": "ViT-B-32",
        "pretrained": "laion2b_s34b_b79k",
        "regions_per_image": None,
        "sampling_seed": None,
    }
    check_meta_matches_config(meta, config)


def test_check_meta_rejects_other_model():
    config = make_config()
    meta = {
        "dataset": "flickr30k",
        "dataset_version": "karpathy",
        "model": "ViT-L-14",
        "pretrained": "laion2b_s34b_b79k",
        "regions_per_image": None,
        "sampling_seed": None,
    }
    with pytest.raises(ValueError, match="model"):
        check_meta_matches_config(meta, config)


def test_code_hash_changes_with_content(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    first = compute_code_hash(tmp_path, ["a.py"])
    assert compute_code_hash(tmp_path, ["a.py"]) == first

    (tmp_path / "a.py").write_text("x = 2\n")
    assert compute_code_hash(tmp_path, ["a.py"]) != first


def test_meta_is_valid_json_on_disk(tmp_path):
    output_dir = tmp_path / "features"
    save_small_features(output_dir)
    assert json.loads((output_dir / "meta.json").read_text()) == {"dataset": "flickr30k"}


def failing_check(folder: Path) -> None:
    raise ValueError("bad output")


def test_failed_check_removes_all_output(tmp_path):
    output_dir = tmp_path / "features"
    with pytest.raises(ValueError, match="bad output"):
        save_features(
            output_dir,
            image_embeddings=np.eye(2),
            image_ids=np.array([10, 20]),
            text_embeddings=np.eye(2),
            text_image_ids=np.array([10, 20]),
            text_id_name="caption_index",
            text_id_values=np.array([0, 0]),
            meta={"dataset": "flickr30k"},
            check=failing_check,
        )
    assert list(tmp_path.iterdir()) == []


def test_check_runs_before_the_final_folder_exists(tmp_path):
    output_dir = tmp_path / "features"
    seen = []

    def record_state(folder: Path) -> None:
        seen.append((output_dir.exists(), (folder / "meta.json").exists()))

    save_features(
        output_dir,
        image_embeddings=np.eye(2),
        image_ids=np.array([10, 20]),
        text_embeddings=np.eye(2),
        text_image_ids=np.array([10, 20]),
        text_id_name="caption_index",
        text_id_values=np.array([0, 0]),
        meta={"dataset": "flickr30k"},
        check=record_state,
    )
    assert seen == [(False, True)]
    assert output_dir.exists()
