"""Plain CLIP evaluation on tiny synthetic data (ADR-0043)."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("faiss")

from mmxai.data.splits import save_split  # noqa: E402
from mmxai.eval.clip_baseline import (  # noqa: E402
    ClipEvalConfig,
    evaluate_clip,
    load_clip_eval_config,
    run_clip_evaluation,
)
from mmxai.features.cache import save_features  # noqa: E402
from mmxai.utils.runs import make_run_dir  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def unit_rows(array: np.ndarray) -> np.ndarray:
    return array / np.linalg.norm(array, axis=1, keepdims=True)


def make_data():
    """Six images with orthogonal embeddings; each image has two texts equal to its own vector."""
    image_embeddings = np.eye(6, dtype=np.float32)
    image_ids = np.arange(1, 7)
    text_embeddings = np.repeat(image_embeddings, 2, axis=0)
    text_image_ids = np.repeat(image_ids, 2)
    return image_embeddings, image_ids, text_embeddings, text_image_ids


def test_perfect_embeddings_give_perfect_scores():
    images, image_ids, texts, text_image_ids = make_data()

    metrics = evaluate_clip(
        images, image_ids, texts, text_image_ids, np.array([1, 2, 3]), 3, [1, 3], 0.9999
    )

    assert metrics["R@1"] == 1.0
    assert metrics["MRR@3"] == 1.0
    assert metrics["num_queries"] == 6
    assert metrics["num_images"] == 3


def test_only_images_of_the_evaluated_split_are_searched():
    # A query of image 1 would match image 4 best, but image 4 is not in the split.
    images, image_ids, texts, text_image_ids = make_data()
    texts = texts.copy()
    texts[0] = unit_rows(np.array([[0.1, 0, 0, 1, 0, 0]], dtype=np.float32))[0]

    metrics = evaluate_clip(
        images, image_ids, texts, text_image_ids, np.array([1, 2, 3]), 3, [1], 0.9999
    )

    # Image 4 is absent, so the query still finds image 1 first.
    assert metrics["R@1"] == 1.0
    assert metrics["num_queries"] == 6


def test_query_whose_image_ranks_second_by_hand():
    images, image_ids, texts, text_image_ids = make_data()
    texts = texts.copy()
    # Closer to image 2 than to its own image 1: rank 2 for this one query.
    texts[0] = unit_rows(np.array([[0.5, 1, 0, 0, 0, 0]], dtype=np.float32))[0]

    metrics = evaluate_clip(
        images, image_ids, texts, text_image_ids, np.array([1, 2, 3]), 3, [1, 3], 0.9999
    )

    assert metrics["R@1"] == pytest.approx(5 / 6)
    assert metrics["R@3"] == 1.0
    assert metrics["MRR@3"] == pytest.approx((5 + 0.5) / 6)


def test_split_image_without_embedding_is_an_error():
    images, image_ids, texts, text_image_ids = make_data()

    with pytest.raises(ValueError, match="no embedding"):
        evaluate_clip(images, image_ids, texts, text_image_ids, np.array([1, 99]), 1, [1], 0.9999)


def test_config_rejects_train_as_eval_split(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        (REPO_ROOT / "configs" / "clip_vg.yaml")
        .read_text(encoding="utf-8")
        .replace("eval_split: val", "eval_split: train"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="eval_split"):
        load_clip_eval_config(path)


def test_committed_config_evaluates_val_by_default():
    config = load_clip_eval_config(REPO_ROOT / "configs" / "clip_vg.yaml")

    assert config.eval_split == "val"
    assert config.top_k == 100


def test_run_dir_is_never_reused(tmp_path):
    now = datetime(2026, 11, 5, 9, 30)
    first = make_run_dir(tmp_path, "clip_vg", 0, now)

    assert first.name == "20261105-0930_clip_vg_seed0"
    with pytest.raises(FileExistsError):
        make_run_dir(tmp_path, "clip_vg", 0, now)


def test_run_writes_traceable_files(tmp_path):
    images, image_ids, texts, text_image_ids = make_data()
    features_dir = tmp_path / "features"
    save_features(
        features_dir,
        image_embeddings=images,
        image_ids=image_ids,
        text_embeddings=texts,
        text_image_ids=text_image_ids,
        text_id_name="text_ids",
        text_id_values=np.arange(12),
        meta={"dataset": "visual_genome", "is_partial": False, "code_hash": "abc"},
    )
    split_file = tmp_path / "split_seed0.json"
    save_split(
        {
            "dataset": "visual_genome",
            "seed": 0,
            "train": ["1", "2"],
            "val": ["3", "4"],
            "test": ["5", "6"],
        },
        split_file,
    )
    config = ClipEvalConfig(
        experiment="clip_vg",
        dataset="visual_genome",
        split_file=str(split_file),
        features_dir=str(features_dir),
        eval_split="test",
        top_k=2,
        recall_ks=[1],
        same_text_threshold=0.9999,
        output_dir=str(tmp_path / "runs"),
    )

    run_dir = run_clip_evaluation(config, seed=0)

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["split"] == "test"
    assert metrics["num_images"] == 2
    assert len(metrics["split_sha256"]) == 64
    assert (run_dir / "config.yaml").exists()
    assert "git_commit" in json.loads((run_dir / "env.json").read_text(encoding="utf-8"))
