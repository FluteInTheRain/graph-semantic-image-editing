"""Saved plain-CLIP vector stores: one per split, never mixing splits."""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("faiss")

from mmxai.data.splits import save_split  # noqa: E402
from mmxai.features.cache import save_features  # noqa: E402
from mmxai.index.image_index import load_image_index, search_images  # noqa: E402
from mmxai.index.vector_store import (  # noqa: E402
    VectorStoreConfig,
    build_vector_stores,
    load_vector_store_config,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def make_config(tmp_path: Path, splits: list[str]) -> VectorStoreConfig:
    """Six images 1..6 with orthogonal embeddings; train 1-2, val 3-4, test 5-6."""
    features_dir = tmp_path / "features" / "ViT-B-32__laion2b_s34b_b79k"
    save_features(
        features_dir,
        image_embeddings=np.eye(6, dtype=np.float32),
        image_ids=np.arange(1, 7),
        text_embeddings=np.eye(6, dtype=np.float32),
        text_image_ids=np.arange(1, 7),
        text_id_name="text_ids",
        text_id_values=np.arange(6),
        meta={
            "dataset": "visual_genome",
            "is_partial": False,
            "model": "ViT-B-32",
            "pretrained": "laion2b_s34b_b79k",
            "code_hash": "abc",
        },
    )
    split_file = tmp_path / "split_seed0.json"
    split = {
        "dataset": "visual_genome",
        "seed": 0,
        "train": ["1", "2"],
        "val": ["3", "4"],
        "test": ["5", "6"],
    }
    save_split(split, split_file)
    return VectorStoreConfig(
        dataset="visual_genome",
        split_file=str(split_file),
        features_dir=str(features_dir),
        splits=splits,
        output_root=str(tmp_path / "stores"),
    )


def test_each_store_holds_only_its_split_images(tmp_path):
    folders = build_vector_stores(make_config(tmp_path, ["val", "test"]))

    stores = {folder.name: load_image_index(folder) for folder in folders}
    assert sorted(stores["val"].image_ids.tolist()) == [3, 4]
    assert sorted(stores["test"].image_ids.tolist()) == [5, 6]


def test_store_folder_mirrors_the_feature_folder(tmp_path):
    folders = build_vector_stores(make_config(tmp_path, ["test"]))

    expected = tmp_path / "stores" / "visual_genome" / "ViT-B-32__laion2b_s34b_b79k" / "test"
    assert folders == [expected]


def test_search_in_a_store_returns_the_matching_image(tmp_path):
    folders = build_vector_stores(make_config(tmp_path, ["test"]))
    store = load_image_index(folders[0])

    query = np.eye(6, dtype=np.float32)[[5]]  # the embedding of image 6
    _, ids = search_images(store, query, top_k=1)

    assert ids.tolist() == [[6]]


def test_meta_records_where_the_vectors_came_from(tmp_path):
    folders = build_vector_stores(make_config(tmp_path, ["test"]))

    meta = json.loads((folders[0] / "meta.json").read_text(encoding="utf-8"))
    assert meta["split"] == "test"
    assert meta["num_images"] == 2
    assert meta["features_code_hash"] == "abc"
    assert len(meta["split_sha256"]) == 64


def test_config_rejects_unknown_split_name(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        (REPO_ROOT / "configs" / "vector_store_vg.yaml")
        .read_text(encoding="utf-8")
        .replace("splits: [val, test]", "splits: [val, tset]"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="tset"):
        load_vector_store_config(path)


def test_committed_config_builds_val_and_test():
    config = load_vector_store_config(REPO_ROOT / "configs" / "vector_store_vg.yaml")

    assert config.splits == ["val", "test"]
