"""Build and save the plain-CLIP image vector store of each split (no graph).

Decision: one store (folder) per split instead of one index of all images.
  Why: a search in the test store can then never return a train image (ADR-0042).
  Decided by: Claude Code, 2026-11-05, after FluteInTheRain asked to save the
  vector store (not separately reviewed).

Decision: build only val and test (config `splits`).
  Why: nothing searches train yet. Stores rebuild in seconds from the Kaggle
  features, so Kaggle stays the single shared source (ADR-0041).
  Decided by: Claude Code, 2026-11-05 (not separately reviewed).
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import faiss
import numpy as np

from mmxai.data.splits import SPLIT_NAMES, load_split
from mmxai.features.cache import load_image_features, load_meta
from mmxai.index.image_index import build_image_index, save_image_index
from mmxai.utils.runs import file_sha256
from mmxai.utils.yaml_config import load_yaml_dataclass


@dataclass(frozen=True)
class VectorStoreConfig:
    """Which feature set and split to index, and where to write the stores."""

    dataset: str
    split_file: str
    features_dir: str
    splits: list[str]
    output_root: str


def load_vector_store_config(path: str | Path) -> VectorStoreConfig:
    """Load the YAML config and reject unknown split names."""
    config = load_yaml_dataclass(path, VectorStoreConfig)
    for name in config.splits:
        if name not in SPLIT_NAMES:
            raise ValueError(f"unknown split {name!r}; expected one of {SPLIT_NAMES}")
    return config


def store_dir(config: VectorStoreConfig, split_name: str) -> Path:
    """<output_root>/<dataset>/<model>__<pretrained>/<split>/, mirroring the feature folders."""
    feature_folder = Path(config.features_dir).name
    return Path(config.output_root) / config.dataset / feature_folder / split_name


def select_split_images(
    embeddings: np.ndarray, image_ids: np.ndarray, split_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the embeddings and ids of the split's images; every split image must exist."""
    missing = np.setdiff1d(split_ids, image_ids)
    if len(missing) > 0:
        raise ValueError(f"{len(missing)} split images have no embedding, e.g. {missing[0]}")
    rows = np.isin(image_ids, split_ids)
    return embeddings[rows], image_ids[rows]


def build_vector_stores(config: VectorStoreConfig) -> list[Path]:
    """Index the images of each configured split and save one store per split."""
    feature_meta = load_meta(config.features_dir)
    if feature_meta["dataset"] != config.dataset or feature_meta["is_partial"]:
        raise ValueError(f"{config.features_dir} is not a complete {config.dataset} feature set")

    embeddings, image_ids = load_image_features(config.features_dir)
    split = load_split(config.split_file)

    folders = []
    for split_name in config.splits:
        split_embeddings, split_image_ids = select_split_images(
            embeddings, image_ids, split[split_name]
        )
        image_index = build_image_index(split_embeddings, split_image_ids)
        # Enough for someone who downloads only this folder to know what it holds.
        meta = {
            "dataset": config.dataset,
            "split": split_name,
            "num_images": len(split_image_ids),
            "split_file": config.split_file,
            "split_sha256": file_sha256(config.split_file),
            "model": feature_meta["model"],
            "pretrained": feature_meta["pretrained"],
            "features_code_hash": feature_meta["code_hash"],
            "index_type": "IndexFlatIP",
            "similarity": "inner product = cosine, embeddings are L2-normalised",
            "faiss_version": faiss.__version__,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        folder = store_dir(config, split_name)
        save_image_index(image_index, folder, meta)
        folders.append(folder)
    return folders
