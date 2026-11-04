"""Save and load cached embeddings as .npz + meta.json (ADR-0041). Needs numpy only."""

import hashlib
import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np

from mmxai.utils.config import FeatureConfig

IMAGE_FILE = "image_features.npz"
TEXT_FILE = "text_features.npz"
META_FILE = "meta.json"

# A cache is only valid for the settings that produced it.
META_FIELDS_TO_CHECK = [
    "dataset",
    "dataset_version",
    "model",
    "pretrained",
    "regions_per_image",
    "sampling_seed",
]


def feature_dir(config: FeatureConfig, is_partial: bool) -> Path:
    """Return the output folder. Trial runs get their own folder so they are never published."""
    dataset_folder = config.dataset
    if is_partial:
        dataset_folder = f"{config.dataset}_partial"
    return Path(config.output_root) / dataset_folder / f"{config.model}__{config.pretrained}"


def save_features(
    output_dir: Path,
    image_embeddings: np.ndarray,
    image_ids: np.ndarray,
    text_embeddings: np.ndarray,
    text_image_ids: np.ndarray,
    text_id_name: str,
    text_id_values: np.ndarray,
    meta: dict,
    check: Callable[[Path], None] | None = None,
) -> None:
    """Write one complete feature set. Refuses to overwrite an existing one.

    `check` runs on the finished files before they get their final name, so
    output that fails it is removed and can never be mistaken for a real result.
    """
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"{output_dir} already exists; delete it to extract again")

    # Writing to a temporary folder first means an interrupted run can never
    # leave a folder that looks complete.
    temp_dir = output_dir.with_name(output_dir.name + ".tmp")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)

    np.savez(
        temp_dir / IMAGE_FILE,
        embeddings=np.asarray(image_embeddings, dtype=np.float32),
        image_ids=np.asarray(image_ids, dtype=np.int64),
    )
    text_arrays = {
        "embeddings": np.asarray(text_embeddings, dtype=np.float32),
        "image_ids": np.asarray(text_image_ids, dtype=np.int64),
        text_id_name: np.asarray(text_id_values, dtype=np.int64),
    }
    np.savez(temp_dir / TEXT_FILE, **text_arrays)
    with open(temp_dir / META_FILE, "w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2)

    if check is not None:
        try:
            check(temp_dir)
        except Exception:
            shutil.rmtree(temp_dir)
            raise

    os.replace(temp_dir, output_dir)


def load_meta(output_dir: Path) -> dict:
    """Read meta.json. A folder without it is not a complete feature set."""
    with open(Path(output_dir) / META_FILE, encoding="utf-8") as file:
        return json.load(file)


def load_image_features(output_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (embeddings, image_ids)."""
    load_meta(output_dir)
    with np.load(Path(output_dir) / IMAGE_FILE, allow_pickle=False) as data:
        return data["embeddings"], data["image_ids"]


def load_text_features(output_dir: Path) -> dict[str, np.ndarray]:
    """Return every array of text_features.npz, keyed by name."""
    load_meta(output_dir)
    with np.load(Path(output_dir) / TEXT_FILE, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def check_meta_matches_config(meta: dict, config: FeatureConfig) -> None:
    """Raise ValueError if the cache was made with different settings (ADR-0008)."""
    for field_name in META_FIELDS_TO_CHECK:
        expected = getattr(config, field_name)
        if meta.get(field_name) != expected:
            raise ValueError(
                f"cache has {field_name}={meta.get(field_name)!r}, config has {expected!r}"
            )


def compute_code_hash(package_root: Path, module_paths: list[str]) -> str:
    """Hash the modules that ran, so two runs can be checked to use the same code."""
    digest = hashlib.sha256()
    for module_path in module_paths:
        digest.update(module_path.encode("utf-8"))
        digest.update((Path(package_root) / module_path).read_bytes())
    return digest.hexdigest()
