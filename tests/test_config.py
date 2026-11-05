"""Tests for reading feature extraction configs."""

from pathlib import Path

import pytest
import yaml

from mmxai.utils.config import load_feature_config

CONFIG_DIR = Path(__file__).parent.parent / "configs"


def write_config(tmp_path: Path, **changes) -> Path:
    """Write a valid Flickr30k config with some keys changed (None removes a key)."""
    values = yaml.safe_load((CONFIG_DIR / "features_flickr30k.yaml").read_text())
    values.update(changes)
    values = {key: value for key, value in values.items() if value is not None}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(values))
    return path


def test_loads_visual_genome_config():
    config = load_feature_config(CONFIG_DIR / "features_visual_genome.yaml")
    assert config.dataset == "visual_genome"
    assert config.regions_per_image == 5
    assert config.sampling_seed == 0


def test_loads_flickr30k_config():
    config = load_feature_config(CONFIG_DIR / "features_flickr30k.yaml")
    assert config.dataset == "flickr30k"
    assert config.regions_per_image is None
    assert config.max_failed_image_fraction == 0.001


def test_unknown_key_raises(tmp_path):
    path = write_config(tmp_path, batch_sizee=4)
    with pytest.raises(ValueError, match="unknown"):
        load_feature_config(path)


def test_missing_required_key_raises(tmp_path):
    path = write_config(tmp_path, model=None)
    with pytest.raises(ValueError, match="missing"):
        load_feature_config(path)


def test_visual_genome_requires_sampling_fields(tmp_path):
    path = write_config(tmp_path, dataset="visual_genome")
    with pytest.raises(ValueError, match="regions_per_image"):
        load_feature_config(path)


def test_flickr30k_rejects_sampling_fields(tmp_path):
    path = write_config(tmp_path, regions_per_image=5, sampling_seed=0)
    with pytest.raises(ValueError, match="flickr30k"):
        load_feature_config(path)


def test_unknown_dataset_raises(tmp_path):
    path = write_config(tmp_path, dataset="coco")
    with pytest.raises(ValueError, match="coco"):
        load_feature_config(path)
