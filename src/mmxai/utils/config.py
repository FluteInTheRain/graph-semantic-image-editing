"""Read feature extraction configs from YAML into a typed dataclass (ADR-0007)."""

from dataclasses import MISSING, dataclass, fields
from pathlib import Path

import yaml

VISUAL_GENOME = "visual_genome"
FLICKR30K = "flickr30k"


@dataclass(frozen=True)
class FeatureConfig:
    """All settings of one CLIP feature extraction run."""

    dataset: str
    dataset_version: str
    data_dir: str
    output_root: str
    model: str
    pretrained: str
    device: str
    image_batch_size: int
    text_batch_size: int
    num_workers: int
    max_failed_image_fraction: float
    images_dir: str | None = None
    regions_per_image: int | None = None
    sampling_seed: int | None = None


def validate_feature_config(config: FeatureConfig) -> None:
    """Reject settings that do not make sense for the chosen dataset."""
    if config.dataset == VISUAL_GENOME:
        if config.regions_per_image is None or config.sampling_seed is None:
            raise ValueError("visual_genome needs regions_per_image and sampling_seed")
    elif config.dataset == FLICKR30K:
        if config.regions_per_image is not None or config.sampling_seed is not None:
            raise ValueError("flickr30k uses every caption, so it takes no sampling settings")
    else:
        raise ValueError(f"unsupported dataset: {config.dataset}")


def load_feature_config(path: str | Path) -> FeatureConfig:
    """Load and validate one YAML config. A mistyped key is an error, never ignored."""
    with open(path, encoding="utf-8") as file:
        values = yaml.safe_load(file)

    known_keys = {field.name for field in fields(FeatureConfig)}
    required_keys = {field.name for field in fields(FeatureConfig) if field.default is MISSING}
    unknown_keys = set(values) - known_keys
    missing_keys = required_keys - set(values)
    if unknown_keys:
        raise ValueError(f"unknown config keys: {sorted(unknown_keys)}")
    if missing_keys:
        raise ValueError(f"missing config keys: {sorted(missing_keys)}")

    config = FeatureConfig(**values)
    validate_feature_config(config)
    return config
