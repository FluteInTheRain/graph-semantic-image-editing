"""Load a YAML file into a dataclass, rejecting mistyped or missing keys (ADR-0007).

Kept apart from utils/config.py because that file is copied into the Kaggle
notebook and must stay unchanged (ADR-0041).
"""

from dataclasses import MISSING, fields
from pathlib import Path
from typing import TypeVar

import yaml

ConfigT = TypeVar("ConfigT")


def load_yaml_dataclass(path: str | Path, config_class: type[ConfigT]) -> ConfigT:
    """Read one YAML file into `config_class`. A mistyped key is an error, never ignored."""
    with open(path, encoding="utf-8") as file:
        values = yaml.safe_load(file)

    known_keys = {field.name for field in fields(config_class)}
    required_keys = {field.name for field in fields(config_class) if field.default is MISSING}
    unknown_keys = set(values) - known_keys
    missing_keys = required_keys - set(values)
    if unknown_keys:
        raise ValueError(f"unknown config keys: {sorted(unknown_keys)}")
    if missing_keys:
        raise ValueError(f"missing config keys: {sorted(missing_keys)}")
    return config_class(**values)
