"""Settings of the explanation evaluation, read from configs/explain_*.yaml (ADR-0007)."""

from dataclasses import dataclass
from pathlib import Path

from mmxai.utils.yaml_config import load_yaml_dataclass

EXPLAINERS = ("occlusion", "integrated_gradients", "gnn_explainer")


@dataclass(frozen=True)
class ExplainConfig:
    """Every hyperparameter of the xAI module; chosen on val only (PDF section 6)."""

    experiment: str
    eval_split: str
    seeds: list[int]
    explainers: list[str]
    num_layers: int
    both_directions: bool
    top_m: int
    min_contribution: float
    plus_weight: float
    random_repeats: int
    robust_remove_prob: float
    robust_add_back_prob: float
    robust_repeats: int
    ig_steps: int
    gnn_explainer_epochs: int
    gnn_explainer_learning_rate: float
    gnn_explainer_size_weight: float
    gnn_explainer_entropy_weight: float
    output_dir: str


def load_explain_config(path: str | Path) -> ExplainConfig:
    """Load the YAML config and reject values the explain code cannot use."""
    config = load_yaml_dataclass(path, ExplainConfig)
    if config.eval_split not in ("val", "test"):
        raise ValueError(f"eval_split must be val or test, got {config.eval_split!r}")
    unknown = sorted(set(config.explainers) - set(EXPLAINERS))
    if unknown:
        raise ValueError(f"unknown explainers: {unknown}")
    if len(config.seeds) < 3:
        raise ValueError("main experiments need at least 3 seeds (PDF section 6)")
    for name in ("robust_remove_prob", "robust_add_back_prob", "plus_weight"):
        if not 0.0 <= getattr(config, name) <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    return config
