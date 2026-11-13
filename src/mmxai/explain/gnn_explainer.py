"""GNNExplainer (Ying et al., NeurIPS 2019) adapted to one image's graph score."""

from dataclasses import dataclass

import numpy as np

from mmxai.explain.checks import (
    ScoreAndGradient,
    ScoreImage,
    check_gradient_function,
    check_score_function,
)
from mmxai.explain.integrated_gradients import relation_columns
from mmxai.explain.subgraph import relations_in_reach


@dataclass(frozen=True)
class EdgeMask:
    """GNNExplainer's mask value for one relation: how much it is kept, not a score.

    Use it only to pick relations; measure their effect by removing them (fidelity).
    """

    source: int
    target: int
    mask: float


def gnn_explainer(
    edge_index: np.ndarray,
    image_node: int,
    num_layers: int,
    both_directions: bool,
    score_image: ScoreImage,
    score_and_gradient: ScoreAndGradient,
    ranked_graph_score: float,
    epochs: int,
    learning_rate: float,
    size_weight: float,
    entropy_weight: float,
    seed: int,
) -> list[EdgeMask]:
    """Learn a soft mask over nearby relations that keeps the image's score unchanged.

    Minimises mask_loss with plain gradient steps from a seeded start. As in PyG's
    regression mode, the masked score is fitted to the full score rather than to a class.
    Returns each relation with its mask value in [0, 1], sorted from largest to smallest.
    """
    if epochs < 1:
        raise ValueError(f"epochs must be at least 1, got {epochs}")

    check_score_function(edge_index, image_node, num_layers, score_image, ranked_graph_score)
    relations = relations_in_reach(edge_index, image_node, num_layers, both_directions)
    check_gradient_function(edge_index, relations, both_directions, score_image, score_and_gradient)
    columns = relation_columns(edge_index, relations, both_directions)

    # Start every relation mostly present, with a little noise (as in PyG).
    rng = np.random.default_rng(seed)
    logits = rng.normal(1.0, 0.1, size=len(relations))
    for _ in range(epochs):
        _, logit_gradient = mask_loss(
            logits, columns, score_and_gradient, ranked_graph_score, size_weight, entropy_weight
        )
        logits = logits - learning_rate * logit_gradient

    mask = _sigmoid(logits)
    masks = [
        EdgeMask(source, target, float(value))
        for (source, target), value in zip(relations, mask, strict=True)
    ]
    masks.sort(key=lambda edge: edge.mask, reverse=True)
    return masks


def mask_loss(
    logits: np.ndarray,
    columns: list[np.ndarray],
    score_and_gradient: ScoreAndGradient,
    full_score: float,
    size_weight: float,
    entropy_weight: float,
) -> tuple[float, np.ndarray]:
    """GNNExplainer's loss for a mask, and its gradient with respect to the logits.

    loss = (masked score - full score)^2 + size_weight * sum(mask)
           + entropy_weight * mean(mask entropy)
    The size term keeps the explanation small; the entropy term pushes each value to 0 or 1.
    """
    mask = _sigmoid(logits)
    num_edges = len(columns[0]) if columns else 0
    weights = np.ones(num_edges)
    for index, relation_edges in enumerate(columns):
        weights[relation_edges] = mask[index]
    score, gradient = score_and_gradient(weights)
    score_gradient = np.array([gradient[edges].sum() for edges in columns])

    safe = np.clip(mask, 1e-12, 1 - 1e-12)
    entropy = -(safe * np.log(safe) + (1 - safe) * np.log(1 - safe))
    loss = (score - full_score) ** 2 + size_weight * mask.sum() + entropy_weight * entropy.mean()

    mask_gradient = (
        2.0 * (score - full_score) * score_gradient
        + size_weight
        + entropy_weight * np.log((1 - safe) / safe) / len(mask)
    )
    # Chain rule through the sigmoid.
    return float(loss), mask_gradient * mask * (1 - mask)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def strongest_relations(masks: list[EdgeMask], top_m: int) -> list[tuple[int, int]]:
    """The top_m relations by mask value, ready to be removed by fidelity.

    The mask only picks relations; their effect on the score must be measured by removing
    them, so the explanation shows real score changes.
    """
    if top_m < 1:
        raise ValueError(f"top_m must be at least 1, got {top_m}")
    ranked = sorted(masks, key=lambda edge: edge.mask, reverse=True)
    return [(edge.source, edge.target) for edge in ranked[:top_m]]
