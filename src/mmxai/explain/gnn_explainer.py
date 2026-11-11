"""GNNExplainer (Ying et al., NeurIPS 2019) adapted to one image's graph score."""

import numpy as np

from mmxai.explain.integrated_gradients import ScoreAndGradient, relation_columns
from mmxai.explain.occlusion import EdgeContribution
from mmxai.explain.subgraph import relations_in_reach


def gnn_explainer(
    edge_index: np.ndarray,
    image_node: int,
    num_layers: int,
    both_directions: bool,
    score_and_gradient: ScoreAndGradient,
    epochs: int,
    learning_rate: float,
    size_weight: float,
    entropy_weight: float,
    seed: int,
) -> list[EdgeContribution]:
    """Learn a soft mask over nearby relations that keeps the image's score unchanged.

    The loss is (masked score - full score)^2, plus size_weight * mask size, plus
    entropy_weight * mask entropy; the last two push the mask to be small and close to
    0 or 1. As in PyG's regression mode, a score is kept rather than a class.
    Returns each relation with its mask value in [0, 1], sorted from largest to smallest.
    """
    if epochs < 1:
        raise ValueError(f"epochs must be at least 1, got {epochs}")

    relations = relations_in_reach(edge_index, image_node, num_layers, both_directions)
    columns = relation_columns(edge_index, relations, both_directions)
    full_score, _ = score_and_gradient(np.ones(edge_index.shape[1]))

    # Start every relation mostly present, with a little noise (as in PyG).
    rng = np.random.default_rng(seed)
    logits = rng.normal(1.0, 0.1, size=len(relations))

    for _ in range(epochs):
        mask = 1.0 / (1.0 + np.exp(-logits))
        weights = np.ones(edge_index.shape[1])
        for index, relation_edges in enumerate(columns):
            weights[relation_edges] = mask[index]
        score, gradient = score_and_gradient(weights)
        score_gradient = np.array([gradient[edges].sum() for edges in columns])

        # Gradient of each loss term with respect to the mask.
        fit = 2.0 * (score - full_score) * score_gradient
        size = size_weight * np.ones(len(relations))
        safe_mask = np.clip(mask, 1e-6, 1 - 1e-6)
        entropy = entropy_weight * np.log((1 - safe_mask) / safe_mask) / len(relations)
        mask_gradient = fit + size + entropy

        # Chain rule through the sigmoid, then a plain gradient step.
        logits -= learning_rate * mask_gradient * mask * (1 - mask)

    mask = 1.0 / (1.0 + np.exp(-logits))
    contributions = [
        EdgeContribution(source, target, float(value))
        for (source, target), value in zip(relations, mask, strict=True)
    ]
    contributions.sort(key=lambda edge: edge.contribution, reverse=True)
    return contributions
