"""Integrated Gradients (Sundararajan et al., ICML 2017) over the relations near an image."""

from collections.abc import Callable

import numpy as np

from mmxai.explain.intervention import mark_remaining_edges
from mmxai.explain.occlusion import EdgeContribution
from mmxai.explain.subgraph import relations_in_reach

# Takes one weight per edge (1 = fully present, 0 = absent) and returns the image's graph
# score and its gradient with respect to each edge weight.
ScoreAndGradient = Callable[[np.ndarray], tuple[float, np.ndarray]]


def relation_columns(
    edge_index: np.ndarray, relations: list[tuple[int, int]], both_directions: bool
) -> list[np.ndarray]:
    """For each relation, mark the edges that belong to it (both copies if undirected)."""
    return [
        ~mark_remaining_edges(edge_index, [relation], both_directions) for relation in relations
    ]


def integrated_gradients(
    edge_index: np.ndarray,
    image_node: int,
    num_layers: int,
    both_directions: bool,
    score_and_gradient: ScoreAndGradient,
    steps: int,
) -> tuple[list[EdgeContribution], float]:
    """Share the score change from "no nearby relations" to "all relations" among them.

    All nearby relations fade in together from weight 0 to 1; each relation gets the
    average gradient along the way. Returns the contributions sorted from largest to
    smallest, and the completeness gap: their sum minus the actual score change
    (close to 0 when steps is large enough).
    """
    if steps < 1:
        raise ValueError(f"steps must be at least 1, got {steps}")

    relations = relations_in_reach(edge_index, image_node, num_layers, both_directions)
    columns = relation_columns(edge_index, relations, both_directions)

    total_gradient = np.zeros(len(relations))
    for step in range(steps):
        # Midpoint of each step along the straight path from 0 to 1.
        fraction = (step + 0.5) / steps
        weights = np.ones(edge_index.shape[1])
        for relation_edges in columns:
            weights[relation_edges] = fraction
        _, gradient = score_and_gradient(weights)
        for index, relation_edges in enumerate(columns):
            total_gradient[index] += gradient[relation_edges].sum()
    attributions = total_gradient / steps

    full_score, _ = score_and_gradient(np.ones(edge_index.shape[1]))
    baseline = np.ones(edge_index.shape[1])
    for relation_edges in columns:
        baseline[relation_edges] = 0.0
    baseline_score, _ = score_and_gradient(baseline)
    gap = float(attributions.sum() - (full_score - baseline_score))

    contributions = [
        EdgeContribution(source, target, float(value))
        for (source, target), value in zip(relations, attributions, strict=True)
    ]
    contributions.sort(key=lambda edge: edge.contribution, reverse=True)
    return contributions, gap
