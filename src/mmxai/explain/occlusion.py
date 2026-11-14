"""Occlusion: explain an image's graph score by removing one edge at a time."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from mmxai.explain.intervention import mark_remaining_edges
from mmxai.explain.subgraph import edges_reaching_node, relations_in_reach


@dataclass(frozen=True)
class EdgeContribution:
    """How much one edge adds to an image's graph score."""

    source: int
    target: int
    contribution: float


def occlusion(
    edge_index: np.ndarray,
    image_node: int,
    num_layers: int,
    both_directions: bool,
    score_image: Callable[[np.ndarray], float],
) -> list[EdgeContribution]:
    """Measure how much the image's graph score drops when each nearby edge is removed.

    score_image takes the edges that remain and returns the image's graph score. It must be
    the same score used for ranking, on the same graph, and give the same result every time
    (model in eval mode). Returns the edges sorted from the largest drop to the smallest.
    Raises ValueError if score_image is not repeatable or depends on edges out of reach.
    """
    all_edges = np.ones(edge_index.shape[1], dtype=bool)
    full_score = score_image(all_edges)
    if not np.isclose(score_image(all_edges), full_score):
        raise ValueError("score_image gives different results for the same graph")

    reach = edges_reaching_node(edge_index, image_node, num_layers)
    # If removing every edge out of reach changes the score, num_layers is wrong.
    if not np.isclose(score_image(reach), full_score):
        raise ValueError("edges out of reach change the score; check num_layers")

    contributions = []
    for source, target in relations_in_reach(edge_index, image_node, num_layers, both_directions):
        remaining = mark_remaining_edges(edge_index, [(source, target)], both_directions)
        drop = full_score - score_image(remaining)
        contributions.append(EdgeContribution(source, target, float(drop)))

    contributions.sort(key=lambda edge: edge.contribution, reverse=True)
    return contributions
