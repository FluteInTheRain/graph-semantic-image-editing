"""Occlusion: explain an image's graph score by removing one edge at a time."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from mmxai.explain.intervention import mark_remaining_edges
from mmxai.explain.subgraph import edges_reaching_node


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

    score_image takes the edges that remain and returns the image's graph score.
    Only edges that can reach the image are tried. Returns the edges sorted from the
    largest drop to the smallest.
    """
    all_edges = np.ones(edge_index.shape[1], dtype=bool)
    full_score = score_image(all_edges)
    reach = edges_reaching_node(edge_index, image_node, num_layers)

    contributions = []
    tried = set()
    for column in np.flatnonzero(reach):
        source = int(edge_index[0, column])
        target = int(edge_index[1, column])
        # With both directions, the two copies of one relation are removed together, so try it once.
        key = frozenset((source, target)) if both_directions else (source, target)
        if key in tried:
            continue
        tried.add(key)

        remaining = mark_remaining_edges(edge_index, [(source, target)], both_directions)
        drop = full_score - score_image(remaining)
        contributions.append(EdgeContribution(source, target, float(drop)))

    contributions.sort(key=lambda edge: edge.contribution, reverse=True)
    return contributions
