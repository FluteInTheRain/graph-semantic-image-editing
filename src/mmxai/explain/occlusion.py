"""Occlusion: explain an image's graph score by removing one edge at a time."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from mmxai.explain.checks import check_score_function
from mmxai.explain.intervention import mark_remaining_edges
from mmxai.explain.subgraph import relations_in_reach


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
    ranked_graph_score: float,
) -> list[EdgeContribution]:
    """Measure how much the image's graph score drops when each nearby relation is removed.

    score_image takes the edges that remain and returns the image's graph score; it is
    checked first against ranked_graph_score, the graph score used for ranking
    (see checks.check_score_function). Returns the relations sorted from the largest drop
    to the smallest.
    """
    check_score_function(edge_index, image_node, num_layers, score_image, ranked_graph_score)
    relations = relations_in_reach(edge_index, image_node, num_layers, both_directions)
    return measure_drops(edge_index, relations, both_directions, score_image)


def measure_drops(
    edge_index: np.ndarray,
    relations: list[tuple[int, int]],
    both_directions: bool,
    score_image: Callable[[np.ndarray], float],
) -> list[EdgeContribution]:
    """Remove each given relation on its own and record how much the score drops.

    Returns the relations sorted from the largest drop to the smallest.
    """
    full_score = score_image(np.ones(edge_index.shape[1], dtype=bool))
    contributions = []
    for source, target in relations:
        remaining = mark_remaining_edges(edge_index, [(source, target)], both_directions)
        drop = full_score - score_image(remaining)
        contributions.append(EdgeContribution(source, target, float(drop)))

    contributions.sort(key=lambda edge: edge.contribution, reverse=True)
    return contributions
