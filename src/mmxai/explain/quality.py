"""Sparsity and validity: is the explanation small, and is it a real part of the graph?"""

from dataclasses import dataclass

import numpy as np

from mmxai.explain.subgraph import relations_in_reach


def sparsity(num_explained: int, num_in_reach: int) -> float:
    """Share of the nearby relations left out of the explanation (1 = very small).

    Returns 1 - num_explained / num_in_reach.
    """
    if num_in_reach < 1:
        raise ValueError("there are no relations near the image to explain")
    if not 0 <= num_explained <= num_in_reach:
        raise ValueError(f"cannot explain with {num_explained} of {num_in_reach} relations")
    return 1.0 - num_explained / num_in_reach


@dataclass(frozen=True)
class Validity:
    """Whether every relation is real and near the image, and how many link up to it."""

    all_in_reach: bool
    connected_share: float


def validity(
    edge_index: np.ndarray,
    explanation: list[tuple[int, int]],
    image_node: int,
    num_layers: int,
    both_directions: bool,
) -> Validity:
    """Check the explanation against the graph used for scoring.

    all_in_reach: every relation exists in the graph and can change the image's score.
    connected_share: share of relations that link to the image through relations of the
    explanation itself, so the explanation reads as paths ending at the image.
    """
    if not explanation:
        return Validity(all_in_reach=True, connected_share=1.0)

    nearby = relations_in_reach(edge_index, image_node, num_layers, both_directions)
    if both_directions:
        nearby_keys = {frozenset(edge) for edge in nearby}
        all_in_reach = all(frozenset(edge) in nearby_keys for edge in explanation)
    else:
        nearby_keys = set(nearby)
        all_in_reach = all(edge in nearby_keys for edge in explanation)

    # Grow the set of nodes linked to the image, using only explanation relations
    # and ignoring their direction.
    linked = {image_node}
    grew = True
    while grew:
        grew = False
        for source, target in explanation:
            if (source in linked) != (target in linked):
                linked |= {source, target}
                grew = True

    connected = sum(1 for source, target in explanation if source in linked)
    return Validity(all_in_reach=all_in_reach, connected_share=connected / len(explanation))
