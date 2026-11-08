"""Remove edges from the graph to measure how much they matter to the score (ADR-0016)."""

from collections.abc import Iterable

import numpy as np


def mark_remaining_edges(
    edge_index: np.ndarray,
    removed_edges: Iterable[tuple[int, int]],
    both_directions: bool,
) -> np.ndarray:
    """Mark which edges remain after removing the given ones.

    Returns True for each kept edge and False for each removed edge.
    Raises ValueError if a given edge does not exist in the graph.
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape (2, num_edges), got {edge_index.shape}")

    sources = edge_index[0]
    targets = edge_index[1]
    remaining = np.ones(edge_index.shape[1], dtype=bool)

    for source, target in removed_edges:
        matches = (sources == source) & (targets == target)
        if not matches.any():
            raise ValueError(f"edge ({source}, {target}) is not in the graph")
        remaining[matches] = False

        if both_directions:
            # One undirected relation is stored as two opposite edges, so remove both.
            reverse = (sources == target) & (targets == source)
            remaining[reverse] = False

    return remaining
