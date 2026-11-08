"""Find the part of the graph that can influence one node's graph score."""

import numpy as np


def edges_reaching_node(edge_index: np.ndarray, node: int, num_layers: int) -> np.ndarray:
    """Mark the edges whose messages can reach the node within num_layers rounds.

    A graph model with num_layers layers passes messages num_layers times, so any edge
    outside this set cannot change the node's score and never needs explaining.
    Returns True for each edge that can reach the node.
    """
    if num_layers < 1:
        raise ValueError(f"num_layers must be at least 1, got {num_layers}")

    sources = edge_index[0]
    targets = edge_index[1]

    # Walk backwards from the node: a message into a reached node arrives in time only if
    # it still has rounds left, so nodes found in the last round add no new edges.
    reached = {node}
    frontier = {node}
    for _ in range(num_layers - 1):
        into_frontier = np.isin(targets, list(frontier))
        new_nodes = set(sources[into_frontier].tolist()) - reached
        reached |= new_nodes
        frontier = new_nodes

    return np.isin(targets, list(reached))
