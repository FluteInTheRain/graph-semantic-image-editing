"""Find the part of the graph that can influence one node's graph score."""

import numpy as np


def edges_reaching_node(edge_index: np.ndarray, node: int, num_layers: int) -> np.ndarray:
    """Mark the edges that can change the node's score in a model with num_layers layers.

    Messages travel num_layers steps towards the node. One extra step is included because
    some models (e.g. GCN) weight a message by its sender's degree, so an edge into the
    sender also matters. Edges outside this set cannot change the score.
    Returns True for each edge that can change it.
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape (2, num_edges), got {edge_index.shape}")
    if num_layers < 1:
        raise ValueError(f"num_layers must be at least 1, got {num_layers}")

    sources = edge_index[0]
    targets = edge_index[1]

    # Walk backwards from the node over incoming edges, num_layers steps.
    reached = {node}
    frontier = {node}
    for _ in range(num_layers):
        into_frontier = np.isin(targets, list(frontier))
        new_nodes = set(sources[into_frontier].tolist()) - reached
        reached |= new_nodes
        frontier = new_nodes

    return np.isin(targets, list(reached))


def relations_in_reach(
    edge_index: np.ndarray, node: int, num_layers: int, both_directions: bool
) -> list[tuple[int, int]]:
    """List the relations that can change the node's score, each once.

    With both_directions, the two opposite copies of a relation count as one and are
    reported in a fixed direction: the copy pointing at the node if there is one,
    otherwise the smaller pair. The result does not depend on the order of the edges.
    """
    reach = edges_reaching_node(edge_index, node, num_layers)
    existing = set(zip(edge_index[0].tolist(), edge_index[1].tolist(), strict=True))

    relations = []
    seen = set()
    for column in np.flatnonzero(reach):
        source = int(edge_index[0, column])
        target = int(edge_index[1, column])
        if both_directions:
            if frozenset((source, target)) in seen:
                continue
            seen.add(frozenset((source, target)))
            source, target = _reported_direction(source, target, node, existing)
        relations.append((source, target))
    return relations


def _reported_direction(
    source: int, target: int, node: int, existing: set[tuple[int, int]]
) -> tuple[int, int]:
    copies = [pair for pair in ((source, target), (target, source)) if pair in existing]
    for pair in copies:
        if pair[1] == node:
            return pair
    return min(copies)
