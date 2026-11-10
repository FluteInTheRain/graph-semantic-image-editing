"""Small stand-in graph models for explain tests (no torch needed)."""

import numpy as np


def toy_sum_model(edge_index: np.ndarray, features: np.ndarray, num_layers: int) -> np.ndarray:
    """Each round, a node adds the values sent along its incoming edges."""
    state = features.copy()
    for _ in range(num_layers):
        new_state = state.copy()
        for source, target in edge_index.T:
            new_state[target] += state[source]
        state = new_state
    return state


def toy_gcn(edge_index: np.ndarray, features: np.ndarray, num_layers: int) -> np.ndarray:
    """GCN-style rounds: self-loops, messages scaled by 1/sqrt(deg(sender) * deg(receiver)),
    then a ReLU with a bias so the model is not linear."""
    num_nodes = len(features)
    # Degree counts incoming edges plus the self-loop, as in PyG's GCN normalisation.
    degree = np.ones(num_nodes)
    for target in edge_index[1]:
        degree[target] += 1

    state = features.copy()
    for _ in range(num_layers):
        new_state = state / degree
        for source, target in edge_index.T:
            new_state[target] += state[source] / np.sqrt(degree[source] * degree[target])
        state = np.maximum(new_state - 0.1, 0.0)
    return state
