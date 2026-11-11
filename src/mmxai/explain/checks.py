"""Checks that the function being explained is the one that ranked the images."""

from collections.abc import Callable

import numpy as np

from mmxai.explain.intervention import mark_remaining_edges
from mmxai.explain.subgraph import edges_reaching_node

ScoreImage = Callable[[np.ndarray], float]
ScoreAndGradient = Callable[[np.ndarray], tuple[float, np.ndarray]]


def check_score_function(
    edge_index: np.ndarray,
    image_node: int,
    num_layers: int,
    score_image: ScoreImage,
    ranked_graph_score: float,
) -> float:
    """Check score_image before explaining it, and return the image's full graph score.

    Raises ValueError if score_image gives different results for the same graph (model not
    in eval mode), differs from the graph score used for ranking (wrong model or graph),
    or changes when edges out of reach are removed (num_layers does not match the model).
    """
    all_edges = np.ones(edge_index.shape[1], dtype=bool)
    full_score = score_image(all_edges)
    if not np.isclose(score_image(all_edges), full_score):
        raise ValueError("score_image gives different results for the same graph")
    if not np.isclose(full_score, ranked_graph_score):
        raise ValueError(
            f"score_image gives {full_score}, but the ranking used {ranked_graph_score}"
        )
    reach = edges_reaching_node(edge_index, image_node, num_layers)
    if not np.isclose(score_image(reach), full_score):
        raise ValueError("edges out of reach change the score; check num_layers")
    return full_score


def check_gradient_function(
    edge_index: np.ndarray,
    relations: list[tuple[int, int]],
    both_directions: bool,
    score_image: ScoreImage,
    score_and_gradient: ScoreAndGradient,
) -> None:
    """Check that weight 1 means a kept edge and weight 0 a removed one.

    Gradient explainers fade edges with weights; fidelity removes them. Both must describe
    the same model, so the scores must agree on all edges, no edges, and each of the first
    few relations removed. Raises ValueError otherwise.
    """
    masks = [
        np.ones(edge_index.shape[1], dtype=bool),
        np.zeros(edge_index.shape[1], dtype=bool),
    ]
    for relation in relations[:3]:
        masks.append(mark_remaining_edges(edge_index, [relation], both_directions))

    for mask in masks:
        weighted_score, _ = score_and_gradient(mask.astype(float))
        if not np.isclose(weighted_score, score_image(mask)):
            raise ValueError(
                "score_and_gradient does not match score_image: weight 0 must act like "
                "a removed edge"
            )
