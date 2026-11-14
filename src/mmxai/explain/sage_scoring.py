"""The graph score of one image under the trained ConceptToImageSage, for the explainers.

The level-3 explainers (Occlusion, Integrated Gradients, GNNExplainer) and fidelity call a
score function many times per image. Re-running the model on the whole split graph each
time would cost seconds per call, so this module re-runs it on the image's own edges only.

Why that is the same score as the ranking (ADR-0046 model, one layer, concept -> image):
    h_i = W_self x_i + b + mean_{c in N(i)} W_nb x_c,   graph_score(q, i) = cos(q, h_i)
h_i reads only the image's own vector and its own incoming edges, so the image's score
cannot change when any other edge is removed. tests/test_explain_sage_scoring.py checks
that both functions below agree with ranking_model.graph_scores_with_edges (full graph).

score_and_gradient (for IG and GNNExplainer) needs a weight per edge. A mean has no
weights, so the team chose (docs/research/xai-method-proposal.md section 7c):
    mean_w = sum_c w_c x_c / max(sum_c w_c, 1)
With weights 0 or 1 this is exactly the model's mean over the kept edges (0 when none is
kept, as in PyG), which checks.check_gradient_function verifies before every explanation.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional

from mmxai.eval.graph_rerank import SplitGraph
from mmxai.models.sage import ConceptToImageSage


@dataclass(frozen=True)
class ImageScorer:
    """Score functions of one image for one query, over the image's own edges.

    edge_index: the image's concept -> image edges, in the split graph's node rows.
    score_image(remaining): graph score with only the edges marked True.
    score_and_gradient(weights): graph score with edge weights in [0, 1] and its
    gradient per edge (1 = kept, 0 = removed).
    """

    image_row: int
    edge_index: np.ndarray
    score_image: Callable[[np.ndarray], float]
    score_and_gradient: Callable[[np.ndarray], tuple[float, np.ndarray]]


def image_scorer(
    model: ConceptToImageSage, split_graph: SplitGraph, image_row: int, query: np.ndarray
) -> ImageScorer:
    """Build the score functions of image_row for this query on the trained model."""
    if not isinstance(model, ConceptToImageSage):
        raise ValueError("only the one-layer ConceptToImageSage model (ADR-0046) is supported")
    model.eval()

    into_image = split_graph.edge_index[1] == image_row
    edge_index = split_graph.edge_index[:, into_image]
    concepts = edge_index[0]
    # A tiny bipartite graph: the image is node 0, its concepts are nodes 0..d-1 on the
    # source side. SAGEConv then runs exactly as on the split graph, for this image only.
    source_x = split_graph.x[concepts]
    image_x = split_graph.x[image_row : image_row + 1]
    unit_query = functional.normalize(torch.from_numpy(query).to(image_x.dtype), dim=0)

    def score_image(remaining: np.ndarray) -> float:
        kept = torch.from_numpy(np.flatnonzero(remaining))
        local_edges = torch.stack([kept, torch.zeros_like(kept)])
        with torch.no_grad():
            h = model.conv((source_x, image_x), local_edges)[0]
        return float(functional.normalize(h, dim=0) @ unit_query)

    def score_and_gradient(weights: np.ndarray) -> tuple[float, np.ndarray]:
        w = torch.tensor(weights, dtype=image_x.dtype, requires_grad=True)
        # Decision (section 7c of the xAI proposal): divide by max(sum w, 1), not sum w.
        #   Why: equals the model's mean for 0/1 weights, and stays finite near w = 0,
        #   where sum w x / sum w would not change with w at all and IG would break.
        weighted_mean = (w[:, None] * source_x).sum(dim=0) / torch.clamp(w.sum(), min=1.0)
        h = model.conv.lin_l(weighted_mean) + model.conv.lin_r(image_x[0])
        score = functional.normalize(h, dim=0) @ unit_query
        score.backward()
        return float(score.detach()), w.grad.numpy().astype(float)

    return ImageScorer(image_row, edge_index.numpy(), score_image, score_and_gradient)


def candidate_scores(
    scorer: ImageScorer, ranked_graph_scores: np.ndarray, column: int
) -> Callable[[np.ndarray], np.ndarray]:
    """Graph scores of every candidate of the query when only the image's edges change.

    The other candidates keep the scores the ranking used: their h does not read this
    image's edges (module docstring). fidelity.evaluate_explanation re-ranks with these.
    """

    def graph_scores(remaining: np.ndarray) -> np.ndarray:
        scores = np.array(ranked_graph_scores, dtype=float)
        scores[column] = scorer.score_image(remaining)
        return scores

    return graph_scores
