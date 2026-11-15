"""Split one image's graph score into the paths query -> concept -> image (ADR-0016).

The ADR-0046 model is linear in its inputs, so h_i is a sum of parts:
    h_i = W_self x_i  +  b  +  sum_{c in N(i)} W_nb x_c / |N(i)|
and the graph score divides every part by the same number ||h_i||:
    g(q, i) = q^T h_i / ||h_i||          (q of length 1)
            = self_share + bias_share + sum_c concept_share[c]
    self_share       = q^T W_self x_i / ||h_i||               (the image on its own)
    bias_share       = q^T b / ||h_i||
    concept_share[c] = q^T W_nb x_c / (|N(i)| ||h_i||)        (the edge c -> image)

Each part is a term of the very sum the ranking computed, so the path
"query -> concept c -[contains]-> image i" carries a weight that is part of the score;
no path is searched for. decompose_graph_score checks that the parts add up to the
model's score before returning them.

A share is not what removing the edge would change: removing c also changes |N(i)| and
||h_i||, so an edge with share 0 can still move the score. The removal effect is
Occlusion's number (mmxai.explain.occlusion); the two are shown side by side.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional

from mmxai.eval.graph_rerank import SplitGraph
from mmxai.explain.ranking_model import SCORE_TOLERANCE
from mmxai.explain.sage_scoring import image_scorer
from mmxai.models.sage import ConceptToImageSage


@dataclass(frozen=True)
class ConceptShare:
    """The part of the graph score that reaches the image through one contains edge."""

    concept_row: int  # node row of the concept (source of the edge)
    share: float


@dataclass(frozen=True)
class ScorePath:
    """An image's graph score as the sum self_share + bias_share + every concept share."""

    image_row: int
    graph_score: float  # cos(q, h_i), as the ranking computed it
    self_share: float
    bias_share: float
    concept_shares: list[ConceptShare]  # every edge into the image, largest share first


def decompose_graph_score(
    model: ConceptToImageSage, split_graph: SplitGraph, image_row: int, query: np.ndarray
) -> ScorePath:
    """Split the graph score of image_row for this query into its parts (module docstring).

    Raises ValueError if the parts do not add up to the model's own score.
    """
    if not isinstance(model, ConceptToImageSage):
        raise ValueError("only the one-layer ConceptToImageSage model (ADR-0046) is supported")
    model.eval()

    into_image = split_graph.edge_index[1] == image_row
    concept_rows = split_graph.edge_index[0, into_image]
    image_x = split_graph.x[image_row]
    concept_x = split_graph.x[concept_rows]
    unit_query = functional.normalize(torch.from_numpy(query).to(image_x.dtype), dim=0)

    # In PyG's SAGEConv, lin_r (W_self) has no bias and lin_l (W_nb) holds b (models/sage.py).
    with torch.no_grad():
        self_part = model.conv.lin_r(image_x)
        bias_part = model.conv.lin_l.bias
        # One row per edge: W_nb x_c / |N(i)|. No edge: no row, and h_i = W_self x_i + b,
        # as PyG's mean gives 0 for an image without neighbours.
        concept_parts = concept_x @ model.conv.lin_l.weight.T / max(len(concept_rows), 1)
        h = self_part + bias_part + concept_parts.sum(dim=0)
        norm = torch.linalg.vector_norm(h)
        self_share = float(unit_query @ self_part / norm)
        bias_share = float(unit_query @ bias_part / norm)
        shares = (concept_parts @ unit_query / norm).tolist()

    concept_shares = [
        ConceptShare(int(row), float(share))
        for row, share in zip(concept_rows.tolist(), shares, strict=True)
    ]
    concept_shares.sort(key=lambda item: item.share, reverse=True)

    # The faithfulness guard: the parts must be the score the ranking used, which
    # image_scorer computes with the model's own SAGEConv forward pass.
    every_edge = np.ones(len(concept_rows), dtype=bool)
    graph_score = image_scorer(model, split_graph, image_row, query).score_image(every_edge)
    total = self_share + bias_share + sum(item.share for item in concept_shares)
    if abs(total - graph_score) > SCORE_TOLERANCE:
        raise ValueError(
            f"the parts add up to {total}, but the model scores image row {image_row} "
            f"at {graph_score}"
        )
    return ScorePath(image_row, graph_score, self_share, bias_share, concept_shares)
