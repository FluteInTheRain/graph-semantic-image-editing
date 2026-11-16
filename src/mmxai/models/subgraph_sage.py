"""Stage 2 of ADR-0051: GraphSAGE restricted to the query's PCST subgraph.

For a query q, S_q is the set of concepts its subgraph keeps. For image i:

    m_i(q) = mean of x_c over the concepts c in N(i) that are also in S_q
             (the zero vector when there is none)
    h_i(q) = W_self x_i + W_nb m_i(q) + b
    score  = cos(q, h_i(q))

h depends on the query because the neighbour mean only sees concepts inside S_q:
two queries with different subgraphs give the same image different vectors.
Initialisation is W_self = I, W_nb = 0, b = 0, so the untrained model returns x_i
and the score equals plain CLIP.
"""

import numpy as np
import torch
from torch import nn

from mmxai.graph.contains import ContainsGraph

PAD = -1  # marks an unused slot in the padded concept / subgraph tables


def image_concept_matrix(graph: ContainsGraph) -> np.ndarray:
    """Return (num_images, max_degree): each image's concept rows, padded with PAD."""
    num_images = len(graph.image_ids)
    degree = np.bincount(graph.edge_image, minlength=num_images)
    width = max(int(degree.max()), 1)
    matrix = np.full((num_images, width), PAD, dtype=np.int64)
    filled = np.zeros(num_images, dtype=np.int64)
    # Edges are stored image by image with sorted concepts, so graph row order is kept.
    for image_row, concept_row in zip(graph.edge_image.tolist(), graph.edge_concept.tolist()):
        matrix[image_row, filled[image_row]] = concept_row
        filled[image_row] += 1
    return matrix


def masked_concept_mean(
    concept_x: torch.Tensor, image_concepts: torch.Tensor, subgraph: torch.Tensor
) -> torch.Tensor:
    """Mean CLIP vector of each image's concepts that lie in the query subgraph.

    concept_x (C, dim); image_concepts (Q, M, D) concept rows per image, padded with PAD;
    subgraph (Q, L) concept rows of each query's subgraph, padded with PAD.
    Returns (Q, M, dim); an image with no concept in the subgraph gets the zero vector.
    """
    in_subgraph = (image_concepts[..., None] == subgraph[:, None, None, :]).any(dim=-1)
    # PAD equals PAD in both tensors, so padding would match itself; the >= 0 test stops that.
    keep = (image_concepts >= 0) & in_subgraph
    vectors = concept_x[image_concepts.clamp(min=0)]  # (Q, M, D, dim)
    total = (vectors * keep[..., None]).sum(dim=2)
    count = keep.sum(dim=2, keepdim=True)
    return total / count.clamp(min=1)


class SubgraphSage(nn.Module):
    """h = W_self x + W_nb m + b, starting exactly at h = x (plain CLIP)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.self_weight = nn.Linear(dim, dim, bias=False)
        self.neighbor_weight = nn.Linear(dim, dim, bias=True)
        with torch.no_grad():
            self.self_weight.weight.copy_(torch.eye(dim))
            self.neighbor_weight.weight.zero_()
            self.neighbor_weight.bias.zero_()

    def forward(self, image_x: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
        return self.self_weight(image_x) + self.neighbor_weight(mean)
