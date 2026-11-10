"""One-layer GraphSAGE that turns a CLIP image vector into a graph-aware one (ADR-0046).

    h_i = W_self x_i + b + mean_{c in N(i)} W_nb x_c
    graph_score(q, i) = cos(q, h_i)

Messages only go concept -> image, and there is one layer, so an image's h_i depends
on its own vector and its own concept edges only. Val/test images therefore never
change each other's scores or the score of a train image.
"""

import torch
from torch import nn
from torch_geometric.nn import SAGEConv


class OneLayerSage(nn.Module):
    """SAGEConv(aggr="mean") on the bipartite concept -> image graph, no activation."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        # In PyG's SAGEConv, lin_l (with the bias) acts on the neighbour mean and
        # lin_r on the node itself: lin_l = W_nb (+ b), lin_r = W_self.
        self.conv = SAGEConv((dim, dim), dim, aggr="mean")
        with torch.no_grad():
            # Decision (ADR-0046): start from W_self = I, W_nb = 0, b = 0.
            #   Why: graph_score then equals the CLIP cosine, so training starts from
            #   the baseline and only moves away from it when that lowers the loss.
            self.conv.lin_r.weight.copy_(torch.eye(dim))
            self.conv.lin_l.weight.zero_()
            self.conv.lin_l.bias.zero_()

    def forward(
        self, concept_x: torch.Tensor, image_x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Return h for every row of image_x.

        edge_index is (2, E): row 0 = concept row in concept_x, row 1 = image row in
        image_x. An image with no edge gets W_self x_i + b (the mean of nothing is 0).
        """
        return self.conv((concept_x, image_x), edge_index)


def graph_scores(query: torch.Tensor, image_h: torch.Tensor) -> torch.Tensor:
    """cos(query, h) for one query (d,) against images (n, d), or row by row for (n, d)."""
    return nn.functional.cosine_similarity(query, image_h, dim=-1)
