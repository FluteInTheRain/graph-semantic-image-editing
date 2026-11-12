"""One-layer GraphSAGE that gives each image a graph vector from its concepts (ADR-0046).

    h_i = W_self x_i + b + mean_{c in N(i)} W_nb x_c
    graph_score(q, i) = cos(q, h_i)

x_i (image), x_c (concept) and q (query) are frozen CLIP vectors; only W_self, W_nb
and b are learned. Messages go concept -> image only, and there is no activation,
so an image's score depends on its own vector and its own edges, nothing else.

Node layout is the one of mmxai.graph.pyg.to_pyg_data: images first, then concepts.
In PyG's SAGEConv, lin_l is applied to the neighbour mean (it holds the bias b) and
lin_r to the node itself, so W_nb = lin_l and W_self = lin_r.
"""

import torch
from torch import nn
from torch.nn import functional
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

from mmxai.graph.pyg import CONCEPT_NODE, IMAGE_NODE


class ConceptToImageSage(nn.Module):
    """The ADR-0046 model. use_graph=False is the no-graph control (W_nb fixed at 0)."""

    # Explain code needs to know how far messages travel (docs/research/xai-method-proposal.md §6).
    num_layers = 1

    def __init__(self, dim: int, use_graph: bool = True) -> None:
        super().__init__()
        self.use_graph = use_graph
        self.conv = SAGEConv((dim, dim), dim, aggr="mean")
        # Decision (ADR-0046): start from W_self = I, W_nb = 0, b = 0.
        #   Why: the untrained model scores exactly like CLIP, so training only moves
        #   away from the baseline when that lowers the loss.
        #   Decided by: Claude Code proposal in ADR-0046, 2026-11-10; not yet approved by the team.
        with torch.no_grad():
            self.conv.lin_r.weight.copy_(torch.eye(dim))
            self.conv.lin_l.weight.zero_()
            self.conv.lin_l.bias.zero_()
        if not use_graph:
            # Same training, same bias, but no message from concepts can ever arrive:
            # any gain of this control comes from a learned projection, not the graph.
            self.conv.lin_l.weight.requires_grad_(False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return h for every node; only image rows are meaningful.

        edge_index must hold concept -> image edges only (see concept_to_image_edges).
        This is the path used for scoring candidates and by explain/, which re-runs
        it on a graph with some edges removed.
        """
        return self.conv(x, edge_index)

    def forward_from_neighbor_mean(
        self, image_x: torch.Tensor, neighbor_mean: torch.Tensor
    ) -> torch.Tensor:
        """Same h as forward(), from a precomputed mean of each image's concept vectors.

        Used for training only. With one layer, a mean and no activation, the mean of
        the frozen concept vectors never changes, so it is computed once instead of
        sending 1.6M messages of 512 numbers at every step.
        """
        return self.conv.lin_l(neighbor_mean) + self.conv.lin_r(image_x)


def concept_to_image_edges(data: Data) -> torch.Tensor:
    """Keep the concept -> image half of to_pyg_data's edges (it stores both directions)."""
    sources = data.edge_index[0]
    targets = data.edge_index[1]
    keep = (data.node_type[sources] == CONCEPT_NODE) & (data.node_type[targets] == IMAGE_NODE)
    return data.edge_index[:, keep]


def neighbor_mean(x: torch.Tensor, edge_index: torch.Tensor, num_images: int) -> torch.Tensor:
    """Mean of the incoming concept vectors of each image row; 0 for an image without edges.

    A sparse matrix product, so no per-edge copy of the vectors is made. Returns
    zeros for an image with no edge, as PyG's mean aggregation does.
    """
    sources = edge_index[0]
    targets = edge_index[1]
    degree = torch.zeros(num_images, dtype=x.dtype)
    degree.index_add_(0, targets, torch.ones(len(targets), dtype=x.dtype))
    weights = 1.0 / degree[targets]
    average = torch.sparse_coo_tensor(
        torch.stack([targets, sources]),
        weights,
        size=(num_images, x.shape[0]),
        # Built once per run, so the safety check costs nothing noticeable.
        check_invariants=True,
    )
    return torch.sparse.mm(average, x)


def graph_scores(query: torch.Tensor, image_h: torch.Tensor) -> torch.Tensor:
    """cos(q, h) for every (query, image) pair: shape (num_queries, num_images)."""
    return functional.normalize(query, dim=-1) @ functional.normalize(image_h, dim=-1).T
