"""GraphSAGE models that give each image a graph vector from its concepts.

ConceptToImageSage is the one-layer model of ADR-0046 (below). CooccurrenceSage
(ADR-0048, end of this file) puts a concept -> concept layer in front of it.

ADR-0046 model:

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


class CooccurrenceSage(nn.Module):
    """The ADR-0048 model: concepts first hear the concepts they co-occur with.

        layer 1 (concept -> concept): z_c = W_cself x_c + mean_{c' in co(c)} W_co x_c'
        layer 2 (concept -> image):   h_i = W_self x_i + b + mean_{c in N(i)} W_nb z_c

    Layer 2 is the ADR-0046 model, fed z_c instead of x_c. Co-occurrence edges come
    from train annotations (ADR-0048), so an image's score still depends only on its
    own vector, its own contains edges and one fixed concept graph: val/test images
    never influence each other or train images.
    """

    num_layers = 2

    def __init__(self, dim: int) -> None:
        super().__init__()
        # No bias in layer 1: one bias (in layer 2) is enough for a linear model, and it
        # keeps the training shortcut below exact for images without edges.
        self.concept_conv = SAGEConv(dim, dim, aggr="mean", bias=False)
        self.conv = SAGEConv((dim, dim), dim, aggr="mean")
        # Decision (ADR-0048): start from W_cself = I, W_co = 0, plus the ADR-0046 start.
        #   Why: z_c = x_c, so the untrained model is the untrained ADR-0046 model, which
        #   scores exactly like CLIP. Proposed by Claude Code, 2026-11-12; not yet approved.
        with torch.no_grad():
            self.concept_conv.lin_r.weight.copy_(torch.eye(dim))
            self.concept_conv.lin_l.weight.zero_()
            self.conv.lin_r.weight.copy_(torch.eye(dim))
            self.conv.lin_l.weight.zero_()
            self.conv.lin_l.bias.zero_()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        concept_edge_index: torch.Tensor,
        num_images: int,
    ) -> torch.Tensor:
        """Return h for every node; only image rows are meaningful.

        x: images first, then concepts (to_pyg_data layout). edge_index: concept ->
        image edges in node rows. concept_edge_index: concept -> concept edges in
        CONCEPT rows (0 = first concept). explain/ can remove edges of either kind.
        """
        concept_z = self.concept_conv(x[num_images:], concept_edge_index)
        # Image rows of z are never a message source (edge_index is concept -> image),
        # so they are only placeholders that keep node rows unchanged.
        z = torch.cat([x[:num_images], concept_z])
        # Bipartite call: messages read z, the image's own term reads its CLIP vector x.
        return self.conv((z, x), edge_index)

    def forward_from_means(
        self, image_x: torch.Tensor, neighbor_mean: torch.Tensor, second_hop_mean: torch.Tensor
    ) -> torch.Tensor:
        """Same h as forward(), from two precomputed means; used for training only.

        neighbor_mean[i]   = mean of x_c over the image's concepts c
        second_hop_mean[i] = mean over those c of (mean of x_c' over c's co-occurring c')
        Everything is linear, so mean_c z_c = W_cself neighbor_mean + W_co second_hop_mean.
        """
        concept_mean_z = self.concept_conv.lin_r(neighbor_mean) + self.concept_conv.lin_l(
            second_hop_mean
        )
        return self.conv.lin_l(concept_mean_z) + self.conv.lin_r(image_x)


def second_hop_mean(
    x: torch.Tensor, edge_index: torch.Tensor, concept_edge_index: torch.Tensor, num_images: int
) -> torch.Tensor:
    """For each image: mean over its concepts c of the mean of c's co-occurring concepts.

    0 for a concept without co-occurrence edges, and 0 for an image without
    contains edges, exactly as PyG's mean aggregation gives.
    """
    concept_x = x[num_images:]
    concept_partner_mean = neighbor_mean(concept_x, concept_edge_index, len(concept_x))
    # Swap each concept's own vector for its partners' mean, then average per image.
    return neighbor_mean(torch.cat([x[:num_images], concept_partner_mean]), edge_index, num_images)
