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

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from mmxai.graph.contains import ContainsGraph
from mmxai.models.train import sample_negatives

PAD = -1  # marks an unused slot in the padded concept / subgraph tables


def image_concept_matrix(graph: ContainsGraph) -> np.ndarray:
    """Return (num_images, max_degree): each image's concept rows, padded with PAD."""
    num_images = len(graph.image_ids)
    degree = np.bincount(graph.edge_image, minlength=num_images)
    width = max(int(degree.max()), 1)
    matrix = np.full((num_images, width), PAD, dtype=np.int64)
    filled = np.zeros(num_images, dtype=np.int64)
    # Edges are stored image by image with sorted concepts, so graph row order is kept.
    image_rows = graph.edge_image.tolist()
    concept_rows = graph.edge_concept.tolist()
    for image_row, concept_row in zip(image_rows, concept_rows, strict=True):
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


@dataclass(frozen=True)
class SubgraphTrainingData:
    """Train images, train texts, and each text's PCST subgraph (padded with PAD)."""

    image_x: torch.Tensor  # (num_images, dim) CLIP vectors of train images
    image_concepts: torch.Tensor  # (num_images, D) concept rows per image, padded with PAD
    concept_x: torch.Tensor  # (C, dim) CLIP vectors of concepts
    text_x: torch.Tensor  # (num_texts, dim) CLIP vectors of train texts
    text_image_row: torch.Tensor  # (num_texts,) row of the image each text describes
    candidate_rows: torch.Tensor  # (num_texts, top_k) CLIP's top-k image rows per text
    is_negative: torch.Tensor  # (num_texts, top_k) False for the own image and duplicates
    text_subgraphs: torch.Tensor  # (num_texts, L) concept rows of each text's subgraph


def subgraph_batch_loss(
    model: SubgraphSage,
    data: SubgraphTrainingData,
    text_rows: torch.Tensor,
    num_negatives: int,
    temperature: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """InfoNCE loss of one batch of texts; the correct image is always column 0.

    Kept line by line identical to mmxai.models.train.batch_loss on purpose: only the
    way h is computed differs, so stage 1 and stage 2 can be compared fairly.
    """
    columns, valid = sample_negatives(data.is_negative[text_rows], num_negatives, generator)
    negative_rows = torch.gather(data.candidate_rows[text_rows], 1, columns)
    image_rows = torch.cat([data.text_image_row[text_rows, None], negative_rows], dim=1)

    mean = masked_concept_mean(
        data.concept_x, data.image_concepts[image_rows], data.text_subgraphs[text_rows]
    )
    h = model(data.image_x[image_rows], mean)
    query = functional.normalize(data.text_x[text_rows], dim=-1)
    cosine = (functional.normalize(h, dim=-1) * query[:, None, :]).sum(dim=-1)

    logits = cosine / temperature
    always_valid = torch.ones(len(text_rows), 1, dtype=torch.bool)
    logits = logits.masked_fill(~torch.cat([always_valid, valid], dim=1), float("-inf"))
    targets = torch.zeros(len(text_rows), dtype=torch.long)
    return functional.cross_entropy(logits, targets)
