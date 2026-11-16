import numpy as np
import torch

from mmxai.graph.contains import ContainsGraph
from mmxai.models.subgraph_sage import (
    SubgraphSage,
    image_concept_matrix,
    masked_concept_mean,
)

CONCEPT_X = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])


def test_image_concept_matrix_pads_with_minus_one():
    graph = ContainsGraph(
        image_ids=np.array([10, 11]),
        image_split=np.array(["train", "train"]),
        concept_keys=np.array(["a.n.01", "b.n.01", "c.n.01"]),
        edge_image=np.array([0, 0, 1]),
        edge_concept=np.array([0, 2, 1]),
    )
    assert image_concept_matrix(graph).tolist() == [[0, 2], [1, -1]]


def test_masked_mean_keeps_only_concepts_in_the_subgraph():
    image_concepts = torch.tensor([[[0, 2], [1, -1]]])
    subgraph = torch.tensor([[2, 1]])
    mean = masked_concept_mean(CONCEPT_X, image_concepts, subgraph)
    assert torch.allclose(mean[0, 0], CONCEPT_X[2])
    assert torch.allclose(mean[0, 1], CONCEPT_X[1])


def test_no_concept_in_the_subgraph_gives_zero():
    mean = masked_concept_mean(CONCEPT_X, torch.tensor([[[0, -1]]]), torch.tensor([[1, -1]]))
    assert torch.equal(mean, torch.zeros(1, 1, 2))


def test_padding_in_both_never_matches():
    mean = masked_concept_mean(CONCEPT_X, torch.tensor([[[-1, -1]]]), torch.tensor([[-1]]))
    assert torch.equal(mean, torch.zeros(1, 1, 2))


def test_untrained_model_returns_the_clip_vector():
    image_x = torch.tensor([[[0.3, 0.4]]])
    output = SubgraphSage(2)(image_x, torch.tensor([[[5.0, 5.0]]]))
    assert torch.allclose(output, image_x)
