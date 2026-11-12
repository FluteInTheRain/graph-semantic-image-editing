"""The one-layer concept -> image GraphSAGE (ADR-0046) on a tiny made-up graph.

Nodes (images first, then concepts, as in to_pyg_data):
    0 image A  <- concepts 3, 4
    1 image B  <- concept 4
    2 image C  (no edge)
    3 concept "dog"
    4 concept "sofa"
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from torch_geometric.data import Data  # noqa: E402

from mmxai.graph.pyg import CONCEPT_NODE, IMAGE_NODE  # noqa: E402
from mmxai.models.sage import (  # noqa: E402
    ConceptToImageSage,
    concept_to_image_edges,
    graph_scores,
    neighbor_mean,
)

NUM_IMAGES = 3
DIM = 4
# Concept -> image edges only.
EDGE_INDEX = torch.tensor([[3, 4, 4], [0, 0, 1]])


def make_x() -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.randn(5, DIM, generator=generator)


def trained_like_model(use_graph: bool = True) -> ConceptToImageSage:
    """A model with non-trivial weights, as after some training."""
    torch.manual_seed(0)
    model = ConceptToImageSage(DIM, use_graph=use_graph)
    with torch.no_grad():
        model.conv.lin_l.weight.normal_()
        model.conv.lin_l.bias.normal_()
        model.conv.lin_r.weight.normal_()
    return model.eval()


def test_untrained_model_scores_exactly_like_clip():
    # W_self = I, W_nb = 0, b = 0: the graph vector of an image is its CLIP vector.
    x = make_x()
    model = ConceptToImageSage(DIM)

    h = model(x, EDGE_INDEX)

    assert torch.allclose(h[:NUM_IMAGES], x[:NUM_IMAGES])


def test_scores_are_repeatable_in_eval_mode():
    # explain/ (occlusion) refuses a model that gives two scores for the same graph.
    x = make_x()
    model = trained_like_model()

    assert torch.equal(model(x, EDGE_INDEX), model(x, EDGE_INDEX))


def test_removing_an_edge_changes_only_its_image():
    x = make_x()
    model = trained_like_model()
    without_sofa_to_b = EDGE_INDEX[:, :2]  # drop edge 4 -> 1

    before = model(x, EDGE_INDEX)
    after = model(x, without_sofa_to_b)

    assert not torch.allclose(before[1], after[1])
    assert torch.allclose(before[0], after[0])
    assert torch.allclose(before[2], after[2])


def test_training_shortcut_gives_the_same_vectors_as_sageconv():
    x = make_x()
    model = trained_like_model()

    full = model(x, EDGE_INDEX)[:NUM_IMAGES]
    mean = neighbor_mean(x, EDGE_INDEX, NUM_IMAGES)
    shortcut = model.forward_from_neighbor_mean(x[:NUM_IMAGES], mean)

    assert torch.allclose(full, shortcut, atol=1e-6)


def test_neighbor_mean_is_the_average_of_incoming_concepts():
    x = make_x()

    mean = neighbor_mean(x, EDGE_INDEX, NUM_IMAGES)

    assert torch.allclose(mean[0], (x[3] + x[4]) / 2)
    assert torch.allclose(mean[1], x[4])
    assert torch.equal(mean[2], torch.zeros(DIM))  # no edge, as PyG's mean aggregation


def test_no_graph_control_ignores_the_edges():
    x = make_x()
    model = ConceptToImageSage(DIM, use_graph=False)

    with_edges = model(x, EDGE_INDEX)
    without_edges = model(x, torch.empty(2, 0, dtype=torch.long))

    assert torch.allclose(with_edges[:NUM_IMAGES], without_edges[:NUM_IMAGES])


def test_no_graph_control_cannot_learn_the_neighbor_weight():
    model = ConceptToImageSage(DIM, use_graph=False)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]

    # `is`, not `in`: `in` would compare tensors element by element.
    assert not any(parameter is model.conv.lin_l.weight for parameter in trainable)
    assert any(parameter is model.conv.lin_r.weight for parameter in trainable)


def test_concept_to_image_edges_drops_the_reverse_direction():
    node_type = torch.tensor([IMAGE_NODE, CONCEPT_NODE])
    data = Data(edge_index=torch.tensor([[1, 0], [0, 1]]), node_type=node_type, num_nodes=2)

    edges = concept_to_image_edges(data)

    assert edges.tolist() == [[1], [0]]


def test_graph_scores_are_cosines():
    query = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    image_h = torch.tensor([[3.0, 0.0], [1.0, 1.0]])

    scores = graph_scores(query, image_h)

    expected = torch.tensor([[1.0, 2**-0.5], [0.0, 2**-0.5]])
    assert torch.allclose(scores, expected)
