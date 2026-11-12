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
    CooccurrenceSage,
    concept_to_image_edges,
    graph_scores,
    neighbor_mean,
    second_hop_mean,
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


# CooccurrenceSage (ADR-0048) on the same graph, plus concept -> concept edges in
# concept rows (0 = "dog" = node 3, 1 = "sofa" = node 4): dog hears sofa.
CONCEPT_EDGE_INDEX = torch.tensor([[1], [0]])


def trained_like_cooccurrence_model() -> CooccurrenceSage:
    torch.manual_seed(0)
    model = CooccurrenceSage(DIM)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_()
    return model.eval()


def test_untrained_cooccurrence_model_scores_exactly_like_clip():
    x = make_x()

    h = CooccurrenceSage(DIM)(x, EDGE_INDEX, CONCEPT_EDGE_INDEX, NUM_IMAGES)

    assert torch.allclose(h[:NUM_IMAGES], x[:NUM_IMAGES])


def test_cooccurrence_shortcut_gives_the_same_vectors_as_sageconv():
    x = make_x()
    model = trained_like_cooccurrence_model()

    full = model(x, EDGE_INDEX, CONCEPT_EDGE_INDEX, NUM_IMAGES)[:NUM_IMAGES]
    shortcut = model.forward_from_means(
        x[:NUM_IMAGES],
        neighbor_mean(x, EDGE_INDEX, NUM_IMAGES),
        second_hop_mean(x, EDGE_INDEX, CONCEPT_EDGE_INDEX, NUM_IMAGES),
    )

    assert torch.allclose(full, shortcut, atol=1e-5)


def test_second_hop_mean_averages_the_partners_of_each_images_concepts():
    x = make_x()

    second = second_hop_mean(x, EDGE_INDEX, CONCEPT_EDGE_INDEX, NUM_IMAGES)

    # Image A has dog (hears sofa) and sofa (hears nobody, 0): mean = x_sofa / 2.
    assert torch.allclose(second[0], x[4] / 2)
    assert torch.allclose(second[1], torch.zeros(DIM))  # B has sofa only
    assert torch.allclose(second[2], torch.zeros(DIM))  # C has no concept


def test_removing_a_concept_edge_changes_only_images_with_that_concept():
    # Only image A has dog, so removing sofa -> dog changes A and nothing else.
    x = make_x()
    model = trained_like_cooccurrence_model()
    no_concept_edges = CONCEPT_EDGE_INDEX[:, :0]

    before = model(x, EDGE_INDEX, CONCEPT_EDGE_INDEX, NUM_IMAGES)
    after = model(x, EDGE_INDEX, no_concept_edges, NUM_IMAGES)

    assert not torch.allclose(before[0], after[0])
    assert torch.allclose(before[1], after[1])
    assert torch.allclose(before[2], after[2])
