"""The one-layer GraphSAGE of ADR-0046."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from mmxai.models.one_layer_sage import OneLayerSage, graph_scores  # noqa: E402

CONCEPT_X = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
IMAGE_X = torch.tensor([[0.6, 0.8], [1.0, 0.0]])
# concept 0 -> image 0, concept 1 -> image 0, concept 1 -> image 1
EDGES = torch.tensor([[0, 1, 1], [0, 0, 1]])


def test_untrained_model_returns_the_clip_vector():
    """W_self = I, W_nb = 0, b = 0: graph score starts equal to the CLIP cosine."""
    model = OneLayerSage(2)
    assert torch.allclose(model(CONCEPT_X, IMAGE_X, EDGES), IMAGE_X)


def test_neighbours_are_averaged_through_w_nb():
    model = OneLayerSage(2)
    with torch.no_grad():
        model.conv.lin_l.weight.copy_(torch.eye(2))
    h = model(CONCEPT_X, IMAGE_X, EDGES)
    assert torch.allclose(h[0], IMAGE_X[0] + (CONCEPT_X[0] + CONCEPT_X[1]) / 2)
    assert torch.allclose(h[1], IMAGE_X[1] + CONCEPT_X[1])


def test_removing_an_edge_changes_only_its_image():
    model = OneLayerSage(2)
    with torch.no_grad():
        model.conv.lin_l.weight.copy_(torch.eye(2))
    full = model(CONCEPT_X, IMAGE_X, EDGES)
    without_last = model(CONCEPT_X, IMAGE_X, EDGES[:, :2])
    assert torch.allclose(full[0], without_last[0])
    assert not torch.allclose(full[1], without_last[1])


def test_same_graph_gives_same_scores_in_eval_mode():
    model = OneLayerSage(2).eval()
    query = torch.tensor([0.0, 1.0])
    first = graph_scores(query, model(CONCEPT_X, IMAGE_X, EDGES))
    second = graph_scores(query, model(CONCEPT_X, IMAGE_X, EDGES))
    assert torch.equal(first, second)
