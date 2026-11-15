"""The parts of the graph score add up to the model's score and move it the right way."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("faiss")

from mmxai.eval.graph_rerank import SplitGraph  # noqa: E402
from mmxai.explain.ranking_model import graph_scores_with_edges  # noqa: E402
from mmxai.explain.score_path import decompose_graph_score  # noqa: E402
from mmxai.models.sage import ConceptToImageSage  # noqa: E402
from test_explain_ranking_model import (  # noqa: E402
    EDGE_INDEX,
    GRAPH,
    QUERY,
    X,
    identity_model,
    random_model,
)

# One image (row 0) with two concepts that point in opposite directions:
# concept 1 = +e1 agrees with the query, concept 2 = -e1 disagrees with it.
OPPOSITE_X = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
OPPOSITE_EDGES = torch.tensor([[1, 2], [0, 0]])
OPPOSITE_QUERY = np.array([1.0, 1.0], dtype=np.float32)


def opposite_graph(edge_index: torch.Tensor) -> SplitGraph:
    return SplitGraph(np.array([100]), OPPOSITE_X, edge_index, None)


def identity_model_2d() -> ConceptToImageSage:
    """h_i = x_i + mean of its concepts' x_c, in 2 dimensions."""
    model = ConceptToImageSage(2)
    with torch.no_grad():
        model.conv.lin_l.weight.copy_(torch.eye(2))
    return model


def model_score(model, graph: SplitGraph, query: np.ndarray, image_row: int) -> float:
    """The image's graph score from the model run on the whole graph (the ranking's path)."""
    every_edge = np.ones(graph.edge_index.shape[1], dtype=bool)
    return float(graph_scores_with_edges(model, graph, every_edge, query, np.array([image_row]))[0])


@pytest.mark.parametrize("image_row", [0, 1, 2])
def test_parts_add_up_to_the_ranking_score(image_row):
    model = random_model()
    path = decompose_graph_score(model, GRAPH, image_row, QUERY)

    total = path.self_share + path.bias_share + sum(item.share for item in path.concept_shares)
    assert total == pytest.approx(model_score(model, GRAPH, QUERY, image_row), abs=1e-5)
    assert path.graph_score == pytest.approx(total, abs=1e-5)


def test_every_edge_into_the_image_is_on_the_path_and_no_other():
    path = decompose_graph_score(random_model(), GRAPH, 1, QUERY)

    # Image 1 has concepts 4 and 5 (EDGE_INDEX); concept 3 is not its neighbour.
    assert sorted(item.concept_row for item in path.concept_shares) == [4, 5]


def test_shares_by_hand_on_the_identity_model():
    # h_0 = x_0 + (x_3 + x_4) / 2 = e0 + e3/2 + e4/2, ||h_0|| = sqrt(1.5).
    path = decompose_graph_score(identity_model(), GRAPH, 0, QUERY)
    unit_query = QUERY / np.linalg.norm(QUERY)
    norm = np.sqrt(1.5)

    assert path.self_share == pytest.approx(unit_query[0] / norm, abs=1e-6)
    assert path.bias_share == pytest.approx(0.0, abs=1e-6)
    shares = {item.concept_row: item.share for item in path.concept_shares}
    assert shares[3] == pytest.approx(0.5 * unit_query[3] / norm, abs=1e-6)
    assert shares[4] == pytest.approx(0.0, abs=1e-6)
    assert path.concept_shares[0].concept_row == 3  # largest share first


def test_image_without_edges_has_only_its_own_part_and_the_bias():
    model = random_model()
    keep = EDGE_INDEX[1] != 2  # image 2 loses its only edge
    graph = SplitGraph(GRAPH.image_ids, X, EDGE_INDEX[:, keep], None)
    path = decompose_graph_score(model, graph, 2, QUERY)

    assert path.concept_shares == []
    assert path.self_share + path.bias_share == pytest.approx(
        model_score(model, graph, QUERY, 2), abs=1e-5
    )


def test_removing_the_positive_edge_lowers_the_score_and_the_negative_one_raises_it():
    model = identity_model_2d()
    path = decompose_graph_score(model, opposite_graph(OPPOSITE_EDGES), 0, OPPOSITE_QUERY)
    shares = {item.concept_row: item.share for item in path.concept_shares}
    assert shares[1] > 0 > shares[2]

    without_positive = opposite_graph(OPPOSITE_EDGES[:, [1]])
    without_negative = opposite_graph(OPPOSITE_EDGES[:, [0]])
    assert model_score(model, without_positive, OPPOSITE_QUERY, 0) < path.graph_score
    assert model_score(model, without_negative, OPPOSITE_QUERY, 0) > path.graph_score


def test_a_zero_share_is_not_a_zero_removal_effect():
    # Why the demo shows Occlusion next to the shares: concept 4 adds nothing to q^T h_0,
    # but removing it changes the mean (x_3 / 1 instead of / 2) and ||h_0||.
    model = identity_model()
    path = decompose_graph_score(model, GRAPH, 0, QUERY)
    keep = np.ones(EDGE_INDEX.shape[1], dtype=bool)
    keep[1] = False  # the edge 4 -> 0
    after = float(graph_scores_with_edges(model, GRAPH, keep, QUERY, np.array([0]))[0])

    assert {item.concept_row: item.share for item in path.concept_shares}[4] == pytest.approx(0)
    assert after > path.graph_score + 0.1


def test_other_models_are_rejected():
    with pytest.raises(ValueError, match="one-layer"):
        decompose_graph_score(torch.nn.Linear(6, 6), GRAPH, 0, QUERY)
