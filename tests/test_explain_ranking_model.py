"""Occlusion on the real one-layer model (ConceptToImageSage) and a tiny hand-built graph."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("faiss")

from mmxai.eval.graph_rerank import SplitGraph  # noqa: E402
from mmxai.explain.ranking_model import explain_candidate, graph_scores_with_edges  # noqa: E402
from mmxai.models.sage import ConceptToImageSage, graph_scores  # noqa: E402

# Node rows: images 0, 1, 2, then concepts 3, 4, 5. Every vector is a one-hot direction.
X = torch.eye(6)
# Concept -> image edges: image 0 has concepts 3 and 4, image 1 has 4 and 5, image 2 has 5.
EDGE_INDEX = torch.tensor([[3, 4, 4, 5, 5], [0, 0, 1, 1, 2]])
GRAPH = SplitGraph(
    image_ids=np.array([100, 101, 102]),
    x=X,
    edge_index=EDGE_INDEX,
    concept_edge_index=None,
)
# The query points mostly at concept 3, a little at image 0.
QUERY = np.array([0.1, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
CANDIDATE_ROWS = np.array([1, 2, 0])  # CLIP's order: image 101, 102, then 100
CLIP_SCORES = np.array([0.9, 0.8, 0.7])
EVERY_EDGE = np.ones(5, dtype=bool)


def identity_model() -> ConceptToImageSage:
    """h_i = x_i + mean of its concepts' x_c: easy to check by hand."""
    model = ConceptToImageSage(6)
    with torch.no_grad():
        model.conv.lin_l.weight.copy_(torch.eye(6))
    return model


def random_model() -> ConceptToImageSage:
    torch.manual_seed(0)
    model = ConceptToImageSage(6)
    with torch.no_grad():
        model.conv.lin_l.weight.normal_()
        model.conv.lin_l.bias.normal_()
        model.conv.lin_r.weight.normal_()
    return model


def direct_score(model, edge_index, image_row) -> float:
    """cos(q, h_i) straight from the model, without any explain code."""
    model.eval()
    with torch.no_grad():
        h = model(X, edge_index)
    return float(graph_scores(torch.from_numpy(QUERY)[None], h[image_row][None])[0, 0])


def ranking_scores(model) -> np.ndarray:
    return np.array([direct_score(model, EDGE_INDEX, row) for row in CANDIDATE_ROWS])


def explain(model, num_edges=1, column=2):
    return explain_candidate(
        model,
        GRAPH,
        QUERY,
        CANDIDATE_ROWS,
        CLIP_SCORES,
        ranking_scores(model),
        alpha=0.5,
        column=column,
        num_edges=num_edges,
    )


def test_scores_with_every_edge_are_the_model_scores():
    model = random_model()

    scores = graph_scores_with_edges(model, GRAPH, EVERY_EDGE, QUERY, CANDIDATE_ROWS)

    assert scores == pytest.approx(ranking_scores(model), abs=1e-6)


def test_explained_edges_are_exactly_the_edges_into_the_image():
    result = explain(identity_model())

    assert {(edge.source, edge.target) for edge in result.contributions} == {(3, 0), (4, 0)}


def test_removing_the_top_supporting_edge_lowers_the_score():
    model = identity_model()
    result = explain(model)
    top = result.contributions[0]

    without_top = EDGE_INDEX[:, ~((EDGE_INDEX[0] == top.source) & (EDGE_INDEX[1] == top.target))]

    assert (top.source, top.target) == (3, 0)  # the concept the query asks for
    assert direct_score(model, without_top, 0) < direct_score(model, EDGE_INDEX, 0)
    expected_drop = direct_score(model, EDGE_INDEX, 0) - direct_score(model, without_top, 0)
    assert top.contribution == pytest.approx(expected_drop, abs=1e-6)


def test_an_edge_that_dilutes_the_query_concept_is_opposing():
    # Image 0's mean is (x_3 + x_4) / 2; without concept 4 it is x_3, closer to the query.
    result = explain(identity_model())

    opposing = [edge for edge in result.contributions if edge.contribution < 0]
    assert [(edge.source, edge.target) for edge in opposing] == [(4, 0)]


def test_removing_an_edge_outside_the_image_neighbourhood_changes_nothing():
    model = random_model()
    keep = np.array([True, True, False, True, True])  # drop 4 -> 1, an edge of image 1

    before = graph_scores_with_edges(model, GRAPH, EVERY_EDGE, QUERY, np.array([0]))
    after = graph_scores_with_edges(model, GRAPH, keep, QUERY, np.array([0]))

    assert after == pytest.approx(before, abs=1e-7)


def test_deletion_check_matches_removing_the_edges_directly():
    model = random_model()
    result = explain(model, num_edges=2, column=0)

    removed = {(edge.source, edge.target) for edge in result.deletion.removed}
    keep = [
        (int(s), int(t)) not in removed for s, t in zip(EDGE_INDEX[0], EDGE_INDEX[1], strict=True)
    ]
    expected = direct_score(model, EDGE_INDEX[:, torch.tensor(keep)], int(CANDIDATE_ROWS[0]))

    assert result.deletion.graph_score_after == pytest.approx(expected, abs=1e-6)


def test_without_edges_removes_every_edge_into_the_image():
    model = random_model()
    result = explain(model, num_edges=1, column=0)
    image_row = int(CANDIDATE_ROWS[0])

    removed = {(edge.source, edge.target) for edge in result.without_edges.removed}
    keep = EDGE_INDEX[1] != image_row
    expected = direct_score(model, EDGE_INDEX[:, keep], image_row)

    assert removed == {(4, 1), (5, 1)}  # both concept edges of image 101
    assert result.without_edges.graph_score_after == pytest.approx(expected, abs=1e-6)


def test_deletion_check_removes_only_supporting_edges():
    result = explain(identity_model(), num_edges=3)

    # Image 0 has one supporting edge (3 -> 0); the opposing edge 4 -> 0 must stay.
    assert [(edge.source, edge.target) for edge in result.deletion.removed] == [(3, 0)]
    assert result.deletion.graph_score_after < result.graph_score


def test_deletion_check_reranks_with_the_saved_alpha():
    model = identity_model()
    result = explain(model, num_edges=1)

    # Re-rank by hand: only image 0's graph score changes (edges point at images only).
    graph_after = ranking_scores(model)
    graph_after[2] = result.deletion.graph_score_after
    final_after = 0.5 * CLIP_SCORES + 0.5 * graph_after
    expected_rank = 1 + int((final_after > final_after[2]).sum())

    assert result.deletion.rank_after == expected_rank


def test_ranks_are_reported_under_clip_and_clip_plus_graph():
    model = identity_model()
    result = explain(model)

    final = 0.5 * CLIP_SCORES + 0.5 * ranking_scores(model)
    assert result.clip_rank == 3
    assert result.rank == 1 + int((final > final[2]).sum())
    assert result.image_id == 100


def test_rejects_scores_that_did_not_come_from_this_model():
    model = identity_model()
    wrong = ranking_scores(model) + 0.1

    with pytest.raises(ValueError, match="does not reproduce"):
        explain_candidate(
            model, GRAPH, QUERY, CANDIDATE_ROWS, CLIP_SCORES, wrong, 0.5, column=2, num_edges=1
        )
