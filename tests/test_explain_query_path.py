"""search_and_explain ranks like the evaluation and explains with the ranking's own scores."""

import dataclasses

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("faiss")

from mmxai.eval.graph_rerank import image_vectors  # noqa: E402
from mmxai.explain.query_path import QueryEngine, search_and_explain  # noqa: E402
from mmxai.explain.ranking_model import graph_scores_with_edges  # noqa: E402
from mmxai.index.image_index import build_image_index  # noqa: E402
from test_explain_ranking_model import EDGE_INDEX, GRAPH, random_model  # noqa: E402

# CLIP image vectors of the 3 images, not equal to their graph rows, so CLIP and the
# graph disagree on some candidates.
IMAGE_EMBEDDINGS = np.array(
    [
        [0.6, 0.8, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.6, 0.8, 0.0, 0.0, 0.0],
        [0.8, 0.0, 0.6, 0.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)
QUERY = np.array([0.5, 0.5, 0.5, 0.5, 0.0, 0.0], dtype=np.float32)  # length 1


def make_engine(alpha: float) -> QueryEngine:
    model = random_model()
    return QueryEngine(
        model=model,
        split_graph=GRAPH,
        image_h=image_vectors(model, GRAPH),
        image_index=build_image_index(IMAGE_EMBEDDINGS, GRAPH.image_ids),
        alpha=alpha,
        top_k=3,
    )


def expected_final_scores(engine: QueryEngine) -> np.ndarray:
    """final score of image rows 0, 1, 2, computed without search_and_explain."""
    every_edge = np.ones(EDGE_INDEX.shape[1], dtype=bool)
    rows = np.array([0, 1, 2])
    graph = graph_scores_with_edges(engine.model, GRAPH, every_edge, QUERY, rows)
    clip = IMAGE_EMBEDDINGS @ QUERY
    return engine.alpha * clip + (1 - engine.alpha) * graph


def test_results_are_ordered_by_the_fused_score():
    engine = make_engine(alpha=0.5)
    results = search_and_explain(engine, QUERY, k=3)

    expected = expected_final_scores(engine)
    expected_ids = GRAPH.image_ids[np.argsort(-expected)].tolist()
    assert [item.image_id for item in results] == expected_ids
    assert [item.rank for item in results] == [1, 2, 3]
    for item in results:
        row = int(np.flatnonzero(GRAPH.image_ids == item.image_id)[0])
        assert item.final_score == pytest.approx(expected[row], abs=1e-5)


def test_alpha_one_keeps_the_clip_order():
    results = search_and_explain(make_engine(alpha=1.0), QUERY, k=3)

    assert [item.clip_rank for item in results] == [1, 2, 3]
    assert [item.rank for item in results] == [1, 2, 3]


def test_every_path_adds_up_to_the_graph_score_that_ranked_the_image():
    for item in search_and_explain(make_engine(alpha=0.5), QUERY, k=3):
        path = item.path
        total = path.self_share + path.bias_share + sum(c.share for c in path.concept_shares)
        assert total == pytest.approx(item.graph_score, abs=1e-5)


def test_occlusion_drop_is_the_score_lost_when_the_edge_is_removed():
    engine = make_engine(alpha=0.5)
    results = search_and_explain(engine, QUERY, k=3)
    image_0 = next(item for item in results if item.image_id == 100)

    keep = np.ones(EDGE_INDEX.shape[1], dtype=bool)
    keep[0] = False  # the edge concept 3 -> image 0
    after = graph_scores_with_edges(engine.model, GRAPH, keep, QUERY, np.array([0]))[0]
    assert image_0.occlusion_drop[3] == pytest.approx(image_0.graph_score - after, abs=1e-5)
    assert sorted(image_0.occlusion_drop) == [3, 4]  # only the image's own edges


def test_k_limits_the_number_of_explained_images():
    assert len(search_and_explain(make_engine(alpha=0.5), QUERY, k=2)) == 2


@pytest.mark.parametrize("k", [0, 4])
def test_k_outside_the_candidates_is_rejected(k):
    with pytest.raises(ValueError, match="k must be"):
        search_and_explain(make_engine(alpha=0.5), QUERY, k=k)


def test_graph_vectors_from_another_model_are_rejected():
    engine = make_engine(alpha=0.5)
    # image_h from a different model: the paths would explain a score that did not rank.
    torch.manual_seed(1)
    other = random_model()
    with torch.no_grad():
        other.conv.lin_r.weight.normal_()
    stale = dataclasses.replace(engine, image_h=image_vectors(other, GRAPH))

    with pytest.raises(ValueError, match="the ranking used"):
        search_and_explain(stale, QUERY, k=3)
