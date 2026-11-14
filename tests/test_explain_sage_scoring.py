"""The per-image score functions give the trained model's scores on the full graph."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("faiss")

from mmxai.eval.query_ranks import rank_of_column  # noqa: E402
from mmxai.explain.checks import check_gradient_function  # noqa: E402
from mmxai.explain.fidelity import rank_of  # noqa: E402
from mmxai.explain.ranking_model import graph_scores_with_edges  # noqa: E402
from mmxai.explain.sage_scoring import candidate_scores, image_scorer  # noqa: E402
from mmxai.explain.subgraph import relations_in_reach  # noqa: E402
from test_explain_ranking_model import (  # noqa: E402
    CANDIDATE_ROWS,
    EDGE_INDEX,
    GRAPH,
    QUERY,
    random_model,
)

IMAGE_ROW = 0  # image 0 has concepts 3 and 4 (columns 0 and 1 of EDGE_INDEX)


def full_graph_score(model, keep_columns: np.ndarray, image_row: int) -> float:
    """The image's score with the model run on the whole split graph."""
    return float(
        graph_scores_with_edges(model, GRAPH, keep_columns, QUERY, np.array([image_row]))[0]
    )


def test_scorer_keeps_only_the_edges_into_the_image():
    scorer = image_scorer(random_model(), GRAPH, IMAGE_ROW, QUERY)

    assert scorer.edge_index.tolist() == [[3, 4], [0, 0]]


@pytest.mark.parametrize("kept", [[True, True], [True, False], [False, True], [False, False]])
def test_score_image_equals_the_model_on_the_full_graph(kept):
    model = random_model()
    scorer = image_scorer(model, GRAPH, IMAGE_ROW, QUERY)
    # The same removal on the full graph: edges into other images stay.
    keep_columns = np.ones(EDGE_INDEX.shape[1], dtype=bool)
    keep_columns[:2] = kept

    assert scorer.score_image(np.array(kept)) == pytest.approx(
        full_graph_score(model, keep_columns, IMAGE_ROW), abs=1e-6
    )


def test_weights_zero_and_one_act_like_removed_and_kept_edges():
    scorer = image_scorer(random_model(), GRAPH, IMAGE_ROW, QUERY)
    relations = relations_in_reach(scorer.edge_index, IMAGE_ROW, 1, both_directions=False)

    # Raises if any 0/1 weighting disagrees with real removal.
    check_gradient_function(
        scorer.edge_index, relations, False, scorer.score_image, scorer.score_and_gradient
    )


def test_gradient_matches_a_finite_difference():
    scorer = image_scorer(random_model(), GRAPH, IMAGE_ROW, QUERY)
    weights = np.array([0.7, 0.9])
    _, gradient = scorer.score_and_gradient(weights)

    step = 1e-3
    for edge in range(2):
        up = weights.copy()
        up[edge] += step
        down = weights.copy()
        down[edge] -= step
        numeric = (scorer.score_and_gradient(up)[0] - scorer.score_and_gradient(down)[0]) / (
            2 * step
        )
        assert gradient[edge] == pytest.approx(numeric, abs=1e-3)


def test_candidate_scores_change_only_the_explained_column():
    model = random_model()
    ranked = np.array(
        [full_graph_score(model, np.ones(5, dtype=bool), row) for row in CANDIDATE_ROWS]
    )
    column = 2  # CANDIDATE_ROWS[2] is image 0
    scorer = image_scorer(model, GRAPH, IMAGE_ROW, QUERY)

    scores = candidate_scores(scorer, ranked, column)(np.array([False, True]))

    keep_columns = np.array([False, True, True, True, True])
    expected = [full_graph_score(model, keep_columns, row) for row in CANDIDATE_ROWS]
    assert scores == pytest.approx(expected, abs=1e-6)


def test_rank_of_matches_the_pipeline_rank_with_ties():
    final = np.array([0.5, 0.9, 0.5, 0.5, 0.1])

    for column in range(len(final)):
        assert rank_of(final, column) == rank_of_column(final, column)
