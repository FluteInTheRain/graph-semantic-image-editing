"""explain_quantitatively measures the explanation of each result on the ranking model."""

import dataclasses

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("faiss")

from mmxai.explain.config import ExplainConfig  # noqa: E402
from mmxai.explain.query_path import search_and_explain  # noqa: E402
from mmxai.explain.query_quantitative import explain_quantitatively  # noqa: E402
from mmxai.explain.ranking_model import graph_scores_with_edges  # noqa: E402
from test_explain_query_path import QUERY, make_engine  # noqa: E402
from test_explain_ranking_model import EDGE_INDEX, GRAPH  # noqa: E402

# In the tiny graph, images 100 and 101 have 2 concept edges each, image 102 has 1.


def config(top_m: int = 1) -> ExplainConfig:
    return ExplainConfig(
        experiment="explain_test",
        ranking_config="unused.yaml",
        eval_split="val",
        num_queries=10,
        query_sample_seed=0,
        seeds=[0, 1, 2],
        explainers=["integrated_gradients"],
        num_layers=1,
        both_directions=False,
        top_m_options=[top_m],
        main_explainer="integrated_gradients",
        top_m=top_m,
        min_contribution=1e-6,
        plus_weight=0.5,
        random_repeats=5,
        robust_remove_prob=0.1,
        robust_add_back_prob=0.9,
        robust_repeats=5,
        ig_steps=20,
        gnn_explainer_epochs=1,
        gnn_explainer_learning_rate=1.0,
        gnn_explainer_size_weight=0.005,
        gnn_explainer_entropy_weight=1.0,
        output_dir="unused",
    )


def row_of(image_id: int) -> int:
    return int(np.flatnonzero(GRAPH.image_ids == image_id)[0])


def test_images_come_in_the_order_clip_plus_graph_shows_them():
    engine = make_engine(alpha=0.5)

    quantified = explain_quantitatively(engine, QUERY, 3, config(), seed=0)
    shown = search_and_explain(engine, QUERY, 3)

    assert [item.image_id for item in quantified] == [item.image_id for item in shown]
    assert [item.rank for item in quantified] == [1, 2, 3]


def test_explanation_edges_are_real_edges_into_the_image():
    edges = set(zip(EDGE_INDEX[0].tolist(), EDGE_INDEX[1].tolist(), strict=True))
    for item in explain_quantitatively(make_engine(alpha=0.5), QUERY, 3, config(), seed=0):
        assert len(item.explanation) == 1
        for concept_row, image_row in item.explanation:
            assert image_row == row_of(item.image_id)
            assert (concept_row, image_row) in edges


def test_fidelity_plus_is_the_score_lost_when_the_explanation_is_removed():
    engine = make_engine(alpha=0.5)
    for item in explain_quantitatively(engine, QUERY, 3, config(), seed=0):
        if item.report is None:
            continue
        removed = set(item.explanation)
        remaining = np.array(
            [
                (source, target) not in removed
                for source, target in zip(
                    EDGE_INDEX[0].tolist(), EDGE_INDEX[1].tolist(), strict=True
                )
            ]
        )
        row = np.array([row_of(item.image_id)])
        every = np.ones(EDGE_INDEX.shape[1], dtype=bool)
        full = graph_scores_with_edges(engine.model, GRAPH, every, QUERY, row)[0]
        without = graph_scores_with_edges(engine.model, GRAPH, remaining, QUERY, row)[0]
        assert item.report.fidelity_plus == pytest.approx(full - without, abs=1e-5)


def test_an_explanation_keeping_every_edge_is_not_measured():
    for item in explain_quantitatively(make_engine(alpha=0.5), QUERY, 3, config(), seed=0):
        if item.num_edges <= 1:  # image 102: its only edge is the whole explanation
            assert item.report is None
        else:
            assert item.report is not None


def test_alpha_one_measures_nothing_because_the_graph_takes_no_part():
    for item in explain_quantitatively(make_engine(alpha=1.0), QUERY, 3, config(), seed=0):
        assert item.report is None


def test_explainer_not_chosen_on_val_is_rejected():
    unchosen = dataclasses.replace(config(), main_explainer=None)
    with pytest.raises(ValueError):
        explain_quantitatively(make_engine(alpha=0.5), QUERY, 3, unchosen, seed=0)
