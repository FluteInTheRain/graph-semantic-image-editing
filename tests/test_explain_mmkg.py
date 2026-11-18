import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from mmxai.eval.mmkg_transfer import build_transfer_graph, clip_candidates  # noqa: E402
from mmxai.explain.config import ExplainConfig  # noqa: E402
from mmxai.explain.mmkg import (  # noqa: E402
    check_reproduces_transfer,
    explain_seed,
    summarize_mmkg,
    table_rows,
)
from mmxai.models.sage import ConceptToImageSage  # noqa: E402

DIM = 4
CONCEPTS = ["drum", "bronze", "pagoda", "lake"]
IMAGE_IDS = np.array([0, 1, 2], dtype=np.int64)


def unit(rows) -> np.ndarray:
    array = np.array(rows, dtype=np.float32)
    return array / np.linalg.norm(array, axis=1, keepdims=True)


IMAGES = unit([[1, 0.2, 0, 0], [0.9, 0.1, 0.2, 0], [0, 0, 1, 0.3]])
CONCEPT_VECTORS = unit([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
QUERIES = [
    {"qid": "a_direct_en", "type": "direct", "lang": "en", "relevant_items": ["A"]},
    {"qid": "a_indirect_vi", "type": "indirect", "lang": "vi", "relevant_items": ["A"]},
]
QUERY_VECTORS = unit([[1, 0.1, 0, 0], [0, 0.2, 1, 0]])


def config(top_m: int = 2) -> ExplainConfig:
    return ExplainConfig(
        experiment="explain_test",
        ranking_config="unused.yaml",
        eval_split="test",
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


def setup(seed: int = 0):
    graph = build_transfer_graph(IMAGE_IDS, IMAGES, CONCEPTS, CONCEPT_VECTORS, [], k=3)
    candidates = clip_candidates(QUERY_VECTORS, IMAGES, IMAGE_IDS, [{0, 1}, {2}], top_k=3)
    torch.manual_seed(seed)
    model = ConceptToImageSage(DIM)
    with torch.no_grad():
        # Images hear their concepts, so edges really move the score.
        model.conv.lin_l.weight.copy_(torch.eye(DIM) + 0.3 * torch.randn(DIM, DIM))
    return graph, candidates, model


def test_reproduction_check_stops_on_a_different_mrr():
    with pytest.raises(ValueError, match="transfer run recorded"):
        check_reproduces_transfer([1, 2], top_k=3, expected_mrr=0.5)
    check_reproduces_transfer([1, 2], top_k=3, expected_mrr=0.75)  # no error


def test_explanations_are_edges_of_the_rank1_image_and_change_its_score():
    graph, candidates, model = setup()

    records = explain_seed(model, 0.0, 0, graph, candidates, QUERIES, CONCEPTS, config(), None)

    assert len(records) == len(QUERIES)
    for record in records:
        image_row = record["image_id"]  # ids equal rows in this graph
        edges = {tuple(edge) for edge in graph.edge_index.T.tolist()}
        assert all(tuple(edge) in edges and edge[1] == image_row for edge in record["relations"])
        assert len(record["relations"]) == 2
        assert record["fidelity_plus"] != pytest.approx(0.0)
        assert record["connected_share"] == pytest.approx(1.0)  # validity


def test_explaining_stops_when_the_ranking_is_not_the_reported_one():
    graph, candidates, model = setup()

    with pytest.raises(ValueError, match="transfer run recorded"):
        explain_seed(model, 0.0, 0, graph, candidates, QUERIES, CONCEPTS, config(), 0.123)


def test_m_covering_every_edge_is_skipped():
    graph, candidates, model = setup()

    records = explain_seed(
        model, 0.0, 0, graph, candidates, QUERIES, CONCEPTS, config(top_m=3), None, log=print
    )

    assert records == []


def test_summary_and_table_have_one_row_overall_and_per_group():
    graph, candidates, model = setup()
    records = []
    for seed in (0, 1, 2):
        records += explain_seed(
            model, 0.0, seed, graph, candidates, QUERIES, CONCEPTS, config(), None
        )

    rows = table_rows(summarize_mmkg(records, config()), "run_x")

    assert [row["group"] for row in rows] == ["all", "direct_en", "indirect_vi"]
    assert rows[0]["num_records"] == 6
    assert "fidelity_plus_mean" in rows[0] and "p_rank_drop_vs_random" in rows[0]
    assert rows[0]["run"] == "run_x"
