"""Retrieval + explanation of the top image, on the tiny dataset of test_models_training."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("faiss")

from mmxai.explain.demo import (  # noqa: E402
    cases_table,
    cases_to_markdown,
    explain_image,
    format_result,
    pick_fixed_cases,
    run_demo,
)
from mmxai.explain.figures import find_image  # noqa: E402
from mmxai.models.one_layer_sage import OneLayerSage, graph_scores  # noqa: E402
from mmxai.models.training import (  # noqa: E402
    load_bipartite,
    save_chosen_alpha,
    train_gnn,
)
from test_models_training import make_dataset  # noqa: E402

CPU = torch.device("cpu")


def test_explanation_names_the_concepts_of_the_image(tmp_path):
    graph = load_bipartite(make_dataset(tmp_path), train_only=False, device=CPU)
    model = OneLayerSage(graph.image_x.shape[1]).eval()
    with torch.no_grad():
        model.conv.lin_l.weight.copy_(torch.eye(graph.image_x.shape[1]))
    query = graph.concept_x[0]  # a query equal to concept "a": its edge must push up
    # Image row 0 has edges to concepts 0 ("a") and 1 ("b").
    ranked = float(
        graph_scores(query, model(graph.concept_x, graph.image_x[:1], graph.edge_index[:, :2])[0])
    )
    explanation = explain_image(
        model, graph, query, 0, ranked, alpha=0.0, top_m=3, min_contribution=0.0
    )
    assert explanation.graph_used
    assert explanation.supporting[0].source == "a"
    assert explanation.supporting[0].target == "image 100"


@torch.no_grad()
def test_removing_the_top_supporting_edge_lowers_the_score(tmp_path):
    graph = load_bipartite(make_dataset(tmp_path), train_only=False, device=CPU)
    model = OneLayerSage(graph.image_x.shape[1]).eval()
    with torch.no_grad():
        model.conv.lin_l.weight.copy_(torch.eye(graph.image_x.shape[1]))
    query = graph.concept_x[0]
    full = graph_scores(
        query, model(graph.concept_x, graph.image_x[:1], torch.tensor([[0, 1], [0, 0]]))
    )
    without_a = graph_scores(
        query, model(graph.concept_x, graph.image_x[:1], torch.tensor([[1], [0]]))
    )
    assert float(without_a[0]) < float(full[0])


def test_clip_mode_says_the_graph_was_not_used(tmp_path):
    config = make_dataset(tmp_path)
    results = run_demo(config, "clip", 0, "test", 2, [], 0, 3, 3, 0.0, None)
    assert len(results) == 2
    assert all(not result.explanation.graph_used for result in results)
    assert all(set(result.top_image_ids) <= {109, 110, 111} for result in results)


def test_clip_graph_mode_explains_the_top_image(tmp_path):
    config = make_dataset(tmp_path)
    train_gnn(config, seed=0)
    save_chosen_alpha(config, 0, 0.5)
    results = run_demo(config, "clip_graph", 0, "test", 3, [], 0, 3, 3, 0.0, None)
    assert len(results) == 3
    for result in results:
        assert result.explanation.graph_used
        assert result.first_relevant_rank is not None


def test_fixed_cases_follow_the_rule_in_text_id_order():
    # Positions 0..5 with text ids out of order; ranks: 1 = success, None or > 10 = failure.
    text_ids = np.array([50, 10, 40, 20, 30, 60])
    ranks = [1, None, 3, 1, 12, 1]
    success, failure = pick_fixed_cases(text_ids, ranks, num_cases=2, failure_after=10)
    assert success == [3, 0]  # text ids 20, 50 (the first two successes by id)
    assert failure == [1, 4]  # text ids 10 (no correct image), 30 (rank 12 > 10)


def test_fixed_cases_never_count_a_near_miss_as_failure():
    success, failure = pick_fixed_cases(np.array([1, 2]), [10, 2], num_cases=5, failure_after=10)
    assert success == []
    assert failure == []


def test_fixed_pick_returns_labelled_cases(tmp_path):
    config = make_dataset(tmp_path)
    results = run_demo(config, "clip", 0, "test", 2, [], 0, 3, 3, 0.0, None, pick="fixed")
    assert results
    assert {result.case for result in results} <= {"success", "failure"}
    for result in results:
        if result.case == "success":
            assert result.first_relevant_rank == 1


def test_results_report_the_clip_rank_and_the_source_image(tmp_path):
    config = make_dataset(tmp_path)
    train_gnn(config, seed=0)
    save_chosen_alpha(config, 0, 0.5)
    results = run_demo(config, "clip_graph", 0, "test", 3, [], 0, 3, 3, 0.0, None)
    for result in results:
        assert result.clip_rank is not None  # 3 test images, all in the top 3
        assert result.source_image_id in {109, 110, 111}
        assert result.source_rank is not None
        # The source image gets its own explanation only when it is not already first.
        if result.source_image_id == result.top_image_ids[0]:
            assert result.source_explanation is None
        else:
            assert result.source_explanation is not None


def test_cases_to_markdown_has_one_section_per_case(tmp_path):
    config = make_dataset(tmp_path)
    results = run_demo(config, "clip", 0, "test", 2, [], 0, 3, 3, 0.0, None)
    text = cases_to_markdown(results, "clip")
    assert text.startswith("## Cases (clip)")
    assert text.count("\n### ") == 2
    assert "With CLIP alone: rank" in text


def test_find_image_looks_in_every_folder(tmp_path):
    first, second = tmp_path / "VG_100K", tmp_path / "VG_100K_2"
    first.mkdir()
    second.mkdir()
    (second / "7.jpg").write_bytes(b"")
    assert find_image([first, second], 7) == second / "7.jpg"
    assert find_image([first, second], 8) is None


def test_format_result_shows_ranks_and_both_explanations(tmp_path):
    config = make_dataset(tmp_path)
    train_gnn(config, seed=0)
    save_chosen_alpha(config, 0, 0.5)
    results = run_demo(config, "clip_graph", 0, "test", 3, [], 0, 3, 3, 0.0, None)
    text = format_result(results[0], 1, "clip_graph")
    assert text.startswith("=== Query 1 (clip_graph)")
    assert "CLIP alone: rank" in text
    assert "is first:" in text
    with_source = [result for result in results if result.source_explanation is not None]
    for result in with_source:
        assert "Source image" in format_result(result, 1, "clip_graph")


def test_cases_table_has_one_row_per_case(tmp_path):
    config = make_dataset(tmp_path)
    results = run_demo(config, "clip", 0, "test", 2, [], 0, 3, 3, 0.0, None, pick="fixed")
    rows = cases_table(results)
    assert len(rows) == len(results)
    assert {"case", "query", "rank", "rank_clip_only", "pushed_up_by"} <= set(rows[0])
