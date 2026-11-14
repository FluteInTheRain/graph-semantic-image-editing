"""Level-3 explanation evaluation end to end on the tiny files of test_graph_rerank_pipeline."""

import dataclasses
import json
from datetime import datetime

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("faiss")

from mmxai.eval.graph_rerank import (  # noqa: E402
    MODEL_FILE,
    SELECTION_FILE,
    build_split_graph,
    image_vectors,
    load_inputs,
    make_model,
    prepare,
    seed_model_dir,
    split_candidates,
    train_and_select,
)
from mmxai.explain.config import ExplainConfig, load_explain_config  # noqa: E402
from mmxai.explain.evaluation import (  # noqa: E402
    check_settings,
    evaluate_split,
    format_summary,
    make_explain_run_dir,
    pick_query_rows,
    write_evaluation,
)
from mmxai.explain.ranking_model import graph_scores_with_edges  # noqa: E402
from mmxai.rerank.fusion import candidate_graph_scores, fuse_scores, sweep_alpha  # noqa: E402
from test_graph_rerank_pipeline import write_inputs  # noqa: E402

SEEDS = [0, 1, 2]
EXPLAINERS = ["occlusion", "integrated_gradients", "gnn_explainer"]
TOP_M = [1, 2]  # every tiny image has 2 concept edges, so m = 2 is skipped
EXPLAINED_M = [1]
NUM_QUERIES = 4  # of the 6 val texts


def save_trained_seeds(config, alpha: float):
    """Train every seed, then give each a strong random W_nb so the graph moves ranks.

    Saves the val metrics of the new weights, as training does, so the evaluation can
    check that it reproduces them.
    """
    prepared = prepare(config, log=str)
    for seed in SEEDS:
        train_and_select(config, prepared, seed=seed, log=str)
        model = make_model(config, prepared.val_graph.x.shape[1])
        torch.manual_seed(seed)
        with torch.no_grad():
            model.conv.lin_l.weight.normal_()
        model_dir = seed_model_dir(config, seed)
        torch.save(model.state_dict(), model_dir / MODEL_FILE)
        val_scores = candidate_graph_scores(
            prepared.val_candidates,
            image_vectors(model, prepared.val_graph),
            prepared.val_graph.image_ids,
        )
        sweep = sweep_alpha(prepared.val_candidates, val_scores, [alpha], config.recall_ks)
        with open(model_dir / SELECTION_FILE, encoding="utf-8") as file:
            selection = json.load(file)
        selection["alpha"] = alpha
        selection["alpha_sweep"] = {str(key): value for key, value in sweep.items()}
        with open(model_dir / SELECTION_FILE, "w", encoding="utf-8") as file:
            json.dump(selection, file)


def explain_config(tmp_path, ranking_config, **changes) -> ExplainConfig:
    """The committed explain config, pointed at the tiny model and sized for it."""
    ranking_file = tmp_path / "ranking.yaml"
    ranking_file.write_text(yaml.safe_dump(dataclasses.asdict(ranking_config)), encoding="utf-8")
    config = load_explain_config("configs/explain_vg.yaml")
    values = {
        "ranking_config": str(ranking_file),
        "num_queries": NUM_QUERIES,
        "seeds": SEEDS,
        "explainers": EXPLAINERS,
        "top_m_options": TOP_M,
        "random_repeats": 5,
        "robust_repeats": 5,
        "ig_steps": 20,
        "gnn_explainer_epochs": 20,
        "output_dir": str(tmp_path / "explain_runs"),
    }
    values.update(changes)
    return dataclasses.replace(config, **values)


@pytest.fixture
def val_evaluation(tmp_path, monkeypatch):
    ranking = write_inputs(tmp_path, monkeypatch)
    save_trained_seeds(ranking, alpha=0.5)
    config = explain_config(tmp_path, ranking)
    return ranking, config, evaluate_split(config, log=str)


def test_pick_query_rows_is_repeatable_sorted_and_without_repeats():
    first = pick_query_rows(100, 20, sample_seed=0)

    assert first.tolist() == pick_query_rows(100, 20, sample_seed=0).tolist()
    assert first.tolist() == sorted(set(first.tolist()))
    assert len(first) == 20


def test_every_seed_query_explainer_and_size_gets_a_record(val_evaluation):
    _, _, evaluation = val_evaluation

    expected = len(SEEDS) * NUM_QUERIES * len(EXPLAINERS) * len(EXPLAINED_M)
    assert len(evaluation["records"]) == expected


def test_a_size_covering_every_edge_is_not_an_explanation(val_evaluation):
    _, _, evaluation = val_evaluation

    # With every edge kept, characterization is 1 by construction, so m = 2 is skipped.
    assert {record["top_m"] for record in evaluation["records"]} == {1}


def test_the_explained_image_is_the_one_ranked_first(tmp_path, val_evaluation):
    ranking, _, evaluation = val_evaluation
    inputs = load_inputs(ranking)
    candidates = split_candidates(ranking, inputs, "val")
    graph = build_split_graph(inputs, "val")

    for record in evaluation["records"]:
        model = make_model(ranking, graph.x.shape[1])
        model_dir = seed_model_dir(ranking, record["seed"])
        model.load_state_dict(torch.load(model_dir / MODEL_FILE, weights_only=True))
        scores = candidate_graph_scores(candidates, image_vectors(model, graph), graph.image_ids)
        final = fuse_scores(candidates.clip_scores, scores, record["alpha"])
        query = record["query_row"]
        first = candidates.candidate_ids[query, int(np.argsort(-final[query], kind="stable")[0])]
        assert record["image_id"] == first


def test_fidelity_plus_equals_removing_the_edges_on_the_full_graph(val_evaluation):
    ranking, _, evaluation = val_evaluation
    inputs = load_inputs(ranking)
    candidates = split_candidates(ranking, inputs, "val")
    graph = build_split_graph(inputs, "val")
    edge_index = graph.edge_index.numpy()
    image_row = {image_id: row for row, image_id in enumerate(graph.image_ids.tolist())}

    for record in evaluation["records"][:6]:
        model = make_model(ranking, graph.x.shape[1])
        model_dir = seed_model_dir(ranking, record["seed"])
        model.load_state_dict(torch.load(model_dir / MODEL_FILE, weights_only=True))
        keep = np.ones(edge_index.shape[1], dtype=bool)
        for source, target in record["relations"]:
            keep &= ~((edge_index[0] == source) & (edge_index[1] == target))
        query = candidates.query_embeddings[record["query_row"]]
        rows = np.array([image_row[record["image_id"]]])
        before = graph_scores_with_edges(model, graph, np.ones_like(keep), query, rows)[0]
        after = graph_scores_with_edges(model, graph, keep, query, rows)[0]
        assert record["fidelity_plus"] == pytest.approx(before - after, abs=1e-5)


def test_explanations_are_valid_and_sparsity_follows_m(val_evaluation):
    _, _, evaluation = val_evaluation

    for record in evaluation["records"]:
        assert record["all_in_reach"]
        assert record["connected_share"] == 1.0  # every edge points at the image
        assert record["sparsity"] == pytest.approx(1 - record["top_m"] / 2)


def test_val_summary_has_mean_and_std_and_proposes_a_choice(val_evaluation):
    _, _, evaluation = val_evaluation
    summary = evaluation["summary"]

    assert len(summary["rows"]) == len(EXPLAINERS) * len(EXPLAINED_M)
    assert set(summary["rows"][0]["fidelity_plus"]) == {"mean", "std"}
    assert summary["val_proposal"]["main_explainer"] in EXPLAINERS
    assert summary["val_proposal"]["top_m"] in EXPLAINED_M


def test_test_split_runs_only_the_val_choice(tmp_path, monkeypatch):
    ranking = write_inputs(tmp_path, monkeypatch)
    save_trained_seeds(ranking, alpha=0.5)
    config = explain_config(
        tmp_path,
        ranking,
        eval_split="test",
        main_explainer="occlusion",
        top_m=1,
        num_queries=3,
    )

    evaluation = evaluate_split(config, log=str)

    assert {(r["explainer"], r["top_m"]) for r in evaluation["records"]} == {("occlusion", 1)}
    assert evaluation["summary"]["val_proposal"] is None


def test_test_evaluation_runs_only_once(tmp_path):
    now = datetime(2026, 11, 14, 9, 0)
    run_dir = make_explain_run_dir(tmp_path, "explain_vg", "test", now)
    (run_dir / "evaluation.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="test runs once"):
        make_explain_run_dir(tmp_path, "explain_vg", "test", datetime(2026, 11, 14, 9, 5))


def test_run_folder_holds_the_evaluation_but_no_metrics_file(tmp_path, val_evaluation):
    _, config, evaluation = val_evaluation
    run_dir = make_explain_run_dir(tmp_path / "out", config.experiment, "val", datetime.now())

    write_evaluation(run_dir, evaluation, config)

    names = sorted(path.name for path in run_dir.iterdir())
    assert names == ["config.yaml", "env.json", "evaluation.json", "summary.md"]


def test_rejects_settings_that_do_not_match_the_one_layer_model(tmp_path, monkeypatch):
    ranking = write_inputs(tmp_path, monkeypatch)
    config = explain_config(tmp_path, ranking, num_layers=2)

    with pytest.raises(ValueError, match="num_layers"):
        check_settings(config, ranking)


def test_summary_table_shows_validity_columns():
    measure = {"mean": 0.5, "std": 0.1}
    row = {"explainer": "occlusion", "top_m": 3, "num_records": 6}
    for name in (
        "fidelity_plus",
        "random_plus_mean",
        "rank_drop",
        "random_rank_drop_mean",
        "fidelity_minus",
        "characterization",
        "robust_plus",
        "sparsity",
    ):
        row[name] = measure
    row["all_in_reach_share"] = 0.75
    row["connected_share"] = {"mean": 0.9, "std": 0.05}
    row["p_fidelity_plus_vs_random"] = 0.01
    row["p_rank_drop_vs_random"] = 0.02
    evaluation = {
        "settings": {
            "ranking_experiment": "toy",
            "split": "val",
            "seeds": [0, 1],
            "query_rows": [0, 1, 2],
            "explained_image": "rank 1",
        },
        "summary": {"rows": [row], "val_proposal": None},
    }

    lines = format_summary(evaluation).splitlines()
    header = [cell.strip() for cell in lines[4].strip("|").split("|")]
    cells = [cell.strip() for cell in lines[6].strip("|").split("|")]

    assert len(cells) == len(header) == lines[5].count("---")
    assert cells[header.index("valid: in reach")] == "0.75"
    assert cells[header.index("valid: connected")] == "0.9000 ± 0.0500"
