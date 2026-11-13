"""scripts/explain.py logic end to end on the tiny files of test_graph_rerank_pipeline."""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("faiss")

from mmxai.eval.graph_rerank import (  # noqa: E402
    MODEL_FILE,
    SELECTION_FILE,
    graph_only_val_score,
    make_model,
    prepare,
    seed_model_dir,
    train_and_select,
)
from mmxai.explain.query_analysis import analyse_split, write_analysis  # noqa: E402
from test_graph_rerank_pipeline import write_inputs  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def saved_model_with_graph(config, alpha: float):
    """Train seed 0, then replace its weights with a strong random W_nb and set alpha.

    The tiny training run stays at its CLIP-like start, so no rank would move; a
    random W_nb makes the graph change ranks while keeping the saved-file format.
    """
    prepared = prepare(config, log=str)
    train_and_select(config, prepared, seed=0, log=str)
    model = make_model(config, prepared.val_graph.x.shape[1])
    torch.manual_seed(0)
    with torch.no_grad():
        model.conv.lin_l.weight.normal_()
    model_dir = seed_model_dir(config, 0)
    torch.save(model.state_dict(), model_dir / MODEL_FILE)
    with open(model_dir / SELECTION_FILE, encoding="utf-8") as file:
        selection = json.load(file)
    selection["alpha"] = alpha
    with open(model_dir / SELECTION_FILE, "w", encoding="utf-8") as file:
        json.dump(selection, file)
    return model, prepared


def write_region_file(directory: Path) -> None:
    """A tiny region_descriptions.json: text id k of the features is region k."""
    entries = []
    for region_id in range(24):
        image_id = 1 + region_id // 2
        region = {"image_id": image_id, "region_id": region_id, "phrase": f"phrase {region_id}"}
        entries.append({"id": image_id, "regions": [region]})
    directory.mkdir()
    with open(directory / "region_descriptions.json", "w", encoding="utf-8") as file:
        json.dump(entries, file)


def test_ranks_are_the_ones_the_pipeline_reports(tmp_path, monkeypatch):
    config = write_inputs(tmp_path, monkeypatch)
    model, prepared = saved_model_with_graph(config, alpha=0.0)

    analysis = analyse_split(config, 0, "val", 1, None, log=str)

    # alpha = 0 is the graph score alone, which training's early stopping also measures.
    expected = graph_only_val_score(config, model, prepared)
    assert analysis["summary"]["graph_metrics"]["MRR@3"] == pytest.approx(expected)


def test_picked_queries_moved_in_the_right_direction(tmp_path, monkeypatch):
    config = write_inputs(tmp_path, monkeypatch)
    saved_model_with_graph(config, alpha=0.0)

    analysis = analyse_split(config, 0, "val", 1, None, log=str)

    groups = [item["group"] for item in analysis["queries"]]
    assert "success" in groups and "failure" in groups  # the random W_nb moves ranks
    for item in analysis["queries"]:
        if item["group"] == "success":
            assert item["graph_rank"] < item["clip_rank"]
        else:
            assert item["graph_rank"] > item["clip_rank"]


def test_explained_relevant_image_has_the_query_rank(tmp_path, monkeypatch):
    config = write_inputs(tmp_path, monkeypatch)
    saved_model_with_graph(config, alpha=0.0)

    analysis = analyse_split(config, 0, "val", 1, None, log=str)

    for item in analysis["queries"]:
        relevant = item["explained"][0]
        assert relevant["image_id"] in item["relevant_image_ids"]
        assert relevant["graph_rank"] == item["graph_rank"]


def test_failures_also_explain_the_image_ranked_first(tmp_path, monkeypatch):
    config = write_inputs(tmp_path, monkeypatch)
    saved_model_with_graph(config, alpha=0.0)

    analysis = analyse_split(config, 0, "val", 1, None, log=str)

    for item in analysis["queries"]:
        if item["group"] == "failure":
            assert [candidate["graph_rank"] for candidate in item["explained"]] == [
                item["graph_rank"],
                1,
            ]


def test_query_text_is_read_from_the_region_file(tmp_path, monkeypatch):
    config = write_inputs(tmp_path, monkeypatch)
    saved_model_with_graph(config, alpha=0.0)
    write_region_file(tmp_path / "vg")

    analysis = analyse_split(config, 0, "val", 1, tmp_path / "vg", log=str)

    for item in analysis["queries"]:
        assert item["text"] == f"phrase {item['text_id']}"


def test_run_folder_has_the_analysis_but_no_metrics_file(tmp_path, monkeypatch):
    config = write_inputs(tmp_path, monkeypatch)
    saved_model_with_graph(config, alpha=0.0)
    analysis = analyse_split(config, 0, "val", 1, None, log=str)

    run_dir = write_analysis(config, analysis, {"seed": 0}, datetime(2026, 11, 13, 10, 0))

    assert run_dir.name == "20261113-1000_clip_graph_tiny_explain_seed0"
    names = sorted(path.name for path in run_dir.iterdir())
    assert names == ["analysis.json", "analysis.md", "config.yaml", "env.json"]


def test_test_split_is_refused(tmp_path, monkeypatch):
    config = write_inputs(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="only val"):
        analyse_split(config, 0, "test", 3, None, log=str)


def test_cli_refuses_the_test_split():
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "explain.py"),
        "--config",
        str(REPO_ROOT / "configs" / "clip_graph_vg_lr1e4.yaml"),
        "--seed",
        "0",
        "--split",
        "test",
    ]

    result = subprocess.run(command, capture_output=True, text=True, cwd=REPO_ROOT, check=False)

    assert result.returncode != 0
    assert "only val" in result.stderr
