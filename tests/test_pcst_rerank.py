"""Selection logic of the PCST reranker (ADR-0051). Synthetic numbers only."""

import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from mmxai.eval import pcst_rerank
from mmxai.eval.pcst_rerank import (
    PcstRerankConfig,
    choose_setting,
    grid_search,
    load_pcst_rerank_config,
    top_concepts_per_query,
)
from mmxai.rerank.fusion import Candidates


def make_config(**changes) -> PcstRerankConfig:
    config = PcstRerankConfig(
        experiment="t",
        dataset="vg",
        split_file="s.json",
        features_dir="f",
        graph_dir="g",
        concept_edges_dir="c",
        subgraph_mode="pcst",
        shuffle_concept_edges=False,
        top_k=3,
        recall_ks=[1],
        same_text_threshold=0.99,
        prize_k_grid=[1, 2],
        edge_cost_grid=[0.5, 1.0],
        alpha_grid=[0.0, 0.5, 1.0],
        selection_metric="MRR@3",
        selection_dir="sel",
        output_dir="out",
    )
    return replace(config, **changes)


def test_choose_setting_takes_the_best_metric():
    results = [
        {"prize_k": 3, "edge_cost": 0.5, "alpha": 0.8, "metric": 0.12},
        {"prize_k": 5, "edge_cost": 0.5, "alpha": 0.7, "metric": 0.13},
    ]

    assert choose_setting(results)["prize_k"] == 5


def test_ties_prefer_larger_alpha_then_smaller_k_then_larger_cost():
    results = [
        {"prize_k": 5, "edge_cost": 0.5, "alpha": 0.8, "metric": 0.13},
        {"prize_k": 3, "edge_cost": 1.0, "alpha": 0.9, "metric": 0.13},
        {"prize_k": 3, "edge_cost": 0.5, "alpha": 0.9, "metric": 0.13},
    ]

    chosen = choose_setting(results)

    assert (chosen["alpha"], chosen["prize_k"], chosen["edge_cost"]) == (0.9, 3, 1.0)


def test_config_rejects_unknown_subgraph_mode(tmp_path):
    values = asdict(make_config(subgraph_mode="shortest_path"))
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")

    with pytest.raises(ValueError, match="subgraph_mode"):
        load_pcst_rerank_config(path)


def test_grid_search_on_synthetic_inputs():
    # Concepts 0, 1, 2 on a path 0 - 1 - 2. The query is closest to concept 0.
    concept_vectors = np.array([[1.0, 0.0], [0.7, 0.7], [0.0, 1.0]])
    pairs = np.array([[0, 1], [1, 2]])
    # Image 1 (CLIP's 2nd) has concept 0; image 2 (CLIP's 1st) has concept 2 only.
    sets = {1: {0}, 2: {2}, 3: set()}
    candidates = Candidates(
        query_embeddings=np.array([[1.0, 0.0]]),
        candidate_ids=np.array([[2, 1, 3]]),
        clip_scores=np.array([[0.9, 0.8, 0.1]], dtype=np.float32),
        relevant=[{1}],
    )

    results = grid_search(make_config(), candidates, sets, concept_vectors, pairs, log=print)

    settings = [(r["prize_k"], r["edge_cost"]) for r in results]
    assert settings == [(1, 0.5), (1, 1.0), (2, 0.5), (2, 1.0)]
    # alpha = 1 is plain CLIP: the relevant image 1 is at rank 2, so MRR = 0.5.
    # With prize_k = 1 the subgraph is {0}: image 1 scores 1, image 2 scores 0, and
    # a graph weight beats CLIP's 0.1 gap, so the best MRR is 1.0.
    assert results[0]["metric"] == 1.0
    assert results[0]["alpha"] < 1.0
    assert choose_setting(results)["metric"] == 1.0


CONFIG_DIR = Path(__file__).parent.parent / "configs"
CONFIG_NAMES = ["clip_pcst_vg", "clip_pcst_vg_no_tree", "clip_pcst_vg_shuffled"]
ABLATION_FIELDS = {"experiment", "subgraph_mode", "shuffle_concept_edges", "selection_dir"}


def test_the_three_configs_differ_only_in_the_ablation_fields():
    main, no_tree, shuffled = [
        asdict(load_pcst_rerank_config(CONFIG_DIR / f"{name}.yaml")) for name in CONFIG_NAMES
    ]

    for other in (no_tree, shuffled):
        differing = {key for key in main if main[key] != other[key]}
        assert differing <= ABLATION_FIELDS
    assert main["subgraph_mode"] == "pcst"
    assert main["shuffle_concept_edges"] is False
    assert no_tree["subgraph_mode"] == "top_k_only"
    assert shuffled["shuffle_concept_edges"] is True


def test_top_concepts_do_not_depend_on_the_chunk_size():
    rng = np.random.default_rng(1)
    queries = rng.normal(size=(7, 4))
    concepts = rng.normal(size=(12, 4))

    whole = top_concepts_per_query(queries, concepts, 3, chunk=100)
    chunked = top_concepts_per_query(queries, concepts, 3, chunk=2)

    assert whole.shape == (7, 3)
    assert whole.tolist() == chunked.tolist()


# --- guards and the val-only rule, with fake inputs (no artifacts on disk) ---


def fake_inputs(edge_source: str = "clip_topk") -> SimpleNamespace:
    """Just the fields pcst_rerank reads: three concepts on a path, no real graph."""
    edges = SimpleNamespace(edge_source=np.array([0, 1]), edge_target=np.array([1, 2]))
    return SimpleNamespace(
        graph="graph",
        concept_features=np.array([[1.0, 0.0], [0.7, 0.7], [0.0, 1.0]]),
        graph_meta={"edge_source": edge_source, "git": {"git_commit": "g"}},
        feature_meta={"code_hash": "f"},
        concept_edges=edges,
    )


def fake_candidates() -> Candidates:
    return Candidates(
        query_embeddings=np.array([[1.0, 0.0]]),
        candidate_ids=np.array([[2, 1, 3]]),
        clip_scores=np.array([[0.9, 0.8, 0.1]], dtype=np.float32),
        relevant=[{1}],
    )


@pytest.fixture
def patched(monkeypatch, tmp_path):
    """Replace every disk-reading step of pcst_rerank; record which splits were asked for."""
    asked = {"candidates": [], "subgraph": []}

    def split_candidates(config, inputs, split_name):
        asked["candidates"].append(split_name)
        return fake_candidates()

    def split_subgraph(graph, split_name):
        asked["subgraph"].append(split_name)
        return graph

    monkeypatch.setattr(pcst_rerank, "load_inputs", lambda config: fake_inputs())
    monkeypatch.setattr(pcst_rerank, "split_candidates", split_candidates)
    monkeypatch.setattr(pcst_rerank, "split_subgraph", split_subgraph)
    monkeypatch.setattr(pcst_rerank, "image_concept_sets", lambda g: {1: {0}, 2: {2}, 3: set()})
    monkeypatch.setattr(pcst_rerank, "file_sha256", lambda path: "hash")
    monkeypatch.setattr(pcst_rerank, "git_state", lambda: {"git_commit": "c"})
    monkeypatch.setattr(pcst_rerank, "write_run", lambda config, metrics, seed: tmp_path)
    config = make_config(selection_dir=str(tmp_path / "sel"), output_dir=str(tmp_path / "out"))
    return config, asked


def write_selection(config, seed, **changes):
    selection = {
        "prize_k": 1,
        "edge_cost": 0.5,
        "alpha": 0.5,
        "split_sha256": "hash",
        "git": {"git_commit": "c"},
    }
    selection.update(changes)
    path = pcst_rerank.selection_path(config, seed)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(selection), encoding="utf-8")


def test_select_refuses_the_annotation_graph(patched, monkeypatch):
    config, _ = patched
    monkeypatch.setattr(pcst_rerank, "load_inputs", lambda c: fake_inputs("annotations"))

    with pytest.raises(ValueError, match="clip_topk"):
        pcst_rerank.select_on_val(config, 0, log=lambda *a: None)


def test_evaluate_refuses_the_annotation_graph(patched, monkeypatch):
    config, _ = patched
    write_selection(config, 0)
    monkeypatch.setattr(pcst_rerank, "load_inputs", lambda c: fake_inputs("annotations"))

    with pytest.raises(ValueError, match="clip_topk"):
        pcst_rerank.evaluate_test(config, 0, log=lambda *a: None)


def test_run_facts_record_the_graph_edge_source(patched):
    config, _ = patched

    facts = pcst_rerank.run_facts(config, fake_inputs(), "val", 0)

    assert facts["graph_edge_source"] == "clip_topk"


def test_evaluate_refuses_a_selection_made_on_another_split_file(patched):
    config, _ = patched
    write_selection(config, 0, split_sha256="old")

    with pytest.raises(ValueError, match="split"):
        pcst_rerank.evaluate_test(config, 0, log=lambda *a: None)
