"""Stage 2 pipeline of ADR-0051: cache, train, test-once guards. Synthetic objects only."""

import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from mmxai.eval import pcst_sage_rerank
from mmxai.eval.pcst_sage_rerank import (
    PcstSageConfig,
    build_subgraph_training_data,
    cache_path,
    evaluate_test,
    load_or_compute_test_rows,
    load_pcst_sage_config,
    prepare_caches,
)


def make_config(tmp_path, **changes) -> PcstSageConfig:
    config = PcstSageConfig(
        experiment="t",
        dataset="vg",
        split_file="s.json",
        features_dir="f",
        graph_dir="g",
        concept_edges_dir="c",
        subgraph_mode="pcst",
        shuffle_concept_edges=False,
        prize_k=3,
        edge_cost=0.5,
        cache_dir=str(tmp_path / "cache"),
        model_dir=str(tmp_path / "model"),
        top_k=3,
        recall_ks=[1],
        same_text_threshold=0.99,
        num_hard_negatives=2,
        temperature=0.05,
        learning_rate=0.001,
        batch_size=2,
        max_epochs=1,
        patience=1,
        score_chunk=2,
        alpha_grid=[0.0, 1.0],
        selection_metric="MRR@3",
        output_dir=str(tmp_path / "runs"),
    )
    return replace(config, **changes)


def test_bad_subgraph_mode_is_rejected(tmp_path):
    config = make_config(tmp_path, subgraph_mode="bogus")
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config.__dict__), encoding="utf-8")

    with pytest.raises(ValueError, match="subgraph_mode"):
        load_pcst_sage_config(path)


def test_cache_path_is_shared_unless_shuffled(tmp_path):
    plain = make_config(tmp_path)
    shuffled = make_config(tmp_path, shuffle_concept_edges=True)

    assert cache_path(plain, 0) == cache_path(plain, 2)
    assert cache_path(plain, 0).name == "shared"
    assert cache_path(shuffled, 1).name == "seed1"


def test_prepare_caches_never_asks_for_test(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    inputs = SimpleNamespace(
        graph_meta={"git": {"git_commit": "abc"}}, concept_features=np.zeros((2, 3))
    )
    asked_graphs, asked_candidates, saved = [], [], []

    def fake_candidates(_config, _inputs, split_name):
        asked_candidates.append(split_name)
        return SimpleNamespace(query_embeddings=np.zeros((4, 3), dtype=np.float32))

    def fake_graph(_inputs, split_name):
        asked_graphs.append(split_name)
        return SimpleNamespace(image_ids=np.arange(2))

    def fake_save(_folder, split, _offsets, _values, _meta):
        saved.append(split)

    training = SimpleNamespace(text_x=torch.zeros((2, 3)))
    rows = (np.zeros(3, dtype=np.int64), np.zeros(0, dtype=np.int64))
    monkeypatch.setattr(pcst_sage_rerank, "load_checked_inputs", lambda _c: inputs)
    monkeypatch.setattr(pcst_sage_rerank, "concept_pairs", lambda *_a: np.zeros((0, 2)))
    monkeypatch.setattr(pcst_sage_rerank, "build_split_graph", fake_graph)
    monkeypatch.setattr(pcst_sage_rerank, "build_training_data", lambda *_a: training)
    monkeypatch.setattr(pcst_sage_rerank, "split_candidates", fake_candidates)
    monkeypatch.setattr(pcst_sage_rerank, "compute_subgraph_rows", lambda *_a: rows)
    monkeypatch.setattr(pcst_sage_rerank, "save_cache", fake_save)
    monkeypatch.setattr(pcst_sage_rerank, "file_sha256", lambda _p: "sha")

    prepare_caches(config, 0, log=lambda _m: None)

    assert asked_graphs == ["train"]
    assert asked_candidates == ["val"]
    assert saved == ["train", "val"]


def test_evaluate_test_refuses_a_second_run(tmp_path, monkeypatch):
    monkeypatch.setattr(pcst_sage_rerank, "has_test_run", lambda _c, _s: True)

    with pytest.raises(FileExistsError, match="already run"):
        evaluate_test(make_config(tmp_path), 0)


def test_evaluate_test_refuses_a_different_split_file(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    seed_dir = tmp_path / "model" / "seed0"
    seed_dir.mkdir(parents=True)
    selection = {"alpha": 0.5, "best_epoch": 1, "git": "x", "split_sha256": "old"}
    (seed_dir / "selection.json").write_text(json.dumps(selection), encoding="utf-8")
    monkeypatch.setattr(pcst_sage_rerank, "has_test_run", lambda _c, _s: False)
    monkeypatch.setattr(pcst_sage_rerank, "file_sha256", lambda _p: "new")

    with pytest.raises(ValueError, match="different split file"):
        evaluate_test(config, 0)


def test_training_data_rejects_a_different_image_order(monkeypatch):
    contains = SimpleNamespace(image_ids=np.array([1, 2, 3]))
    inputs = SimpleNamespace(graph=object())
    train_graph = SimpleNamespace(image_ids=np.array([1, 3, 2]), x=torch.zeros((5, 2)))
    monkeypatch.setattr(pcst_sage_rerank, "split_subgraph", lambda _g, _s: contains)

    with pytest.raises(ValueError, match="different order"):
        build_subgraph_training_data(None, inputs, train_graph, np.zeros((1, 1), dtype=np.int64))


def test_committed_stage2_configs_differ_only_where_intended():
    names = ["clip_pcst_sage_vg", "clip_pcst_sage_vg_no_tree", "clip_pcst_sage_vg_shuffled"]
    configs = [load_pcst_sage_config(f"configs/{name}.yaml") for name in names]
    allowed = {"experiment", "subgraph_mode", "shuffle_concept_edges", "cache_dir", "model_dir"}

    main = configs[0]
    assert main.prize_k == 3
    assert main.edge_cost == 0.5
    for other in configs[1:]:
        different = {key for key, value in asdict(main).items() if asdict(other)[key] != value}
        assert different <= allowed
    assert configs[1].subgraph_mode == "top_k_only"
    assert configs[2].shuffle_concept_edges is True
    assert len({config.cache_dir for config in configs}) == 3
    assert len({config.model_dir for config in configs}) == 3


def test_test_cache_is_computed_once_and_shared_by_seeds(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    inputs = SimpleNamespace(graph_meta={"git": {"git_commit": "abc"}})
    candidates = SimpleNamespace(query_embeddings=np.zeros((2, 3), dtype=np.float32))
    calls = []

    def fake_compute(*_args):
        calls.append(1)
        return np.array([0, 1, 2]), np.array([5, 6])

    monkeypatch.setattr(pcst_sage_rerank, "compute_subgraph_rows", fake_compute)
    monkeypatch.setattr(pcst_sage_rerank, "file_sha256", lambda _p: "sha")

    first = load_or_compute_test_rows(config, inputs, 0, None, candidates)
    second = load_or_compute_test_rows(config, inputs, 1, None, candidates)

    assert len(calls) == 1
    assert np.array_equal(first[1], second[1])


def test_test_cache_with_other_settings_is_refused(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    inputs = SimpleNamespace(graph_meta={"git": {"git_commit": "abc"}})
    candidates = SimpleNamespace(query_embeddings=np.zeros((2, 3), dtype=np.float32))
    rows = (np.array([0, 1, 2]), np.array([5, 6]))
    monkeypatch.setattr(pcst_sage_rerank, "compute_subgraph_rows", lambda *_a: rows)
    monkeypatch.setattr(pcst_sage_rerank, "file_sha256", lambda _p: "sha")
    load_or_compute_test_rows(config, inputs, 0, None, candidates)

    with pytest.raises(ValueError, match="prize_k"):
        load_or_compute_test_rows(replace(config, prize_k=5), inputs, 1, None, candidates)
