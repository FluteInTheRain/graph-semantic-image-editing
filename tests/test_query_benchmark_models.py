"""Trained re-rankers on the query benchmark: synthetic data, no network, no pcst_fast."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mmxai.eval import query_benchmark_models as module
from mmxai.eval.graph_rerank import SplitGraph, image_vectors
from mmxai.eval.query_benchmark_models import (
    GraphRerankRanker,
    ModelBenchmarkEvalConfig,
    PcstSageRanker,
    clip_candidates,
    has_benchmark_run,
    load_model_benchmark_eval_config,
    reranked_ids,
    run_model_benchmark_evaluation,
)
from mmxai.graph.contains import ContainsGraph
from mmxai.graph.pcst_cache import pack_subgraphs
from mmxai.models.sage import ConceptToImageSage
from mmxai.models.subgraph_sage import SubgraphSage
from mmxai.rerank.fusion import Candidates
from mmxai.utils.runs import file_sha256

DIM = 8
NUM_IMAGES = 12
IMAGE_IDS = np.arange(100, 100 + NUM_IMAGES, dtype=np.int64)


def unit_rows(count: int, seed: int) -> np.ndarray:
    rows = np.random.default_rng(seed).normal(size=(count, DIM)).astype(np.float32)
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


IMAGES = unit_rows(NUM_IMAGES, seed=0)
QUERIES = unit_rows(3, seed=1)


def expected_clip_order(top_k: int) -> np.ndarray:
    scores = QUERIES @ IMAGES.T
    order = np.argsort(-scores, axis=1)[:, :top_k]
    return IMAGE_IDS[order]


def test_clip_candidates_are_top_k_best_first():
    candidates = clip_candidates(IMAGES, IMAGE_IDS, QUERIES, top_k=5)

    assert candidates.candidate_ids.shape == (3, 5)
    assert np.array_equal(candidates.candidate_ids, expected_clip_order(5))
    assert np.all(np.diff(candidates.clip_scores, axis=1) <= 0)
    assert candidates.relevant == [set(), set(), set()]


def test_clip_candidates_reject_top_k_beyond_gallery():
    with pytest.raises(ValueError, match="larger than"):
        clip_candidates(IMAGES, IMAGE_IDS, QUERIES, top_k=NUM_IMAGES + 1)


def test_reranked_ids_follow_final_score_and_keep_clip_order_on_ties():
    candidates = Candidates(
        QUERIES[:2],
        np.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
        np.array([[0.9, 0.8, 0.7, 0.6], [0.9, 0.8, 0.7, 0.6]], dtype=np.float32),
        [set(), set()],
    )
    final = np.array([[0.1, 0.5, 0.5, 0.3], [0.2, 0.2, 0.2, 0.2]])

    result = reranked_ids(candidates, final)

    # Row 0: 2 and 3 tie at 0.5 and keep CLIP order; row 1 is all ties, so unchanged.
    assert result.tolist() == [[2, 3, 4, 1], [5, 6, 7, 8]]


def test_reranked_ids_with_alpha_one_is_clip_order():
    candidates = clip_candidates(IMAGES, IMAGE_IDS, QUERIES, top_k=6)
    from mmxai.rerank.fusion import fuse_scores

    graph = np.random.default_rng(2).normal(size=candidates.clip_scores.shape)
    final = fuse_scores(candidates.clip_scores, graph, alpha=1.0)

    assert np.array_equal(reranked_ids(candidates, final), candidates.candidate_ids)


def tiny_graph() -> tuple[ContainsGraph, SplitGraph]:
    concept_x = unit_rows(3, seed=5)
    graph = ContainsGraph(
        image_ids=IMAGE_IDS,
        image_split=np.array(["test"] * NUM_IMAGES),
        concept_keys=np.array(["a.n.01", "b.n.01", "c.n.01"]),
        edge_image=np.arange(NUM_IMAGES, dtype=np.int64),
        edge_concept=(np.arange(NUM_IMAGES) % 3).astype(np.int64),
    )
    x = torch.from_numpy(np.concatenate([IMAGES, concept_x]))
    edges = np.array([NUM_IMAGES + (np.arange(NUM_IMAGES) % 3), np.arange(NUM_IMAGES)])
    edge_index = torch.from_numpy(edges)
    return graph, SplitGraph(IMAGE_IDS, x, edge_index, None)


def test_graph_rerank_ranker_with_untrained_model_returns_clip_order():
    _, split_graph = tiny_graph()
    # Untrained: W_self = I and W_nb = 0, so h equals the image embedding.
    h = image_vectors(ConceptToImageSage(DIM), split_graph)
    assert np.allclose(h, IMAGES, atol=1e-6)
    ranker = GraphRerankRanker(h, IMAGE_IDS, IMAGES, IMAGE_IDS, alpha=0.5, top_k=5)

    ranked = ranker.rank(["q1", "q2", "q3"], QUERIES, 5)

    assert np.array_equal(ranked, expected_clip_order(5))


def test_pcst_sage_ranker_with_untrained_model_returns_clip_order(monkeypatch):
    graph, split_graph = tiny_graph()
    inputs = SimpleNamespace(graph=graph, concept_features=split_graph.x[NUM_IMAGES:].numpy())
    config = SimpleNamespace(score_chunk=2)

    def fake_subgraph_rows(config, inputs, pairs, queries):
        # Every query keeps concepts 0 and 1; PCST itself is not under test here.
        return pack_subgraphs([np.array([0, 1], dtype=np.int64) for _ in queries])

    monkeypatch.setattr(module, "compute_subgraph_rows", fake_subgraph_rows)
    ranker = PcstSageRanker(
        config,
        inputs,
        np.zeros((0, 2), dtype=np.int64),
        split_graph,
        SubgraphSage(DIM),
        0.5,
        IMAGES,
        IMAGE_IDS,
    )

    ranked = ranker.rank(["q1", "q2", "q3"], QUERIES, 5)

    assert np.array_equal(ranked, expected_clip_order(5))


def make_config(tmp_path, **changes) -> ModelBenchmarkEvalConfig:
    values = {
        "experiment": "demo_kg_query_benchmark",
        "model_kind": "pcst_sage",
        "model_config": "configs/clip_pcst_sage_vg.yaml",
        "benchmark_path": "queries/visual_genome_kg_test.jsonl",
        "model": "ViT-B-32",
        "pretrained": "laion2b_s34b_b79k",
        "device": "cpu",
        "text_batch_size": 4,
        "recall_ks": [1, 5],
        "output_dir": str(tmp_path / "runs"),
    }
    values.update(changes)
    return ModelBenchmarkEvalConfig(**values)


def test_model_kind_other_than_known_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "experiment: x\nmodel_kind: gat\nmodel_config: a.yaml\nbenchmark_path: b.jsonl\n"
        "model: m\npretrained: p\ndevice: cpu\ntext_batch_size: 1\nrecall_ks: [1]\n"
        "output_dir: out\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="model_kind"):
        load_model_benchmark_eval_config(path)


def test_second_run_of_same_experiment_and_seed_is_refused(tmp_path):
    config = make_config(tmp_path)
    run_dir = tmp_path / "runs" / "20261121-1200_demo_kg_query_benchmark_seed0"
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text("experiment: demo_kg_query_benchmark\n", encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps({"split": "test"}), encoding="utf-8")

    assert has_benchmark_run(config, 0)
    assert not has_benchmark_run(config, 1)
    with pytest.raises(FileExistsError):
        run_model_benchmark_evaluation(config, 0)


def test_selection_made_on_another_split_is_refused(tmp_path, monkeypatch):
    split_file = tmp_path / "split.json"
    split_file.write_text("{}", encoding="utf-8")
    model_dir = tmp_path / "models"
    (model_dir / "seed0").mkdir(parents=True)
    selection = {"alpha": 0.5, "best_epoch": 3, "git": "abc", "split_sha256": "not-this-file"}
    (model_dir / "seed0" / "selection.json").write_text(json.dumps(selection), encoding="utf-8")
    model_config = SimpleNamespace(split_file=str(split_file), model_dir=str(model_dir))

    monkeypatch.setattr(module, "load_model_config", lambda config: model_config)
    monkeypatch.setattr(module, "load_split", lambda path: {"test": np.array([1, 2])})
    monkeypatch.setattr(module, "load_verified_benchmark", lambda path, ids: [])

    with pytest.raises(ValueError, match="different split file"):
        run_model_benchmark_evaluation(make_config(tmp_path), 0)

    # The same check accepts the matching hash.
    selection["split_sha256"] = file_sha256(split_file)
    module.check_selection_split(selection, str(split_file))
