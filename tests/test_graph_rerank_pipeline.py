"""CLIP + Graph end to end on tiny files: train on train, choose on val, test once."""

import dataclasses
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch_geometric")
pytest.importorskip("faiss")

from mmxai.data.splits import save_split  # noqa: E402
from mmxai.eval import graph_rerank  # noqa: E402
from mmxai.eval.graph_rerank import (  # noqa: E402
    GraphRerankConfig,
    build_split_graph,
    evaluate_test,
    graph_only_val_score,
    load_graph_rerank_config,
    load_inputs,
    make_model,
    prepare,
    train_and_select,
)
from mmxai.features.cache import save_features  # noqa: E402
from mmxai.graph.clip_edges import clip_topk_concepts  # noqa: E402
from mmxai.graph.contains import build_contains_graph, save_contains_graph  # noqa: E402
from mmxai.graph.cooccurrence import ConceptEdges, save_concept_edges  # noqa: E402
from mmxai.utils.runs import file_sha256  # noqa: E402

DIM = 8
IMAGE_IDS = np.arange(1, 13)  # 1-6 train, 7-9 val, 10-12 test
SPLIT = {"train": IMAGE_IDS[:6], "val": IMAGE_IDS[6:9], "test": IMAGE_IDS[9:]}
CONCEPTS = ["cat.n.01", "dog.n.01", "sofa.n.01", "tree.n.01"]


def unit_rows(matrix: np.ndarray) -> np.ndarray:
    return (matrix / np.linalg.norm(matrix, axis=1, keepdims=True)).astype(np.float32)


def write_inputs(tmp_path, monkeypatch) -> GraphRerankConfig:
    generator = np.random.default_rng(0)
    images = unit_rows(generator.normal(size=(12, DIM)))
    texts = unit_rows(np.repeat(images, 2, axis=0) + generator.normal(scale=0.5, size=(24, DIM)))
    concept_features = unit_rows(generator.normal(size=(4, DIM)))

    features_dir = tmp_path / "features"
    save_features(
        features_dir,
        image_embeddings=images,
        image_ids=IMAGE_IDS,
        text_embeddings=texts,
        text_image_ids=np.repeat(IMAGE_IDS, 2),
        text_id_name="text_ids",
        text_id_values=np.arange(24),
        meta={
            "dataset": "visual_genome",
            "is_partial": False,
            "code_hash": "abc",
            "model": "ViT-B-32",
            "pretrained": "laion2b_s34b_b79k",
        },
    )
    split_file = tmp_path / "split_seed0.json"
    split_json = {"dataset": "visual_genome", "seed": 0}
    for name, ids in SPLIT.items():
        split_json[name] = [str(image_id) for image_id in ids.tolist()]
    save_split(split_json, split_file)

    image_concepts = clip_topk_concepts(IMAGE_IDS, images, CONCEPTS, concept_features, k=2)
    graph = build_contains_graph(image_concepts, SPLIT, CONCEPTS)
    graph_dir = tmp_path / "graph"
    save_contains_graph(
        graph_dir,
        graph,
        concept_names=np.array(CONCEPTS),
        concept_features=concept_features,
        meta={
            "edge_source": "clip_topk",
            "split_sha256": file_sha256(split_file),
            "git": {"git_commit": "abc"},
            "config": {"model": "ViT-B-32", "pretrained": "laion2b_s34b_b79k"},
        },
    )

    # Runs are named by the minute; give each new run a later minute.
    clock = {"now": datetime(2026, 11, 12, 9, 0)}

    class FakeDatetime:
        @staticmethod
        def now():
            clock["now"] += timedelta(minutes=1)
            return clock["now"]

    monkeypatch.setattr(graph_rerank, "datetime", FakeDatetime)

    return GraphRerankConfig(
        experiment="clip_graph_tiny",
        dataset="visual_genome",
        split_file=str(split_file),
        features_dir=str(features_dir),
        graph_dir=str(graph_dir),
        concept_edges_dir=None,
        model_dir=str(tmp_path / "models"),
        use_graph=True,
        top_k=3,
        recall_ks=[1, 3],
        same_text_threshold=0.9999,
        num_hard_negatives=2,
        temperature=0.05,
        learning_rate=0.01,
        batch_size=4,
        max_epochs=3,
        patience=2,
        alpha_grid=[0.0, 0.5, 1.0],
        selection_metric="MRR@3",
        output_dir=str(tmp_path / "runs"),
    )


def read_metrics(run_dir) -> dict:
    with open(run_dir / "metrics.json", encoding="utf-8") as file:
        return json.load(file)


def test_training_never_sees_val_or_test_images(tmp_path, monkeypatch):
    config = write_inputs(tmp_path, monkeypatch)

    prepared = prepare(config, log=str)

    data = prepared.training_data
    assert len(data.image_x) == 6  # train images only
    # At most the 2 texts of each train image (some are left out by the ADR-0047 filter).
    assert 0 < len(data.text_x) <= 12
    assert int(data.candidate_rows.max()) < 6  # hard negatives are train rows


def test_every_training_text_has_its_image_among_its_candidates(tmp_path, monkeypatch):
    config = write_inputs(tmp_path, monkeypatch)

    data = prepare(config, log=str).training_data

    in_candidates = (data.candidate_rows == data.text_image_row[:, None]).any(dim=1)
    assert in_candidates.all()


def test_split_graph_holds_only_concept_to_image_edges_of_that_split(tmp_path, monkeypatch):
    config = write_inputs(tmp_path, monkeypatch)
    inputs = load_inputs(config)

    test_graph = build_split_graph(inputs, "test")

    assert test_graph.image_ids.tolist() == [10, 11, 12]
    num_images = len(test_graph.image_ids)
    assert (test_graph.edge_index[0] >= num_images).all()  # sources are concepts
    assert (test_graph.edge_index[1] < num_images).all()  # targets are images
    assert test_graph.edge_index.shape[1] == 3 * 2  # k = 2 concepts per image


def test_val_run_records_the_alpha_chosen_on_val(tmp_path, monkeypatch):
    config = write_inputs(tmp_path, monkeypatch)
    prepared = prepare(config, log=str)

    metrics = read_metrics(train_and_select(config, prepared, seed=0, log=str))

    assert metrics["split"] == "val"
    assert metrics["alpha"] in config.alpha_grid
    best = max(values["MRR@3"] for values in metrics["alpha_sweep"].values())
    assert metrics["MRR@3"] == best


def test_test_uses_the_saved_alpha_and_runs_only_once(tmp_path, monkeypatch):
    config = write_inputs(tmp_path, monkeypatch)
    prepared = prepare(config, log=str)
    val_metrics = read_metrics(train_and_select(config, prepared, seed=0, log=str))

    test_metrics = read_metrics(evaluate_test(config, seed=0, log=str))

    assert test_metrics["split"] == "test"
    assert test_metrics["alpha"] == val_metrics["alpha"]
    assert "alpha_sweep" not in test_metrics  # nothing is chosen on test
    with pytest.raises(FileExistsError, match="already"):
        evaluate_test(config, seed=0, log=str)


def test_training_a_seed_twice_is_refused(tmp_path, monkeypatch):
    config = write_inputs(tmp_path, monkeypatch)
    prepared = prepare(config, log=str)
    train_and_select(config, prepared, seed=0, log=str)

    with pytest.raises(FileExistsError, match="seed0"):
        train_and_select(config, prepared, seed=0, log=str)


def test_graph_built_on_another_split_is_rejected(tmp_path, monkeypatch):
    config = write_inputs(tmp_path, monkeypatch)
    other_split = tmp_path / "other_split.json"
    save_split(
        {"dataset": "visual_genome", "seed": 1, "train": ["1"], "val": [], "test": []}, other_split
    )

    with pytest.raises(ValueError, match="different split"):
        load_inputs(GraphRerankConfig(**{**config.__dict__, "split_file": str(other_split)}))


def test_committed_configs_load_and_differ_only_in_the_graph_switch():
    graph = load_graph_rerank_config("configs/clip_graph_vg.yaml")
    no_graph = load_graph_rerank_config("configs/clip_nograph_vg.yaml")

    assert graph.use_graph and not no_graph.use_graph
    differences = {
        name for name in graph.__dict__ if graph.__dict__[name] != no_graph.__dict__[name]
    }
    assert differences == {"experiment", "model_dir", "use_graph"}


def with_concept_edges(config: GraphRerankConfig, tmp_path, concept_keys=CONCEPTS):
    """The same experiment with co-occurrence edges (ADR-0048): dog <-> sofa, cat -> tree."""
    edges = ConceptEdges(
        concept_keys=np.array(concept_keys),
        edge_source=np.array([1, 2, 0]),
        edge_target=np.array([2, 1, 3]),
        edge_pmi=np.array([1.0, 1.0, 0.5]),
    )
    edges_dir = tmp_path / "concept_edges"
    save_concept_edges(edges_dir, edges, {"split_sha256": file_sha256(config.split_file)})
    return dataclasses.replace(
        config,
        experiment="clip_cooc_tiny",
        concept_edges_dir=str(edges_dir),
        model_dir=str(tmp_path / "cooc_models"),
    )


def test_cooccurrence_model_trains_on_val_and_tests_once(tmp_path, monkeypatch):
    config = with_concept_edges(write_inputs(tmp_path, monkeypatch), tmp_path)
    prepared = prepare(config, log=str)

    assert prepared.training_data.second_hop_mean is not None
    val_metrics = read_metrics(train_and_select(config, prepared, seed=0, log=str))
    test_metrics = read_metrics(evaluate_test(config, seed=0, log=str))

    assert test_metrics["alpha"] == val_metrics["alpha"]
    assert test_metrics["concept_edges_dir"] == config.concept_edges_dir


def test_untrained_cooccurrence_model_gives_the_plain_clip_val_score(tmp_path, monkeypatch):
    config = write_inputs(tmp_path, monkeypatch)
    cooc_config = with_concept_edges(config, tmp_path)

    plain = graph_only_val_score(config, make_model(config, DIM), prepare(config, log=str))
    cooc = graph_only_val_score(
        cooc_config, make_model(cooc_config, DIM), prepare(cooc_config, log=str)
    )

    assert cooc == plain


def test_concept_edges_of_other_concepts_are_rejected(tmp_path, monkeypatch):
    config = write_inputs(tmp_path, monkeypatch)
    renamed = ["a.n.01", "b.n.01", "c.n.01", "d.n.01"]
    config = with_concept_edges(config, tmp_path, concept_keys=renamed)

    with pytest.raises(ValueError, match="concepts of the graph"):
        load_inputs(config)


def test_concept_edges_without_the_graph_are_rejected(tmp_path):
    config_file = tmp_path / "config.yaml"
    text = Path("configs/clip_cooc_graph_vg.yaml").read_text(encoding="utf-8")
    config_file.write_text(text.replace("use_graph: true", "use_graph: false"), encoding="utf-8")

    with pytest.raises(ValueError, match="use_graph"):
        load_graph_rerank_config(config_file)


def test_cooccurrence_config_differs_from_clip_graph_only_in_the_concept_edges():
    graph = load_graph_rerank_config("configs/clip_graph_vg.yaml")
    cooc = load_graph_rerank_config("configs/clip_cooc_graph_vg.yaml")

    differences = {name for name in graph.__dict__ if graph.__dict__[name] != cooc.__dict__[name]}
    assert differences == {"experiment", "model_dir", "concept_edges_dir"}
