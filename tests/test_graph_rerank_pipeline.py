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
from mmxai.graph.cooccurrence import (  # noqa: E402
    ConceptEdges,
    load_concept_edges,
    save_concept_edges,
)
from mmxai.graph.shuffle import ShuffleGraphConfig, run_shuffle_graph  # noqa: E402
from mmxai.utils.runs import file_sha256  # noqa: E402

DIM = 8
IMAGE_IDS = np.arange(1, 13)  # 1-6 train, 7-9 val, 10-12 test
SPLIT = {"train": IMAGE_IDS[:6], "val": IMAGE_IDS[6:9], "test": IMAGE_IDS[9:]}
CONCEPTS = ["cat.n.01", "dog.n.01", "sofa.n.01", "tree.n.01"]


def unit_rows(matrix: np.ndarray) -> np.ndarray:
    return (matrix / np.linalg.norm(matrix, axis=1, keepdims=True)).astype(np.float32)


def write_inputs(tmp_path, monkeypatch, dataset: str = "visual_genome") -> GraphRerankConfig:
    """Tiny features, split and graph. flickr30k stores caption positions 0, 1 per image."""
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
        text_id_name="caption_index" if dataset == "flickr30k" else "text_ids",
        text_id_values=np.tile([0, 1], 12) if dataset == "flickr30k" else np.arange(24),
        meta={
            "dataset": dataset,
            "is_partial": False,
            "code_hash": "abc",
            "model": "ViT-B-32",
            "pretrained": "laion2b_s34b_b79k",
        },
    )
    split_file = tmp_path / "split_seed0.json"
    split_json = {"dataset": dataset, "seed": 0}
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
        dataset=dataset,
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


@pytest.mark.parametrize("base", ["clip_graph_vg", "clip_nograph_vg", "clip_cooc_graph_vg"])
def test_lower_lr_configs_differ_only_in_the_learning_rate(base):
    original = load_graph_rerank_config(f"configs/{base}.yaml")
    lower = load_graph_rerank_config(f"configs/{base}_lr1e4.yaml")

    differences = {
        name for name in original.__dict__ if original.__dict__[name] != lower.__dict__[name]
    }
    assert differences == {"experiment", "model_dir", "learning_rate"}
    assert lower.learning_rate == original.learning_rate / 10


def test_shuffled_graph_ablation_trains_on_val_and_tests_once(tmp_path, monkeypatch):
    config = write_inputs(tmp_path, monkeypatch)
    shuffled_dir = tmp_path / "graph_shuffled"
    run_shuffle_graph(ShuffleGraphConfig(config.graph_dir, 0, str(shuffled_dir)))
    config = dataclasses.replace(
        config,
        experiment="clip_graph_shuffled_tiny",
        graph_dir=str(shuffled_dir),
        model_dir=str(tmp_path / "shuffled_models"),
    )

    prepared = prepare(config, log=str)
    val_metrics = read_metrics(train_and_select(config, prepared, seed=0, log=str))
    test_metrics = read_metrics(evaluate_test(config, seed=0, log=str))

    assert test_metrics["alpha"] == val_metrics["alpha"]
    assert test_metrics["graph_edge_source"] == "clip_topk_shuffled"
    with pytest.raises(FileExistsError, match="already run"):
        evaluate_test(config, seed=0, log=str)


def test_shuffled_ablation_config_differs_from_the_main_config_only_in_the_graph():
    main = load_graph_rerank_config("configs/clip_graph_vg_lr1e4.yaml")
    shuffled = load_graph_rerank_config("configs/clip_graph_vg_lr1e4_shuffled.yaml")

    differences = {name for name in main.__dict__ if main.__dict__[name] != shuffled.__dict__[name]}
    assert differences == {"experiment", "graph_dir", "model_dir"}


def with_extensions(config: GraphRerankConfig, tmp_path) -> GraphRerankConfig:
    """The same experiment with nDCG, query groups and image -> text (spec 2026-11-13)."""
    predicates_file = tmp_path / "predicates.yaml"
    predicates_file.write_text(
        "predicate_templates:\n  holding: is holding\n"
        "predicate_aliases:\n  holds: holding\n"
        "action_predicates: [holding]\n",
        encoding="utf-8",
    )
    # Text row r of write_inputs has region id r and belongs to image IMAGE_IDS[r // 2].
    regions = []
    for image_index, image_id in enumerate(IMAGE_IDS.tolist()):
        phrases = ["man holding cup", "red car"]
        regions.append(
            {
                "image_id": image_id,
                "regions": [
                    {"region_id": 2 * image_index + k, "image_id": image_id, "phrase": phrase}
                    for k, phrase in enumerate(phrases)
                ],
            }
        )
    vg_dir = tmp_path / "vg"
    vg_dir.mkdir()
    (vg_dir / "region_descriptions.json").write_text(json.dumps(regions), encoding="utf-8")
    return dataclasses.replace(
        config,
        experiment=f"{config.experiment}_eval2",
        model_dir=f"{config.model_dir}_eval2",
        ndcg_k=3,
        image_to_text=True,
        vg_data_dir=str(vg_dir),
        predicates_file=str(predicates_file),
    )


def read_selection(config: GraphRerankConfig, seed: int) -> dict:
    with open(Path(config.model_dir) / f"seed{seed}" / "selection.json", encoding="utf-8") as file:
        return json.load(file)


RUN_FACT_KEYS = {
    "split",
    "split_file",
    "split_sha256",
    "seed",
    "alpha",
    "best_epoch",
    "use_graph",
    "graph_dir",
    "concept_edges_dir",
    "graph_edge_source",
    "graph_git_commit",
    "features_code_hash",
}


def test_old_graph_config_keeps_selection_and_metrics_keys(tmp_path, monkeypatch):
    config = write_inputs(tmp_path, monkeypatch)
    prepared = prepare(config, log=str)

    val_metrics = read_metrics(train_and_select(config, prepared, seed=0, log=str))
    test_metrics = read_metrics(evaluate_test(config, seed=0, log=str))

    assert prepared.val_text_candidates is None
    assert set(read_selection(config, 0)) == {
        "alpha",
        "best_epoch",
        "history",
        "alpha_sweep",
        "git",
    }
    metric_keys = {"R@1", "R@3", "MRR@3", "num_queries"}
    assert set(val_metrics) == metric_keys | RUN_FACT_KEYS | {"alpha_sweep", "history"}
    assert set(test_metrics) == metric_keys | RUN_FACT_KEYS | {"model_sha256", "model_git_commit"}


def test_image_to_text_alpha_is_chosen_on_val_and_reused_on_test(tmp_path, monkeypatch):
    config = with_extensions(write_inputs(tmp_path, monkeypatch), tmp_path)
    prepared = prepare(config, log=str)

    train_and_select(config, prepared, seed=0, log=str)
    test_metrics = read_metrics(evaluate_test(config, seed=0, log=str))

    selection = read_selection(config, 0)
    sweep = selection["alpha_sweep_image_to_text"]
    assert set(sweep) == {str(alpha) for alpha in config.alpha_grid}
    best = max(values["MRR@3"] for values in sweep.values())
    assert sweep[str(selection["alpha_image_to_text"])]["MRR@3"] == best
    assert test_metrics["image_to_text"]["alpha"] == selection["alpha_image_to_text"]
    assert test_metrics["alpha"] == selection["alpha"]


def test_eval2_test_metrics_have_ndcg_groups_and_image_to_text(tmp_path, monkeypatch):
    config = with_extensions(write_inputs(tmp_path, monkeypatch), tmp_path)
    train_and_select(config, prepare(config, log=str), seed=0, log=str)

    metrics = read_metrics(evaluate_test(config, seed=0, log=str))

    assert "nDCG@3" in metrics
    assert set(metrics["groups"]) == {"specificity", "relation"}
    # Both phrases occur once per image: half the queries are "action", half "none".
    assert metrics["groups"]["relation"]["action"]["num_queries"] == 3
    image_to_text = metrics["image_to_text"]
    assert image_to_text["num_queries"] == 3  # the 3 test images
    assert {"R@1", "R@3", "MRR@3", "nDCG@3", "alpha"} <= set(image_to_text)
    assert "alpha_sweep" not in metrics  # nothing is chosen on test


def test_epoch_and_text_to_image_alpha_do_not_change_when_extensions_are_on(tmp_path, monkeypatch):
    config = write_inputs(tmp_path, monkeypatch)
    extended = with_extensions(config, tmp_path)

    old_val = read_metrics(train_and_select(config, prepare(config, log=str), seed=0, log=str))
    new_val = read_metrics(train_and_select(extended, prepare(extended, log=str), seed=0, log=str))

    assert new_val["best_epoch"] == old_val["best_epoch"]
    assert new_val["alpha"] == old_val["alpha"]


def test_cooccurrence_model_reports_image_to_text(tmp_path, monkeypatch):
    config = with_concept_edges(write_inputs(tmp_path, monkeypatch), tmp_path)
    config = with_extensions(config, tmp_path)
    train_and_select(config, prepare(config, log=str), seed=0, log=str)

    metrics = read_metrics(evaluate_test(config, seed=0, log=str))

    assert metrics["image_to_text"]["alpha"] == read_selection(config, 0)["alpha_image_to_text"]


def test_graph_config_with_image_to_text_needs_ndcg(tmp_path):
    config_file = tmp_path / "config.yaml"
    text = Path("configs/clip_graph_vg.yaml").read_text(encoding="utf-8")
    config_file.write_text(text + "\nimage_to_text: true\n", encoding="utf-8")

    with pytest.raises(ValueError, match="ndcg_k"):
        load_graph_rerank_config(config_file)


GRAPH_EVAL2_SOURCES = [
    "clip_nograph_vg",
    "clip_graph_vg_lr1e4",
    "clip_cooc_graph_vg_lr1e4",
    "clip_graph_vg_lr1e4_shuffled",
]
EXTENSION_FIELDS = {
    "ndcg_k": 10,
    "image_to_text": True,
    "vg_data_dir": "data/visual_genome",
    "predicates_file": "configs/query_benchmark_vg.yaml",
    "extra_action_predicates": None,
    "extra_spatial_predicates": None,
    "excluded_predicates": None,
}


@pytest.mark.parametrize("source_name", GRAPH_EVAL2_SOURCES)
def test_graph_eval2_config_differs_from_its_source_only_in_name_model_dir_and_extensions(
    source_name,
):
    source = load_graph_rerank_config(f"configs/{source_name}.yaml")
    eval2 = load_graph_rerank_config(f"configs/{source_name}_eval2.yaml")

    differences = {
        name for name in source.__dict__ if source.__dict__[name] != eval2.__dict__[name]
    }
    assert differences == {"experiment", "model_dir"} | {
        name for name in EXTENSION_FIELDS if EXTENSION_FIELDS[name] is not None
    }
    assert eval2.experiment == f"{source_name}_eval2"
    assert eval2.model_dir == f"artifacts/models/visual_genome/{source_name}_eval2"
    for name, value in EXTENSION_FIELDS.items():
        assert getattr(eval2, name) == value


def test_val_metrics_match_selection_image_to_text_and_old_values(tmp_path, monkeypatch):
    config = write_inputs(tmp_path, monkeypatch)
    extended = with_extensions(config, tmp_path)

    old_val = read_metrics(train_and_select(config, prepare(config, log=str), seed=0, log=str))
    new_val = read_metrics(train_and_select(extended, prepare(extended, log=str), seed=0, log=str))

    for key in ("R@1", "R@3", "MRR@3", "num_queries"):
        assert new_val[key] == old_val[key]

    selection = read_selection(extended, 0)
    alpha_i2t = selection["alpha_image_to_text"]
    assert set(new_val["groups"]) == {"specificity", "relation"}
    assert new_val["image_to_text"]["alpha"] == alpha_i2t
    assert (
        new_val["image_to_text"]["MRR@3"]
        == selection["alpha_sweep_image_to_text"][str(alpha_i2t)]["MRR@3"]
    )


def test_evaluate_test_raises_when_selection_lacks_alpha_image_to_text(tmp_path, monkeypatch):
    config = write_inputs(tmp_path, monkeypatch)
    train_and_select(config, prepare(config, log=str), seed=0, log=str)
    extended = with_extensions(config, tmp_path)
    # Reuse the model/selection trained without image_to_text by copying artifacts.
    old_seed_dir = Path(config.model_dir) / "seed0"
    new_seed_dir = Path(extended.model_dir) / "seed0"
    new_seed_dir.mkdir(parents=True, exist_ok=True)
    for name in ("model.pt", "selection.json"):
        (new_seed_dir / name).write_bytes((old_seed_dir / name).read_bytes())

    with pytest.raises(ValueError, match="alpha_image_to_text"):
        evaluate_test(extended, seed=0, log=str)


def edges_counted_on_split_a(tmp_path, monkeypatch):
    """Config whose concept edges were counted on split A, while the experiment uses split B."""
    config = write_inputs(tmp_path, monkeypatch)
    split_a = tmp_path / "split_a.json"
    split_a.write_text('{"split": "a"}', encoding="utf-8")
    with_edges = with_concept_edges(config, tmp_path)
    edges, _ = load_concept_edges(with_edges.concept_edges_dir)
    edges_dir = tmp_path / "edges_on_split_a"
    save_concept_edges(edges_dir, edges, {"split_sha256": file_sha256(split_a)})
    return dataclasses.replace(with_edges, concept_edges_dir=str(edges_dir)), split_a


def test_concept_edges_counted_on_another_split_file_are_accepted(tmp_path, monkeypatch):
    config, split_a = edges_counted_on_split_a(tmp_path, monkeypatch)
    config = dataclasses.replace(config, concept_edges_split_file=str(split_a))

    assert load_inputs(config).concept_edges is not None


def test_concept_edges_counted_on_a_third_split_file_are_rejected(tmp_path, monkeypatch):
    config, _ = edges_counted_on_split_a(tmp_path, monkeypatch)
    split_c = tmp_path / "split_c.json"
    split_c.write_text('{"split": "c"}', encoding="utf-8")
    config = dataclasses.replace(config, concept_edges_split_file=str(split_c))

    with pytest.raises(ValueError, match="different split"):
        load_inputs(config)
