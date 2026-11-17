"""Graph for a dataset without annotations, reusing another graph's concepts (ADR-0050).

Made-up data: a source graph with concepts cat/dog/sofa along the first three axes;
a new dataset with images 1-2 train, 3 val, 4 test, whose vectors make the nearest
concept obvious by eye.
"""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from mmxai.data.splits import save_split
from mmxai.features.cache import IMAGE_FILE
from mmxai.features.cache import META_FILE as FEATURE_META_FILE
from mmxai.graph.contains import ContainsGraph, load_contains_graph, save_contains_graph
from mmxai.graph.vocabulary_transfer import (
    VocabularyTransferConfig,
    load_vocabulary_transfer_config,
    run_vocabulary_transfer,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL = {"model": "ViT-B-32", "pretrained": "laion2b_s34b_b79k"}
CONCEPT_KEYS = np.array(["cat.n.01", "dog.n.01", "sofa.n.01"])
CONCEPT_FEATURES = np.eye(3, dtype=np.float32)


def make_source_graph(path: Path) -> None:
    graph = ContainsGraph(
        image_ids=np.array([100], dtype=np.int64),
        image_split=np.array(["train"]),
        concept_keys=CONCEPT_KEYS,
        edge_image=np.array([0], dtype=np.int64),
        edge_concept=np.array([0], dtype=np.int64),
    )
    meta = {"dataset": "visual_genome", "config": dict(MODEL)}
    save_contains_graph(path, graph, np.array(["cat", "dog", "sofa"]), CONCEPT_FEATURES, meta)


def make_config(tmp_path: Path) -> VocabularyTransferConfig:
    make_source_graph(tmp_path / "source")
    split_file = tmp_path / "split.json"
    save_split({"train": ["1", "2"], "val": ["3"], "test": ["4"]}, split_file)
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    np.savez(
        features_dir / IMAGE_FILE,
        image_ids=np.array([4, 3, 2, 1]),  # not in id order: rows must be found by id
        embeddings=np.array([[0, 0, 1], [0, 1, 0.2], [0.9, 0.1, 0], [1, 0, 0.3]], dtype=np.float32),
    )
    (features_dir / FEATURE_META_FILE).write_text(json.dumps(MODEL), encoding="utf-8")
    return VocabularyTransferConfig(
        dataset="flickr30k",
        split_file=str(split_file),
        features_dir=str(features_dir),
        vocabulary_graph_dir=str(tmp_path / "source"),
        clip_topk_k=1,
        model=MODEL["model"],
        pretrained=MODEL["pretrained"],
        output_dir=str(tmp_path / "graph"),
    )


def concepts_by_image(graph: ContainsGraph) -> dict[int, set[str]]:
    result: dict[int, set[str]] = {}
    for image_row, concept_row in zip(graph.edge_image, graph.edge_concept, strict=True):
        image_id = int(graph.image_ids[image_row])
        result.setdefault(image_id, set()).add(str(graph.concept_keys[concept_row]))
    return result


def test_concept_nodes_are_copied_from_the_source_graph(tmp_path):
    config = make_config(tmp_path)
    run_vocabulary_transfer(config)

    graph, names, features, _ = load_contains_graph(config.output_dir)

    assert graph.concept_keys.tolist() == CONCEPT_KEYS.tolist()
    assert names.tolist() == ["cat", "dog", "sofa"]
    np.testing.assert_array_equal(features, CONCEPT_FEATURES)


def test_every_split_image_links_to_its_nearest_concepts(tmp_path):
    config = make_config(tmp_path)
    run_vocabulary_transfer(config)

    graph, _, _, _ = load_contains_graph(config.output_dir)

    assert concepts_by_image(graph) == {
        1: {"cat.n.01"},
        2: {"cat.n.01"},
        3: {"dog.n.01"},
        4: {"sofa.n.01"},
    }
    assert dict(zip(graph.image_ids.tolist(), graph.image_split.tolist(), strict=True)) == {
        1: "train",
        2: "train",
        3: "val",
        4: "test",
    }


def test_source_graph_images_are_not_copied(tmp_path):
    config = make_config(tmp_path)
    run_vocabulary_transfer(config)

    graph, _, _, _ = load_contains_graph(config.output_dir)

    assert 100 not in graph.image_ids.tolist()


def test_meta_records_where_the_vocabulary_came_from(tmp_path):
    config = make_config(tmp_path)

    meta = run_vocabulary_transfer(config)

    assert meta["dataset"] == "flickr30k"
    assert meta["vocabulary_from"]["dataset"] == "visual_genome"
    assert meta["num_concepts"] == 3
    assert meta["num_edges"] == 4
    assert meta["config"]["model"] == MODEL["model"]


def test_image_cache_from_another_model_is_rejected(tmp_path):
    config = dataclasses.replace(make_config(tmp_path), pretrained="openai")

    with pytest.raises(ValueError, match="openai"):
        run_vocabulary_transfer(config)


def test_existing_output_is_never_overwritten(tmp_path):
    config = make_config(tmp_path)
    run_vocabulary_transfer(config)

    with pytest.raises(FileExistsError):
        run_vocabulary_transfer(config)


def test_committed_config_loads():
    config = load_vocabulary_transfer_config(
        REPO_ROOT / "configs" / "graph_flickr30k_clip_topk.yaml"
    )

    assert config.dataset == "flickr30k"
    assert config.clip_topk_k == 16
