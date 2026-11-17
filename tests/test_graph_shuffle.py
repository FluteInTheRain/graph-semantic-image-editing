"""Degree-preserving edge shuffle of the contains graph (ADR-0046 section 6) on synthetic graphs.

Every graph here is made up in the test: 3 splits, a few images each, every image
linked to a few distinct concepts drawn with a fixed seed.
"""

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from mmxai.graph.contains import (
    ContainsGraph,
    load_contains_graph,
    save_contains_graph,
    split_subgraph,
)
from mmxai.graph.pipeline import load_graph_config
from mmxai.graph.shuffle import (
    ShuffleGraphConfig,
    load_shuffle_graph_config,
    run_shuffle_graph,
    shuffle_concept_pairs,
    shuffle_contains_edges,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def make_graph(images_per_split: int, num_concepts: int, degree: int, seed: int) -> ContainsGraph:
    """Random graph: every image links to `degree` distinct concepts, no duplicate edge."""
    rng = np.random.default_rng(seed)
    image_split = []
    for split_name in ("train", "val", "test"):
        image_split.extend([split_name] * images_per_split)
    edge_image = []
    edge_concept = []
    for image_row in range(len(image_split)):
        concepts = rng.choice(num_concepts, size=degree, replace=False)
        for concept in sorted(concepts.tolist()):
            edge_image.append(image_row)
            edge_concept.append(concept)
    return ContainsGraph(
        image_ids=np.arange(100, 100 + len(image_split), dtype=np.int64),
        image_split=np.array(image_split),
        concept_keys=np.array([f"concept{row}.n.01" for row in range(num_concepts)]),
        edge_image=np.array(edge_image, dtype=np.int64),
        edge_concept=np.array(edge_concept, dtype=np.int64),
    )


# Dense on purpose: 6 of 8 concepts per image, so a plain permutation almost surely
# gives some image a concept twice and the repair step has work to do.
DENSE = make_graph(images_per_split=5, num_concepts=8, degree=6, seed=0)
# Sparse and bigger: a shuffle that leaves every edge in place is essentially impossible.
LARGE = make_graph(images_per_split=30, num_concepts=40, degree=5, seed=1)


def edge_pairs(graph: ContainsGraph) -> list[tuple[int, int]]:
    return list(zip(graph.edge_image.tolist(), graph.edge_concept.tolist(), strict=True))


def test_every_image_keeps_its_number_of_edges():
    shuffled = shuffle_contains_edges(DENSE, seed=0)

    assert Counter(shuffled.edge_image.tolist()) == Counter(DENSE.edge_image.tolist())


@pytest.mark.parametrize("split_name", ["train", "val", "test"])
def test_every_concept_keeps_its_number_of_edges_inside_each_split(split_name):
    original = split_subgraph(DENSE, split_name)
    shuffled = split_subgraph(shuffle_contains_edges(DENSE, seed=0), split_name)

    assert Counter(shuffled.edge_concept.tolist()) == Counter(original.edge_concept.tolist())


def test_no_image_concept_pair_appears_twice():
    shuffled = shuffle_contains_edges(DENSE, seed=0)

    pairs = edge_pairs(shuffled)
    assert len(set(pairs)) == len(pairs)


def test_no_edge_links_an_image_to_a_concept_of_another_split():
    # Concepts 0-3 only in train, 4-5 only in val, 6-7 only in test: if an edge moved
    # across splits, some image would get a concept that its split never had.
    graph = ContainsGraph(
        image_ids=np.array([1, 2, 3, 4, 5, 6]),
        image_split=np.array(["train", "train", "val", "val", "test", "test"]),
        concept_keys=np.array([f"c{row}.n.01" for row in range(8)]),
        edge_image=np.array([0, 0, 1, 1, 2, 3, 4, 5]),
        edge_concept=np.array([0, 1, 2, 3, 4, 5, 6, 7]),
    )
    allowed = {"train": {0, 1, 2, 3}, "val": {4, 5}, "test": {6, 7}}

    shuffled = shuffle_contains_edges(graph, seed=0)

    for image_row, concept_row in edge_pairs(shuffled):
        assert concept_row in allowed[str(graph.image_split[image_row])]


def test_nodes_and_splits_are_unchanged():
    shuffled = shuffle_contains_edges(DENSE, seed=0)

    assert np.array_equal(shuffled.image_ids, DENSE.image_ids)
    assert np.array_equal(shuffled.image_split, DENSE.image_split)
    assert np.array_equal(shuffled.concept_keys, DENSE.concept_keys)


def test_same_seed_gives_the_same_graph():
    first = shuffle_contains_edges(LARGE, seed=3)
    second = shuffle_contains_edges(LARGE, seed=3)

    assert edge_pairs(first) == edge_pairs(second)


def test_different_seeds_give_different_graphs():
    first = shuffle_contains_edges(LARGE, seed=3)
    second = shuffle_contains_edges(LARGE, seed=4)

    assert edge_pairs(first) != edge_pairs(second)


def test_most_edges_are_rewired():
    shuffled = shuffle_contains_edges(LARGE, seed=0)

    kept = set(edge_pairs(shuffled)) & set(edge_pairs(LARGE))
    # 5 of 40 concepts per image: a random edge matches an old one ~1/8 of the time.
    assert len(kept) < len(LARGE.edge_image) / 2


def test_a_duplicate_that_cannot_be_removed_raises():
    # Invalid on purpose: one image already holds concept 0 twice and there is no
    # other edge to swap with, so no repair can exist.
    graph = ContainsGraph(
        image_ids=np.array([1]),
        image_split=np.array(["train"]),
        concept_keys=np.array(["c0.n.01"]),
        edge_image=np.array([0, 0]),
        edge_concept=np.array([0, 0]),
    )

    with pytest.raises(RuntimeError, match="duplicate edge"):
        shuffle_contains_edges(graph, seed=0, max_tries_per_edge=10)


def save_source_graph(graph_dir) -> None:
    num_concepts = len(DENSE.concept_keys)
    meta = {"edge_source": "clip_topk", "split_sha256": "abc123", "git": {"git_commit": "c0"}}
    names = np.array([f"name {row}" for row in range(num_concepts)])
    features = np.arange(num_concepts * 3, dtype=np.float32).reshape(num_concepts, 3)
    save_contains_graph(graph_dir, DENSE, names, features, meta)


def test_saved_shuffled_graph_loads_with_the_graph_loader(tmp_path):
    save_source_graph(tmp_path / "source")
    config = ShuffleGraphConfig(str(tmp_path / "source"), 0, str(tmp_path / "shuffled"))

    run_shuffle_graph(config)

    graph, names, features, meta = load_contains_graph(tmp_path / "shuffled")
    _, source_names, source_features, _ = load_contains_graph(tmp_path / "source")
    assert edge_pairs(graph) == edge_pairs(shuffle_contains_edges(DENSE, seed=0))
    assert np.array_equal(names, source_names)
    assert np.array_equal(features, source_features)


def test_shuffled_meta_keeps_the_split_hash_and_records_the_shuffle(tmp_path):
    save_source_graph(tmp_path / "source")
    config = ShuffleGraphConfig(str(tmp_path / "source"), 7, str(tmp_path / "shuffled"))

    run_shuffle_graph(config)

    _, _, _, meta = load_contains_graph(tmp_path / "shuffled")
    assert meta["split_sha256"] == "abc123"
    assert meta["edge_source"] == "clip_topk_shuffled"
    assert meta["shuffle"]["seed"] == 7
    assert meta["shuffle"]["source_graph_dir"] == str(tmp_path / "source")
    assert meta["shuffle"]["source_git"] == {"git_commit": "c0"}


def test_shuffle_refuses_to_overwrite_an_existing_folder(tmp_path):
    save_source_graph(tmp_path / "source")
    (tmp_path / "shuffled").mkdir()
    config = ShuffleGraphConfig(str(tmp_path / "source"), 0, str(tmp_path / "shuffled"))

    with pytest.raises(FileExistsError):
        run_shuffle_graph(config)


def test_committed_config_shuffles_the_graph_of_the_main_model():
    main_graph = load_graph_config(REPO_ROOT / "configs" / "graph_vg_clip_topk.yaml")
    config = load_shuffle_graph_config(REPO_ROOT / "configs" / "graph_vg_clip_topk_shuffled.yaml")

    assert config.graph_dir == main_graph.output_dir
    assert config.output_dir != config.graph_dir


def test_shuffle_concept_pairs_keeps_degrees_and_changes_edges():
    pairs = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [4, 6]])

    shuffled = shuffle_concept_pairs(pairs, seed=0)

    def degrees(edge_pairs):
        return np.bincount(edge_pairs.ravel(), minlength=8).tolist()

    assert degrees(shuffled) == degrees(pairs)
    assert {tuple(p) for p in shuffled.tolist()} != {tuple(p) for p in pairs.tolist()}
    assert shuffle_concept_pairs(pairs, seed=0).tolist() == shuffled.tolist()  # deterministic
