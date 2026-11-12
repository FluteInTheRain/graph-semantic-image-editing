"""Concept -> concept co-occurrence edges (ADR-0048), counted on train annotations only."""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from mmxai.data.splits import save_split
from mmxai.graph.contains import build_contains_graph, save_contains_graph
from mmxai.graph.cooccurrence import (
    ConceptEdgeConfig,
    count_cooccurrence,
    load_concept_edges,
    run_build_concept_edges,
    top_pmi_edges,
)
from mmxai.utils.runs import file_sha256

CONCEPTS = ["cat.n.01", "dog.n.01", "leash.n.01", "sofa.n.01"]
CAT, DOG, LEASH, SOFA = range(4)


def test_counts_pairs_and_concepts_per_image():
    image_concepts = {1: {"dog.n.01", "leash.n.01"}, 2: {"dog.n.01"}, 3: {"dog.n.01", "sofa.n.01"}}

    pairs, counts = count_cooccurrence(image_concepts, np.array([1, 2, 3]), CONCEPTS)

    assert counts.tolist() == [0, 3, 1, 1]
    assert pairs[DOG, LEASH] == pairs[LEASH, DOG] == 1
    assert pairs[DOG, SOFA] == 1
    assert np.diag(pairs).tolist() == [0, 0, 0, 0]


def test_statistical_edges_ignore_test_images():
    # Image 9 (test) has cat + sofa; only train ids 1, 2 may be counted (PDF section 6).
    image_concepts = {1: {"dog.n.01", "leash.n.01"}, 2: {"dog.n.01"}, 9: {"cat.n.01", "sofa.n.01"}}

    pairs, counts = count_cooccurrence(image_concepts, np.array([1, 2]), CONCEPTS)

    assert pairs[CAT, SOFA] == 0
    assert counts[CAT] == counts[SOFA] == 0


def test_chunks_give_the_same_counts_as_one_pass():
    generator = np.random.default_rng(0)
    image_concepts = {
        image_id: set(generator.choice(CONCEPTS, size=2, replace=False).tolist())
        for image_id in range(20)
    }
    train_ids = np.arange(20)

    small = count_cooccurrence(image_concepts, train_ids, CONCEPTS, chunk_size=3)
    whole = count_cooccurrence(image_concepts, train_ids, CONCEPTS, chunk_size=100)

    assert np.array_equal(small[0], whole[0]) and np.array_equal(small[1], whole[1])


def pmi_example() -> tuple[np.ndarray, np.ndarray]:
    """100 train images. Dog and leash go together; sofa meets dog only by chance."""
    counts = np.array([10, 40, 20, 50])
    pairs = np.zeros((4, 4), dtype=np.int64)
    pairs[DOG, LEASH] = pairs[LEASH, DOG] = 18  # PMI = log(18*100/(40*20)) > 0
    pairs[DOG, SOFA] = pairs[SOFA, DOG] = 20  # PMI = log(20*100/(40*50)) = 0
    pairs[CAT, SOFA] = pairs[SOFA, CAT] = 3  # PMI > 0 but seen in too few images
    return pairs, counts


def test_only_pairs_above_chance_and_seen_often_enough_are_kept():
    pairs, counts = pmi_example()

    source, target, pmi = top_pmi_edges(pairs, counts, 100, min_pair_images=5, top_m=10)

    assert sorted(zip(source.tolist(), target.tolist(), strict=True)) == [
        (DOG, LEASH),
        (LEASH, DOG),
    ]
    assert np.allclose(pmi, np.log(18 * 100 / (40 * 20)))


def test_each_concept_keeps_at_most_top_m_partners_by_pmi():
    counts = np.array([10, 10, 10, 10])
    pairs = np.zeros((4, 4), dtype=np.int64)
    for partner, together in [(CAT, 5), (LEASH, 9), (SOFA, 7)]:
        pairs[DOG, partner] = pairs[partner, DOG] = together

    source, target, _ = top_pmi_edges(pairs, counts, 100, min_pair_images=1, top_m=2)

    dog_hears = source[target == DOG].tolist()
    assert dog_hears == [LEASH, SOFA]  # highest PMI first, cat left out


def write_tiny_dataset(tmp_path: Path) -> ConceptEdgeConfig:
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    def image(image_id, synsets):
        objects = [
            {"object_id": image_id * 10 + k, "names": [s.split(".")[0]], "synsets": [s]}
            for k, s in enumerate(synsets)
        ]
        return {"image_id": image_id, "objects": objects}

    objects = [
        image(1, ["dog.n.01", "leash.n.01"]),
        image(2, ["dog.n.01", "leash.n.01"]),
        image(3, ["cat.n.01"]),
        image(4, ["sofa.n.01"]),
        image(5, ["cat.n.01", "sofa.n.01"]),  # test image: must not create cat <-> sofa
    ]
    (data_dir / "objects.json").write_text(json.dumps(objects), encoding="utf-8")
    split_file = tmp_path / "split.json"
    save_split({"train": ["1", "2", "3", "4"], "val": [], "test": ["5"]}, split_file)

    split = {"train": np.array([1, 2, 3, 4]), "val": np.array([], dtype=np.int64), "test": [5]}
    graph = build_contains_graph({}, split, CONCEPTS)
    graph_dir = tmp_path / "graph"
    save_contains_graph(
        graph_dir,
        graph,
        np.array(CONCEPTS),
        np.zeros((4, 2), dtype=np.float32),
        {"split_sha256": file_sha256(split_file)},
    )
    return ConceptEdgeConfig(
        dataset="visual_genome",
        data_dir=str(data_dir),
        objects_file="objects.json",
        split_file=str(split_file),
        graph_dir=str(graph_dir),
        min_pair_images=1,
        top_m=10,
        output_dir=str(tmp_path / "edges"),
    )


def test_build_links_train_pairs_only_and_saves_them(tmp_path):
    config = write_tiny_dataset(tmp_path)

    meta = run_build_concept_edges(config)
    edges, saved_meta = load_concept_edges(config.output_dir)

    pairs = set(zip(edges.edge_source.tolist(), edges.edge_target.tolist(), strict=True))
    assert pairs == {(DOG, LEASH), (LEASH, DOG)}
    assert edges.concept_keys.tolist() == CONCEPTS
    assert saved_meta["num_edges"] == meta["num_edges"] == 2


def test_build_refuses_to_overwrite(tmp_path):
    config = write_tiny_dataset(tmp_path)
    run_build_concept_edges(config)

    with pytest.raises(FileExistsError):
        run_build_concept_edges(config)


def test_build_rejects_a_graph_of_another_split(tmp_path):
    config = write_tiny_dataset(tmp_path)
    other_split = tmp_path / "other_split.json"
    save_split({"train": ["1", "2"], "val": ["3", "4"], "test": ["5"]}, other_split)
    config = dataclasses.replace(config, split_file=str(other_split))

    with pytest.raises(ValueError, match="different split"):
        run_build_concept_edges(config)
