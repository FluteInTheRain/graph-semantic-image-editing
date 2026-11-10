"""The image-contains-concept graph on tiny synthetic data (ADR-0045).

Images and their (made-up) split, nothing here comes from Visual Genome:
    1 train  dog, grass
    2 train  dog, sofa
    3 val    dog
    4 test   dog, zebra        zebra is never seen in train
    5 test   (no objects)
"""

import numpy as np
import pytest

from mmxai.graph.contains import (
    build_contains_graph,
    check_same_clip_model,
    load_contains_graph,
    save_contains_graph,
    select_concepts,
    train_subgraph,
)

IMAGE_CONCEPTS = {
    1: {"dog.n.01", "grass.n.01"},
    2: {"dog.n.01", "sofa.n.01"},
    3: {"dog.n.01"},
    4: {"dog.n.01", "zebra.n.01"},
}
SPLIT = {"train": np.array([1, 2]), "val": np.array([3]), "test": np.array([4, 5])}
TRAIN_IDS = SPLIT["train"]


def test_concepts_are_counted_on_train_images_only():
    # zebra appears only in a test image, so it must not enter the vocabulary.
    concepts = select_concepts(IMAGE_CONCEPTS, TRAIN_IDS, min_train_images=1)

    assert concepts == ["dog.n.01", "grass.n.01", "sofa.n.01"]


def test_concepts_below_min_train_images_are_dropped():
    concepts = select_concepts(IMAGE_CONCEPTS, TRAIN_IDS, min_train_images=2)

    assert concepts == ["dog.n.01"]


def test_test_images_do_not_change_the_vocabulary():
    with_test = select_concepts(IMAGE_CONCEPTS, TRAIN_IDS, min_train_images=1)
    train_only = {image_id: IMAGE_CONCEPTS[image_id] for image_id in TRAIN_IDS.tolist()}

    assert select_concepts(train_only, TRAIN_IDS, min_train_images=1) == with_test


def test_select_concepts_rejects_min_train_images_below_one():
    with pytest.raises(ValueError, match="min_train_images"):
        select_concepts(IMAGE_CONCEPTS, TRAIN_IDS, min_train_images=0)


def test_every_image_gets_a_node_even_without_objects():
    graph = build_contains_graph(IMAGE_CONCEPTS, SPLIT, ["dog.n.01", "sofa.n.01"])

    assert graph.image_ids.tolist() == [1, 2, 3, 4, 5]


def test_contains_edges_link_images_to_known_concepts_only():
    graph = build_contains_graph(IMAGE_CONCEPTS, SPLIT, ["dog.n.01", "sofa.n.01"])

    pairs = set()
    for image_row, concept_row in zip(graph.edge_image, graph.edge_concept, strict=True):
        pairs.add((int(graph.image_ids[image_row]), str(graph.concept_keys[concept_row])))
    # Test image 4 keeps its dog edge; its zebra is not in the vocabulary, so no edge.
    assert pairs == {
        (1, "dog.n.01"),
        (2, "dog.n.01"),
        (2, "sofa.n.01"),
        (3, "dog.n.01"),
        (4, "dog.n.01"),
    }


def test_edges_are_sorted_by_image_then_concept():
    graph = build_contains_graph(IMAGE_CONCEPTS, SPLIT, ["dog.n.01", "sofa.n.01"])

    assert graph.edge_image.tolist() == [0, 1, 1, 2, 3]
    assert graph.edge_concept.tolist() == [0, 0, 1, 0, 0]


def test_graph_is_saved_and_loaded_unchanged(tmp_path):
    graph = build_contains_graph(IMAGE_CONCEPTS, SPLIT, ["dog.n.01", "sofa.n.01"])
    names = np.array(["dog", "sofa"])
    features = np.eye(2, dtype=np.float32)

    save_contains_graph(tmp_path / "graph", graph, names, features, meta={"note": "test"})
    loaded, loaded_names, loaded_features, meta = load_contains_graph(tmp_path / "graph")

    assert loaded.image_ids.tolist() == graph.image_ids.tolist()
    assert loaded.image_split.tolist() == graph.image_split.tolist()
    assert loaded.concept_keys.tolist() == graph.concept_keys.tolist()
    assert loaded.edge_image.tolist() == graph.edge_image.tolist()
    assert loaded.edge_concept.tolist() == graph.edge_concept.tolist()
    assert loaded_names.tolist() == ["dog", "sofa"]
    assert np.array_equal(loaded_features, features)
    assert meta == {"note": "test"}


def test_save_refuses_to_overwrite(tmp_path):
    graph = build_contains_graph(IMAGE_CONCEPTS, SPLIT, ["dog.n.01"])
    names = np.array(["dog"])
    features = np.ones((1, 2), dtype=np.float32)
    save_contains_graph(tmp_path / "graph", graph, names, features, meta={})

    with pytest.raises(FileExistsError):
        save_contains_graph(tmp_path / "graph", graph, names, features, meta={})


def test_save_rejects_one_feature_row_per_concept_mismatch(tmp_path):
    graph = build_contains_graph(IMAGE_CONCEPTS, SPLIT, ["dog.n.01", "sofa.n.01"])

    with pytest.raises(ValueError, match="concept"):
        save_contains_graph(
            tmp_path / "graph",
            graph,
            np.array(["dog", "sofa"]),
            np.ones((1, 2), dtype=np.float32),
            meta={},
        )


GRAPH_META = {"config": {"model": "ViT-B-32", "pretrained": "laion2b_s34b_b79k"}}


def test_same_clip_model_passes():
    feature_meta = {"model": "ViT-B-32", "pretrained": "laion2b_s34b_b79k"}

    check_same_clip_model(GRAPH_META, feature_meta)


def test_a_different_clip_model_is_an_error():
    # Same 512 dimensions, different space: nothing else would catch this.
    feature_meta = {"model": "ViT-B-32", "pretrained": "openai"}

    with pytest.raises(ValueError, match="openai"):
        check_same_clip_model(GRAPH_META, feature_meta)


def test_every_image_node_records_its_split():
    graph = build_contains_graph(IMAGE_CONCEPTS, SPLIT, ["dog.n.01"])

    assert graph.image_split.tolist() == ["train", "train", "val", "test", "test"]


def test_an_image_in_two_splits_is_an_error():
    leaky_split = {"train": np.array([1, 2]), "val": np.array([3]), "test": np.array([2])}

    with pytest.raises(ValueError, match="2"):
        build_contains_graph(IMAGE_CONCEPTS, leaky_split, ["dog.n.01"])


def test_train_subgraph_keeps_only_train_images_and_their_edges():
    graph = build_contains_graph(IMAGE_CONCEPTS, SPLIT, ["dog.n.01", "sofa.n.01"])

    train_graph = train_subgraph(graph)

    assert train_graph.image_ids.tolist() == [1, 2]
    assert set(train_graph.image_split.tolist()) == {"train"}
    pairs = set()
    for image_row, concept_row in zip(
        train_graph.edge_image, train_graph.edge_concept, strict=True
    ):
        image_id = int(train_graph.image_ids[image_row])
        pairs.add((image_id, str(train_graph.concept_keys[concept_row])))
    assert pairs == {(1, "dog.n.01"), (2, "dog.n.01"), (2, "sofa.n.01")}


def test_train_subgraph_keeps_every_concept():
    # Concept rows must not move, so concept vectors still line up with the graph.
    graph = build_contains_graph(IMAGE_CONCEPTS, SPLIT, ["dog.n.01", "sofa.n.01"])

    train_graph = train_subgraph(graph)

    assert train_graph.concept_keys.tolist() == graph.concept_keys.tolist()


def test_graph_saved_without_splits_asks_for_a_rebuild(tmp_path):
    # Graphs built before image_split existed must not load as if they were complete.
    graph_dir = tmp_path / "old_graph"
    graph_dir.mkdir()
    np.savez(
        graph_dir / "graph.npz",
        image_ids=np.array([1]),
        concept_keys=np.array(["dog.n.01"]),
        concept_names=np.array(["dog"]),
        concept_features=np.ones((1, 2), dtype=np.float32),
        edge_image=np.array([0]),
        edge_concept=np.array([0]),
    )
    (graph_dir / "meta.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="build"):
        load_contains_graph(graph_dir)
