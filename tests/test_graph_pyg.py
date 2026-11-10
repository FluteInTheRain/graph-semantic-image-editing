"""Packing the contains graph into a PyTorch Geometric Data object (ADR-0045)."""

import numpy as np
import pytest

pytest.importorskip("torch_geometric")

from mmxai.graph.contains import ContainsGraph, train_subgraph  # noqa: E402
from mmxai.graph.pyg import CONCEPT_NODE, IMAGE_NODE, to_pyg_data  # noqa: E402

# Made-up graph: image 10 (train) contains concept 0, image 20 (test) contains
# concepts 0 and 1.
GRAPH = ContainsGraph(
    image_ids=np.array([10, 20]),
    image_split=np.array(["train", "test"]),
    concept_keys=np.array(["dog.n.01", "sofa.n.01"]),
    edge_image=np.array([0, 1, 1]),
    edge_concept=np.array([0, 0, 1]),
)
CONCEPT_FEATURES = np.array([[0, 0, 1], [0, 1, 0]], dtype=np.float32)
# Feature cache rows are in a different order than the graph's images on purpose.
FEATURE_IMAGE_IDS = np.array([20, 10, 30])
IMAGE_FEATURES = np.array([[2, 2, 2], [1, 1, 1], [3, 3, 3]], dtype=np.float32)
GRAPH_META = {"config": {"model": "ViT-B-32", "pretrained": "laion2b_s34b_b79k"}}
FEATURE_META = {"model": "ViT-B-32", "pretrained": "laion2b_s34b_b79k"}


def make_data():
    return to_pyg_data(
        GRAPH, CONCEPT_FEATURES, IMAGE_FEATURES, FEATURE_IMAGE_IDS, GRAPH_META, FEATURE_META
    )


def test_images_come_first_then_concepts():
    data = make_data()

    assert data.num_nodes == 4
    assert data.node_type.tolist() == [IMAGE_NODE, IMAGE_NODE, CONCEPT_NODE, CONCEPT_NODE]


def test_image_features_follow_the_graph_order_not_the_cache_order():
    data = make_data()

    assert data.x[0].tolist() == [1, 1, 1]  # image 10
    assert data.x[1].tolist() == [2, 2, 2]  # image 20
    assert data.x[2].tolist() == [0, 0, 1]  # concept dog


def test_every_contains_edge_is_stored_in_both_directions():
    # GCN/GraphSAGE pass messages along edge_index; both directions let images
    # learn from concepts and concepts from images.
    data = make_data()

    pairs = set(zip(data.edge_index[0].tolist(), data.edge_index[1].tolist(), strict=True))
    assert pairs == {(0, 2), (2, 0), (1, 2), (2, 1), (1, 3), (3, 1)}


def test_missing_image_feature_is_an_error():
    with pytest.raises(ValueError, match="20"):
        to_pyg_data(
            GRAPH,
            CONCEPT_FEATURES,
            IMAGE_FEATURES[1:],
            FEATURE_IMAGE_IDS[1:],
            GRAPH_META,
            FEATURE_META,
        )


def test_duplicate_image_id_in_the_cache_is_an_error():
    # Without this check the second row would silently replace the first.
    duplicate_ids = np.array([20, 10, 20])

    with pytest.raises(ValueError, match="20"):
        to_pyg_data(
            GRAPH,
            CONCEPT_FEATURES,
            IMAGE_FEATURES,
            duplicate_ids,
            GRAPH_META,
            FEATURE_META,
        )


def test_image_cache_from_another_clip_model_is_an_error():
    other_model = {"model": "ViT-B-32", "pretrained": "openai"}

    with pytest.raises(ValueError, match="openai"):
        to_pyg_data(
            GRAPH,
            CONCEPT_FEATURES,
            IMAGE_FEATURES,
            FEATURE_IMAGE_IDS,
            GRAPH_META,
            other_model,
        )


def test_training_data_has_no_val_or_test_image_in_edge_index():
    # PDF section 6 / ADR-0045: the GNN must not pass messages through test images.
    data = to_pyg_data(
        train_subgraph(GRAPH),
        CONCEPT_FEATURES,
        IMAGE_FEATURES,
        FEATURE_IMAGE_IDS,
        GRAPH_META,
        FEATURE_META,
    )

    assert data.node_type.tolist() == [IMAGE_NODE, CONCEPT_NODE, CONCEPT_NODE]
    assert data.x[0].tolist() == [1, 1, 1]  # image 10, the only train image
    pairs = set(zip(data.edge_index[0].tolist(), data.edge_index[1].tolist(), strict=True))
    assert pairs == {(0, 1), (1, 0)}
