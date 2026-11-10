"""Turn a saved contains graph into a PyTorch Geometric Data object (ADR-0045).

This is the one "box" every GNN reads, whatever the dataset:
    x          node vectors, images first (rows 0..I-1), then concepts (rows I..I+C-1)
    edge_index 2 x (2E) node rows; every contains edge in both directions
    node_type  IMAGE_NODE or CONCEPT_NODE per node
"""

import numpy as np
import torch
from torch_geometric.data import Data

from mmxai.graph.contains import ContainsGraph, check_same_clip_model

IMAGE_NODE = 0
CONCEPT_NODE = 1


def to_pyg_data(
    graph: ContainsGraph,
    concept_features: np.ndarray,
    image_features: np.ndarray,
    feature_image_ids: np.ndarray,
    graph_meta: dict,
    feature_meta: dict,
) -> Data:
    """Build Data from the graph, its concept vectors and the cached CLIP image vectors.

    graph_meta / feature_meta are the meta.json of the graph and of the image cache;
    they are required so the same-model check cannot be skipped. The image cache is
    ordered by its own ids, so rows are looked up by id, never assumed to be in
    graph order.
    """
    check_same_clip_model(graph_meta, feature_meta)
    row_of_id = {}
    for row, image_id in enumerate(feature_image_ids.tolist()):
        if image_id in row_of_id:
            raise ValueError(f"image id {image_id} appears twice in the feature cache")
        row_of_id[image_id] = row
    missing = [image_id for image_id in graph.image_ids.tolist() if image_id not in row_of_id]
    if missing:
        raise ValueError(f"{len(missing)} graph images have no cached feature, e.g. {missing[0]}")
    image_rows = [row_of_id[image_id] for image_id in graph.image_ids.tolist()]

    x = np.concatenate([image_features[image_rows], concept_features]).astype(np.float32)

    num_images = len(graph.image_ids)
    image_nodes = graph.edge_image
    concept_nodes = graph.edge_concept + num_images  # concept rows come after the images
    # Message passing follows edge direction; storing both lets images learn from
    # their concepts and concepts from the images that contain them.
    sources = np.concatenate([image_nodes, concept_nodes])
    targets = np.concatenate([concept_nodes, image_nodes])

    node_type = np.concatenate(
        [
            np.full(num_images, IMAGE_NODE, dtype=np.int64),
            np.full(len(graph.concept_keys), CONCEPT_NODE, dtype=np.int64),
        ]
    )
    return Data(
        x=torch.from_numpy(x),
        edge_index=torch.from_numpy(np.stack([sources, targets]).astype(np.int64)),
        node_type=torch.from_numpy(node_type),
    )
