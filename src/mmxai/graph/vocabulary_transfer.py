"""Build a CLIP top-k contains graph for a dataset that has no object annotations.

Concept nodes (keys, names, CLIP text vectors) are copied from an existing graph;
each image is linked to its k nearest concepts by CLIP cosine, as in ADR-0046.
Used for Flickr30k with the VG vocabulary (ADR-0050).
"""

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from mmxai.data.splits import SPLIT_NAMES, load_split
from mmxai.features.cache import load_image_features, load_meta
from mmxai.graph.clip_edges import clip_topk_concepts
from mmxai.graph.contains import (
    build_contains_graph,
    check_same_clip_model,
    load_contains_graph,
    save_contains_graph,
)
from mmxai.utils.runs import file_sha256, git_state
from mmxai.utils.yaml_config import load_yaml_dataclass


@dataclass(frozen=True)
class VocabularyTransferConfig:
    """Settings for one transferred-vocabulary graph; see configs/graph_flickr30k_clip_topk.yaml."""

    dataset: str
    split_file: str
    features_dir: str
    vocabulary_graph_dir: str  # graph whose concept nodes are copied
    clip_topk_k: int
    model: str
    pretrained: str
    output_dir: str


def load_vocabulary_transfer_config(path: str | Path) -> VocabularyTransferConfig:
    """Load the YAML config (ADR-0007)."""
    return load_yaml_dataclass(path, VocabularyTransferConfig)


def split_image_features(features_dir: str | Path, image_ids: np.ndarray) -> np.ndarray:
    """Cached CLIP vectors of image_ids, one row per id in that order."""
    embeddings, feature_ids = load_image_features(features_dir)
    row_of_id = {image_id: row for row, image_id in enumerate(feature_ids.tolist())}
    missing = [image_id for image_id in image_ids.tolist() if image_id not in row_of_id]
    if missing:
        raise ValueError(f"{len(missing)} split images have no cached feature, e.g. {missing[0]}")
    return embeddings[[row_of_id[image_id] for image_id in image_ids.tolist()]]


def run_vocabulary_transfer(config: VocabularyTransferConfig) -> dict:
    """Build the graph, write it to config.output_dir and return its meta."""
    if Path(config.output_dir).exists():
        raise FileExistsError(f"{config.output_dir} already exists; delete it to build again")

    source_graph, concept_names, concept_features, source_meta = load_contains_graph(
        config.vocabulary_graph_dir
    )
    concepts = source_graph.concept_keys.tolist()

    # Both the copied concept vectors and the new image vectors must come from the
    # CLIP model named in the config, or their cosines would not be comparable.
    source_config = source_meta["config"]
    if (source_config["model"], source_config["pretrained"]) != (config.model, config.pretrained):
        raise ValueError(
            f"vocabulary graph uses {source_config['model']}/{source_config['pretrained']}, "
            f"config asks for {config.model}/{config.pretrained}"
        )
    check_same_clip_model(config.model, config.pretrained, load_meta(Path(config.features_dir)))

    split = load_split(config.split_file)
    all_ids = np.sort(np.concatenate([split[name] for name in SPLIT_NAMES]))
    edge_concepts = clip_topk_concepts(
        all_ids,
        split_image_features(config.features_dir, all_ids),
        concepts,
        concept_features,
        config.clip_topk_k,
    )
    graph = build_contains_graph(edge_concepts, split, concepts)

    meta = {
        "dataset": config.dataset,
        "edge_types": ["contains"],
        "edge_source": "clip_topk",
        "clip_topk_k": config.clip_topk_k,
        "num_images": len(graph.image_ids),
        "num_concepts": len(graph.concept_keys),
        "num_edges": len(graph.edge_image),
        "vocabulary_from": {
            "graph_dir": config.vocabulary_graph_dir,
            "dataset": source_meta["dataset"],
            "graph_sha256": file_sha256(Path(config.vocabulary_graph_dir) / "graph.npz"),
        },
        "feature_dim": int(concept_features.shape[1]),
        "split_sha256": file_sha256(config.split_file),
        "git": git_state(),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        # to_pyg_data reads model and pretrained from here.
        "config": asdict(config),
    }
    save_contains_graph(config.output_dir, graph, concept_names, concept_features, meta)
    return meta
