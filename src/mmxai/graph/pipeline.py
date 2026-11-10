"""Build and save the "contains" graph of one dataset from its config (ADR-0045, ADR-0046).

Steps: read objects -> name -> synset dictionary counted on train -> concepts per
image (dataset-specific) -> vocabulary counted on train -> CLIP text vector per
concept -> contains edges for every split image -> graph.npz + meta.json.

The edges come from config.edge_source:
    clip_topk    each image's k nearest concepts by CLIP cosine (ADR-0046, main model)
    annotations  the image's own annotated objects (ADR-0045; on val/test this reads
                 the same annotations that define the query answers, so it is only an
                 upper-bound ablation)
"""

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from mmxai.data.scene_graph import build_scene_graphs
from mmxai.data.splits import SPLIT_NAMES, load_split
from mmxai.features.cache import load_image_features, load_meta
from mmxai.graph.clip_edges import clip_topk_concepts
from mmxai.graph.contains import (
    build_contains_graph,
    check_same_clip_model,
    save_contains_graph,
    select_concepts,
)
from mmxai.graph.visual_genome import (
    build_name_to_synset,
    concept_text,
    image_concepts_from_scene_graphs,
)
from mmxai.utils.runs import file_sha256, git_state
from mmxai.utils.yaml_config import load_yaml_dataclass

SUPPORTED_DATASETS = ("visual_genome",)
EDGE_SOURCES = ("clip_topk", "annotations")


@dataclass(frozen=True)
class GraphConfig:
    """Settings for building one graph; see configs/graph_vg.yaml."""

    dataset: str
    data_dir: str
    objects_file: str
    split_file: str
    min_train_images: int
    edge_source: str
    clip_topk_k: int  # used only when edge_source is clip_topk
    features_dir: str  # cached CLIP image vectors; used only when edge_source is clip_topk
    concept_prompt: str
    model: str
    pretrained: str
    device: str
    text_batch_size: int
    output_dir: str


def load_graph_config(path: str | Path) -> GraphConfig:
    """Load the YAML config and reject values the builder cannot use."""
    config = load_yaml_dataclass(path, GraphConfig)
    if config.dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"dataset must be one of {SUPPORTED_DATASETS}, got {config.dataset!r}")
    if config.edge_source not in EDGE_SOURCES:
        raise ValueError(f"edge_source must be one of {EDGE_SOURCES}, got {config.edge_source!r}")
    if "{}" not in config.concept_prompt:
        raise ValueError("concept_prompt needs a {} where the concept name goes")
    return config


def clip_text_encoder(config: GraphConfig) -> Callable[[list[str]], np.ndarray]:
    """The frozen CLIP text encoder that made the cached image embeddings (ADR-0041)."""
    # Imported here so the graph code can be tested without torch.
    from mmxai.features.clip_encoder import encode_texts, load_frozen_clip

    bundle = load_frozen_clip(config.model, config.pretrained, config.device)

    def encode(texts: list[str]) -> np.ndarray:
        return encode_texts(bundle, texts, config.text_batch_size)

    return encode


def cached_image_features(config: GraphConfig, image_ids: np.ndarray) -> np.ndarray:
    """CLIP vectors of image_ids from the feature cache, one row per id in that order.

    The cache must come from the model that encodes the concepts, or the two
    cosines being compared would live in different spaces.
    """
    features_dir = Path(config.features_dir)
    check_same_clip_model(config.model, config.pretrained, load_meta(features_dir))
    embeddings, feature_ids = load_image_features(features_dir)
    row_of_id = {}
    for row, image_id in enumerate(feature_ids.tolist()):
        row_of_id[image_id] = row
    missing = [image_id for image_id in image_ids.tolist() if image_id not in row_of_id]
    if missing:
        raise ValueError(f"{len(missing)} split images have no cached feature, e.g. {missing[0]}")
    rows = [row_of_id[image_id] for image_id in image_ids.tolist()]
    return embeddings[rows]


def read_objects(config: GraphConfig) -> list[dict]:
    """Read objects.json, explaining where to get it when it is missing."""
    path = Path(config.data_dir) / config.objects_file
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download objects.json.zip from the Visual Genome page and "
            "unzip it there (docs/guides/xay-do-thi.md, section 2)."
        )
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def run_build_graph(
    config: GraphConfig, encode_texts: Callable[[list[str]], np.ndarray] | None = None
) -> dict:
    """Build the graph, write it to config.output_dir and return its meta.

    encode_texts maps sentences to vectors; tests pass a fake one, the CLI uses CLIP.
    """
    if Path(config.output_dir).exists():
        # Checked first so a finished graph is not rebuilt for nothing.
        raise FileExistsError(f"{config.output_dir} already exists; delete it to build again")
    split = load_split(config.split_file)
    all_ids = np.sort(np.concatenate([split[name] for name in SPLIT_NAMES]))

    # Relations are not needed: the graph has contains edges only (ADR-0045).
    graphs, parse_report = build_scene_graphs(read_objects(config), [], set(all_ids.tolist()))
    name_to_synset = build_name_to_synset(graphs, split["train"])
    image_concepts, synset_sources = image_concepts_from_scene_graphs(graphs, name_to_synset)

    concepts = select_concepts(image_concepts, split["train"], config.min_train_images)
    if not concepts:
        # Checked here: later steps would either crash on an empty array or save an
        # empty graph, and neither says the threshold is the cause.
        raise ValueError(
            f"no concept appears in >= {config.min_train_images} train images; "
            "lower min_train_images"
        )

    # select_concepts returns the concepts sorted, the same order build_contains_graph
    # gives concept rows, so row j of concept_features belongs to concept row j.
    concept_names = [concept_text(key) for key in concepts]
    prompts = [config.concept_prompt.format(name) for name in concept_names]
    if encode_texts is None:
        encode_texts = clip_text_encoder(config)
    concept_features = encode_texts(prompts)

    if config.edge_source == "clip_topk":
        edge_concepts = clip_topk_concepts(
            all_ids,
            cached_image_features(config, all_ids),
            concepts,
            concept_features,
            config.clip_topk_k,
        )
        clip_topk_k = config.clip_topk_k
    else:
        edge_concepts = image_concepts
        clip_topk_k = None
    graph = build_contains_graph(edge_concepts, split, concepts)

    images_with_edges = set(graph.image_ids[graph.edge_image].tolist())
    images_without_edges = {}
    for name in SPLIT_NAMES:
        missing = [
            image_id for image_id in split[name].tolist() if image_id not in images_with_edges
        ]
        images_without_edges[name] = len(missing)

    meta = {
        "dataset": config.dataset,
        "edge_types": ["contains"],
        "edge_source": config.edge_source,
        "clip_topk_k": clip_topk_k,
        "num_images": len(graph.image_ids),
        "num_concepts": len(graph.concept_keys),
        "num_edges": len(graph.edge_image),
        "images_without_edges": images_without_edges,
        "synset_sources": synset_sources,
        "name_dictionary_size": len(name_to_synset),
        "dropped_records": dict(sorted(parse_report.items())),
        "feature_dim": int(concept_features.shape[1]),
        "objects_sha256": file_sha256(Path(config.data_dir) / config.objects_file),
        "split_sha256": file_sha256(config.split_file),
        "git": git_state(),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": asdict(config),
    }
    save_contains_graph(config.output_dir, graph, np.array(concept_names), concept_features, meta)
    return meta
