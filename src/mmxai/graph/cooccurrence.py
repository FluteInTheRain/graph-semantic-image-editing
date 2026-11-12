"""Concept -> concept edges from how often two concepts are annotated together (ADR-0048).

Two concepts are linked when they appear together in TRAIN images much more often
than chance, measured by PMI (pointwise mutual information):

    PMI(a, b) = log( n_ab * N / (n_a * n_b) )

n_ab = train images annotated with both a and b, n_a = train images with a,
N = number of train images. PMI > 0 means "together more often than by chance".
Each concept a keeps its top_m partners b, as edges b -> a (a hears b).

This is information CLIP does not have: it is counted from human annotations,
while the contains edges and concept vectors come from CLIP itself (ADR-0046).
Needs numpy only.
"""

import json
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from mmxai.data.scene_graph import build_scene_graphs
from mmxai.data.splits import load_split
from mmxai.graph.contains import load_contains_graph
from mmxai.graph.visual_genome import build_name_to_synset, image_concepts_from_scene_graphs
from mmxai.utils.runs import file_sha256, git_state
from mmxai.utils.yaml_config import load_yaml_dataclass

EDGES_FILE = "concept_edges.npz"
META_FILE = "meta.json"


@dataclass(frozen=True)
class ConceptEdges:
    """Directed concept -> concept edges, as rows of the contains graph's concept_keys."""

    concept_keys: np.ndarray  # str, shape (num_concepts,), same order as the contains graph
    edge_source: np.ndarray  # int64, shape (num_edges,): the concept that is heard
    edge_target: np.ndarray  # int64, shape (num_edges,): the concept that listens
    edge_pmi: np.ndarray  # float64, shape (num_edges,): kept for explanations and reports


def count_cooccurrence(
    image_concepts: dict[int, set[str]],
    train_ids: np.ndarray,
    concept_keys: list[str],
    chunk_size: int = 4000,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (pair_counts, concept_counts) over TRAIN images only.

    pair_counts[a, b] = train images annotated with both a and b (diagonal set to 0);
    concept_counts[a] = train images annotated with a.

    Decision (ADR-0048, PDF section 6): count on train only.
      Why: co-occurrence is a statistic; counting val/test annotations would put
      the answers of test queries into the graph. train_ids has no default on purpose.
    """
    concept_row = {concept: row for row, concept in enumerate(concept_keys)}
    num_concepts = len(concept_keys)
    pair_counts = np.zeros((num_concepts, num_concepts), dtype=np.int64)
    train_list = np.asarray(train_ids, dtype=np.int64).tolist()

    # One 0/1 row per image; A^T A counts, for each pair, the images that have both.
    # Chunks keep memory small: all ~98k train images at once would need ~1.5 GB.
    for start in range(0, len(train_list), chunk_size):
        chunk = train_list[start : start + chunk_size]
        has_concept = np.zeros((len(chunk), num_concepts), dtype=np.float32)
        for position, image_id in enumerate(chunk):
            for concept in image_concepts.get(image_id, set()):
                if concept in concept_row:
                    has_concept[position, concept_row[concept]] = 1.0
        # float32 counts are exact here: a chunk adds at most chunk_size per cell.
        pair_counts += np.rint(has_concept.T @ has_concept).astype(np.int64)

    concept_counts = np.diag(pair_counts).copy()
    np.fill_diagonal(pair_counts, 0)
    return pair_counts, concept_counts


def top_pmi_edges(
    pair_counts: np.ndarray,
    concept_counts: np.ndarray,
    num_train_images: int,
    min_pair_images: int,
    top_m: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For every concept a, keep up to top_m partners b with the highest PMI.

    A pair is allowed only if n_ab >= min_pair_images and PMI > 0. Returns
    (edge_source, edge_target, edge_pmi) with source = b, target = a.

    Decision (ADR-0048): PMI, n_ab >= 10, top_m = 10.
      Why: raw counts would link everything to "man" and "window", which appear in
      most images; PMI favours specific partners (dog -> leash). The minimum count
      stops one or two lucky images from giving a huge PMI.
      Proposed by Claude Code, 2026-11-12; not yet approved by the team. Tune on val only.
    """
    if min_pair_images < 1 or top_m < 1:
        raise ValueError("min_pair_images and top_m must be >= 1")

    # Where n_ab = 0 the log is meaningless; those pairs are masked out below anyway.
    with np.errstate(divide="ignore"):
        pmi = np.log(
            pair_counts * num_train_images / np.outer(concept_counts, concept_counts).clip(min=1)
        )
    allowed = (pair_counts >= min_pair_images) & (pmi > 0)

    edge_source = []
    edge_target = []
    edge_pmi = []
    for target in range(len(concept_counts)):
        partners = np.flatnonzero(allowed[target])
        # Stable sort on negated PMI: highest first, ties broken by concept row,
        # so the same counts always give the same edges.
        order = np.argsort(-pmi[target, partners], kind="stable")[:top_m]
        for source in partners[order].tolist():
            edge_source.append(source)
            edge_target.append(target)
            edge_pmi.append(float(pmi[target, source]))
    return (
        np.array(edge_source, dtype=np.int64),
        np.array(edge_target, dtype=np.int64),
        np.array(edge_pmi, dtype=np.float64),
    )


def save_concept_edges(output_dir: str | Path, edges: ConceptEdges, meta: dict) -> None:
    """Write concept_edges.npz + meta.json. Refuses to overwrite."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"{output_dir} already exists; delete it to build again")
    # Same pattern as the contains graph: an interrupted run never looks complete.
    temp_dir = output_dir.with_name(output_dir.name + ".tmp")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)
    np.savez(
        temp_dir / EDGES_FILE,
        concept_keys=edges.concept_keys,
        edge_source=edges.edge_source,
        edge_target=edges.edge_target,
        edge_pmi=edges.edge_pmi,
    )
    with open(temp_dir / META_FILE, "w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2)
    os.replace(temp_dir, output_dir)


def load_concept_edges(output_dir: str | Path) -> tuple[ConceptEdges, dict]:
    """Return (edges, meta) written by save_concept_edges."""
    output_dir = Path(output_dir)
    with open(output_dir / META_FILE, encoding="utf-8") as file:
        meta = json.load(file)
    with np.load(output_dir / EDGES_FILE, allow_pickle=False) as data:
        edges = ConceptEdges(
            concept_keys=data["concept_keys"],
            edge_source=data["edge_source"],
            edge_target=data["edge_target"],
            edge_pmi=data["edge_pmi"],
        )
    return edges, meta


@dataclass(frozen=True)
class ConceptEdgeConfig:
    """Settings for building concept edges; see configs/concept_edges_vg_cooccurrence.yaml."""

    dataset: str
    data_dir: str
    objects_file: str
    split_file: str
    graph_dir: str  # the contains graph whose concepts get linked
    min_pair_images: int
    top_m: int
    output_dir: str


def load_concept_edge_config(path: str | Path) -> ConceptEdgeConfig:
    config = load_yaml_dataclass(path, ConceptEdgeConfig)
    if config.dataset != "visual_genome":
        raise ValueError(f"only visual_genome is supported, got {config.dataset!r}")
    return config


def run_build_concept_edges(config: ConceptEdgeConfig, objects: list[dict] | None = None) -> dict:
    """Count co-occurrence on train annotations, keep PMI edges, save them, return meta.

    objects is the parsed objects.json; tests pass a tiny list, the CLI reads the file.
    """
    if Path(config.output_dir).exists():
        raise FileExistsError(f"{config.output_dir} already exists; delete it to build again")
    split = load_split(config.split_file)
    graph, _, _, graph_meta = load_contains_graph(config.graph_dir)
    # The concepts must be the ones of a graph built on this same split.
    if graph_meta["split_sha256"] != file_sha256(config.split_file):
        raise ValueError(f"{config.graph_dir} was built with a different split file")

    objects_path = Path(config.data_dir) / config.objects_file
    if objects is None:
        with open(objects_path, encoding="utf-8") as file:
            objects = json.load(file)
    # Only train images are parsed: val/test annotations are never read here.
    train_ids = split["train"]
    scene_graphs, _ = build_scene_graphs(objects, [], set(train_ids.tolist()))
    # Same name -> synset dictionary as the graph build (counted on train, ADR-0045).
    name_to_synset = build_name_to_synset(scene_graphs, train_ids)
    image_concepts, _ = image_concepts_from_scene_graphs(scene_graphs, name_to_synset)

    concept_keys = graph.concept_keys.tolist()
    pair_counts, concept_counts = count_cooccurrence(image_concepts, train_ids, concept_keys)
    edge_source, edge_target, edge_pmi = top_pmi_edges(
        pair_counts, concept_counts, len(train_ids), config.min_pair_images, config.top_m
    )
    edges = ConceptEdges(graph.concept_keys, edge_source, edge_target, edge_pmi)

    meta = {
        "dataset": config.dataset,
        "edge_type": "cooccurs_with",
        "num_concepts": len(concept_keys),
        "num_edges": len(edge_source),
        "concepts_without_edges": int(len(concept_keys) - len(np.unique(edge_target))),
        "num_train_images": len(train_ids),
        "graph_dir": config.graph_dir,
        "objects_sha256": file_sha256(objects_path) if objects_path.exists() else None,
        "split_sha256": file_sha256(config.split_file),
        "git": git_state(),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": asdict(config),
    }
    save_concept_edges(config.output_dir, edges, meta)
    return meta
