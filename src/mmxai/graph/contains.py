"""The image-contains-concept graph shared by every dataset (ADR-0045). Needs numpy only.

A dataset adapter (e.g. mmxai.graph.visual_genome) turns its annotations into
{image_id: set of concept keys}. Everything here works on that plain mapping, so
the GNN and the re-ranker never need to know which dataset a graph came from.
"""

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

GRAPH_FILE = "graph.npz"
META_FILE = "meta.json"


@dataclass(frozen=True)
class ContainsGraph:
    """Image and concept nodes, and one edge per (image, concept) the image contains.

    Edges are stored as row numbers: edge k links image_ids[edge_image[k]] to
    concept_keys[edge_concept[k]]. Row numbers are what a GNN needs; the id arrays
    turn them back into Visual Genome ids and synsets for explanations.
    """

    image_ids: np.ndarray  # int64, shape (num_images,)
    concept_keys: np.ndarray  # str, shape (num_concepts,), e.g. "dog.n.01"
    edge_image: np.ndarray  # int64, shape (num_edges,)
    edge_concept: np.ndarray  # int64, shape (num_edges,)


def select_concepts(
    image_concepts: dict[int, set[str]], train_ids: np.ndarray, min_train_images: int
) -> list[str]:
    """Return the sorted concepts found in at least min_train_images TRAIN images.

    Decision (ADR-0045, PDF section 6): count on train only.
      Why: the vocabulary is a statistic; letting test images vote on it would leak
      test information into the graph. train_ids has no default on purpose.
    """
    if min_train_images < 1:
        raise ValueError(f"min_train_images must be >= 1, got {min_train_images}")
    counts: dict[str, int] = {}
    for image_id in train_ids.tolist():
        for concept in image_concepts.get(image_id, set()):
            counts[concept] = counts.get(concept, 0) + 1

    kept = []
    for concept, count in counts.items():
        if count >= min_train_images:
            kept.append(concept)
    return sorted(kept)


def build_contains_graph(
    image_concepts: dict[int, set[str]], image_ids: np.ndarray, concepts: list[str]
) -> ContainsGraph:
    """Link every image to each of its concepts that is in `concepts`.

    Every id in image_ids gets a node, even with no edge: test images are search
    candidates whether or not they have objects. Images and concepts are sorted,
    so the same input always gives the same row numbers.
    """
    sorted_image_ids = np.sort(np.asarray(image_ids, dtype=np.int64))
    sorted_concepts = sorted(concepts)
    concept_row = {concept: row for row, concept in enumerate(sorted_concepts)}

    edge_image = []
    edge_concept = []
    for image_row, image_id in enumerate(sorted_image_ids.tolist()):
        rows = []
        for concept in image_concepts.get(image_id, set()):
            if concept in concept_row:
                rows.append(concept_row[concept])
        for row in sorted(rows):
            edge_image.append(image_row)
            edge_concept.append(row)

    return ContainsGraph(
        image_ids=sorted_image_ids,
        concept_keys=np.array(sorted_concepts, dtype=str),
        edge_image=np.array(edge_image, dtype=np.int64),
        edge_concept=np.array(edge_concept, dtype=np.int64),
    )


def save_contains_graph(
    output_dir: str | Path,
    graph: ContainsGraph,
    concept_names: np.ndarray,
    concept_features: np.ndarray,
    meta: dict,
) -> None:
    """Write graph.npz + meta.json. Refuses to overwrite an existing graph.

    concept_names / concept_features hold one row per concept, in concept_keys order.
    """
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"{output_dir} already exists; delete it to build again")
    num_concepts = len(graph.concept_keys)
    if len(concept_names) != num_concepts or len(concept_features) != num_concepts:
        raise ValueError(
            f"{num_concepts} concepts but {len(concept_names)} names "
            f"and {len(concept_features)} feature rows"
        )

    # Same pattern as the feature cache: an interrupted run never leaves a folder
    # that looks complete.
    temp_dir = output_dir.with_name(output_dir.name + ".tmp")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)
    np.savez(
        temp_dir / GRAPH_FILE,
        image_ids=graph.image_ids,
        concept_keys=graph.concept_keys,
        concept_names=np.asarray(concept_names, dtype=str),
        concept_features=np.asarray(concept_features, dtype=np.float32),
        edge_image=graph.edge_image,
        edge_concept=graph.edge_concept,
    )
    with open(temp_dir / META_FILE, "w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2)
    os.replace(temp_dir, output_dir)


def load_contains_graph(
    output_dir: str | Path,
) -> tuple[ContainsGraph, np.ndarray, np.ndarray, dict]:
    """Return (graph, concept_names, concept_features, meta) written by save_contains_graph."""
    output_dir = Path(output_dir)
    with open(output_dir / META_FILE, encoding="utf-8") as file:
        meta = json.load(file)
    with np.load(output_dir / GRAPH_FILE, allow_pickle=False) as data:
        graph = ContainsGraph(
            image_ids=data["image_ids"],
            concept_keys=data["concept_keys"],
            edge_image=data["edge_image"],
            edge_concept=data["edge_concept"],
        )
        concept_names = data["concept_names"]
        concept_features = data["concept_features"]
    return graph, concept_names, concept_features, meta
