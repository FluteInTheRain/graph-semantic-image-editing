"""Degree-preserving edge shuffle of the contains graph (ADR-0046 section 6 ablation).

Question it answers: does the graph help because of WHICH concepts an image links
to, or only because every image gets some extra messages? The shuffled graph keeps
the number of edges of every image and of every concept, but links images to
random concepts. A model trained on it with the same settings shows how much of
the gain survives without the real relations.

Algorithm, inside each split on its own (train, val, test):
  1. Keep every edge's image; give the edges a random permutation of their
     concepts. Image degrees are unchanged (each image keeps its edges) and concept
     degrees are unchanged (the same concepts are handed out, only to other edges).
  2. A permutation can give one image the same concept twice. Repair each such
     duplicate by swapping its concept with the concept of another random edge of
     the same split, but only if the swap creates no new duplicate. A swap moves
     two concepts between two edges, so both degree counts stay the same.
Shuffling inside a split means no edge ever crosses splits, and split_subgraph of
each split keeps its concept degrees.
"""

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from mmxai.graph.contains import (
    GRAPH_FILE,
    ContainsGraph,
    load_contains_graph,
    save_contains_graph,
)
from mmxai.utils.runs import file_sha256, git_state
from mmxai.utils.yaml_config import load_yaml_dataclass


def shuffle_contains_edges(
    graph: ContainsGraph, seed: int, max_tries_per_edge: int = 1000
) -> ContainsGraph:
    """Return a copy of `graph` whose image -> concept edges are rewired at random.

    Kept: node ids, image splits, concept keys, every image's number of edges, and
    every concept's number of edges inside each split. No (image, concept) pair
    appears twice. The same seed always gives the same graph.
    """
    rng = np.random.default_rng(seed)
    new_edge_concept = graph.edge_concept.copy()
    edge_split = graph.image_split[graph.edge_image]

    # np.unique returns the split names sorted, so the random draws always happen
    # in the same order and the result depends on the seed only.
    for split_name in np.unique(graph.image_split).tolist():
        edge_rows = np.flatnonzero(edge_split == split_name)
        images = graph.edge_image[edge_rows].tolist()
        concepts = graph.edge_concept[edge_rows].tolist()
        new_concepts = shuffle_one_split(images, concepts, rng, max_tries_per_edge)
        new_edge_concept[edge_rows] = new_concepts

    # Sort the edges by (image, concept) like build_contains_graph, so a saved
    # shuffled graph looks like any other graph.
    order = np.lexsort((new_edge_concept, graph.edge_image))
    return ContainsGraph(
        image_ids=graph.image_ids,
        image_split=graph.image_split,
        concept_keys=graph.concept_keys,
        edge_image=graph.edge_image[order],
        edge_concept=new_edge_concept[order],
    )


def shuffle_one_split(
    images: list[int], concepts: list[int], rng: np.random.Generator, max_tries_per_edge: int
) -> list[int]:
    """Edge k is images[k] -> concepts[k]. Return the new concept of every edge.

    Step 1 permutes the concepts; step 2 swaps away duplicate pairs (module docstring).
    """
    order = rng.permutation(len(concepts)).tolist()
    new_concepts = [concepts[k] for k in order]

    # How many edges currently link each (image, concept) pair; > 1 is a duplicate.
    pair_count: dict[tuple[int, int], int] = {}
    for image, concept in zip(images, new_concepts, strict=True):
        add_pair(pair_count, (image, concept))

    for edge in range(len(images)):
        tries = 0
        while pair_count[(images[edge], new_concepts[edge])] > 1:
            if tries == max_tries_per_edge:
                raise RuntimeError(
                    f"could not remove a duplicate edge after {max_tries_per_edge} random "
                    f"swaps (image row {images[edge]}, concept row {new_concepts[edge]}); "
                    "the split may be too dense to shuffle"
                )
            tries += 1
            other = int(rng.integers(len(images)))
            if can_swap(images, new_concepts, pair_count, edge, other):
                swap_concepts(images, new_concepts, pair_count, edge, other)
    return new_concepts


def can_swap(
    images: list[int],
    concepts: list[int],
    pair_count: dict[tuple[int, int], int],
    edge: int,
    other: int,
) -> bool:
    """True if giving `edge` the concept of `other` (and back) creates no duplicate."""
    if concepts[edge] == concepts[other]:
        return False  # swapping equal concepts changes nothing
    # Both new pairs must be absent. This also rejects two edges of the same image:
    # that image already has concepts[edge], so (images[other], concepts[edge]) exists.
    new_pair_of_edge = (images[edge], concepts[other])
    new_pair_of_other = (images[other], concepts[edge])
    return new_pair_of_edge not in pair_count and new_pair_of_other not in pair_count


def swap_concepts(
    images: list[int],
    concepts: list[int],
    pair_count: dict[tuple[int, int], int],
    edge: int,
    other: int,
) -> None:
    """Exchange the concepts of two edges, in place, and update pair_count."""
    remove_pair(pair_count, (images[edge], concepts[edge]))
    remove_pair(pair_count, (images[other], concepts[other]))
    concepts[edge], concepts[other] = concepts[other], concepts[edge]
    add_pair(pair_count, (images[edge], concepts[edge]))
    add_pair(pair_count, (images[other], concepts[other]))


def add_pair(pair_count: dict[tuple[int, int], int], pair: tuple[int, int]) -> None:
    pair_count[pair] = pair_count.get(pair, 0) + 1


def remove_pair(pair_count: dict[tuple[int, int], int], pair: tuple[int, int]) -> None:
    # A pair with no edge left is deleted, so "pair in pair_count" means "the edge exists".
    pair_count[pair] -= 1
    if pair_count[pair] == 0:
        del pair_count[pair]


@dataclass(frozen=True)
class ShuffleGraphConfig:
    """Settings for one shuffled graph; see configs/graph_vg_clip_topk_shuffled.yaml."""

    graph_dir: str  # the graph whose edges are shuffled
    shuffle_seed: int
    output_dir: str


def load_shuffle_graph_config(path: str | Path) -> ShuffleGraphConfig:
    """Load the YAML config; a mistyped or missing key is an error."""
    return load_yaml_dataclass(path, ShuffleGraphConfig)


def run_shuffle_graph(config: ShuffleGraphConfig) -> dict:
    """Shuffle the edges of config.graph_dir, save to config.output_dir, return the meta.

    The meta of the source graph is copied, so the split hash and the CLIP model
    checks of graph_rerank.load_inputs and to_pyg_data still pass; a "shuffle"
    entry records the seed and the exact source graph.
    """
    if Path(config.output_dir).exists():
        raise FileExistsError(f"{config.output_dir} already exists; delete it to build again")
    graph, concept_names, concept_features, source_meta = load_contains_graph(config.graph_dir)
    shuffled = shuffle_contains_edges(graph, config.shuffle_seed)

    meta = dict(source_meta)
    # edge_source is copied into every run's metrics.json, so a run on this graph
    # cannot be mistaken for a run on the real graph.
    meta["edge_source"] = f"{source_meta['edge_source']}_shuffled"
    meta["shuffle"] = {
        "method": "degree_preserving_within_split",
        "seed": config.shuffle_seed,
        "source_graph_dir": config.graph_dir,
        "source_graph_sha256": file_sha256(Path(config.graph_dir) / GRAPH_FILE),
        "source_git": source_meta.get("git"),
        "source_created_at": source_meta.get("created_at"),
        "config": asdict(config),
    }
    # git / created_at now describe the shuffle run; the source's are kept above.
    meta["git"] = git_state()
    meta["created_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    save_contains_graph(config.output_dir, shuffled, concept_names, concept_features, meta)
    return meta
