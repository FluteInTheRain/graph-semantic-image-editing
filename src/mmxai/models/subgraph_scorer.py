"""Stage 1 of the G-Retriever reranker: how much of S_q does an image touch? (ADR-0051)

g(q, i) = |N(i) ∩ S_q| / |S_q|, where N(i) are the concepts of image i (CLIP top-16
edges, ADR-0046, same for every split, no annotation) and S_q is the PCST subgraph of
the query. Every concept of S_q counts the same, bridges included: summing prizes
instead would give bridges (prize 0) no weight and lose the indirect-query mechanism.
No training. The explanation is exactly the set the score counts (ADR-0016).
"""

import numpy as np

from mmxai.graph.contains import ContainsGraph
from mmxai.graph.pcst import QuerySubgraph


def image_concept_sets(graph: ContainsGraph) -> dict[int, set[int]]:
    """Image id -> concept rows it contains (every image appears, possibly with no concept)."""
    sets: dict[int, set[int]] = {}
    for image_id in graph.image_ids.tolist():
        sets[image_id] = set()
    for image_row, concept_row in zip(
        graph.edge_image.tolist(), graph.edge_concept.tolist(), strict=True
    ):
        sets[graph.image_ids[image_row].item()].add(concept_row)
    return sets


def overlap(concepts: set[int], subgraph: QuerySubgraph) -> set[int]:
    """The concepts of an image that lie in S_q."""
    return concepts & set(subgraph.concepts.tolist())


def overlap_scores(
    candidate_ids: np.ndarray, subgraphs: list[QuerySubgraph], sets: dict[int, set[int]]
) -> np.ndarray:
    """Score of every candidate, shape (num_queries, top_k); row q uses subgraphs[q]."""
    if len(candidate_ids) != len(subgraphs):
        raise ValueError(f"{len(candidate_ids)} queries but {len(subgraphs)} subgraphs")
    scores = np.zeros(candidate_ids.shape, dtype=np.float32)
    for query, subgraph in enumerate(subgraphs):
        size = len(subgraph.concepts)
        if size == 0:
            continue
        for column, image_id in enumerate(candidate_ids[query].tolist()):
            scores[query, column] = len(overlap(sets[image_id], subgraph)) / size
    return scores


def explain_overlap(
    image_id: int, subgraph: QuerySubgraph, sets: dict[int, set[int]], concept_keys: np.ndarray
) -> dict:
    """The concepts that made the score, which of them are bridges, and the score without each."""
    matched = sorted(overlap(sets[image_id], subgraph))
    size = len(subgraph.concepts)
    score = len(matched) / size if size else 0.0
    items = []
    without_each = []
    for concept_row in matched:
        is_bridge = bool(subgraph.prizes[concept_row] == 0)
        items.append({"concept": str(concept_keys[concept_row]), "bridge": is_bridge})
        # Removing one matched concept removes exactly one hit from the numerator.
        without_each.append((len(matched) - 1) / size)
    return {"matched": items, "score": score, "score_without_each": without_each}
