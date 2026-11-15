"""Query-dependent concept subgraphs with PCST, as G-Retriever retrieves them (ADR-0051).

For a query q, the k concepts most similar to q (CLIP cosine) get prizes k, k-1, ..., 1;
every edge of the concept graph costs the same constant C. A Prize-Collecting Steiner
Tree keeps a connected set of concepts that collects much prize for little edge cost,
so a zero-prize concept is kept when it is the cheap way to join two prized ones (a
"bridge"). This mirrors retrieval_via_pcst of the G-Retriever code; our edges have no
text, so edges get no prizes (the one difference, written in ADR-0051).
"""

from dataclasses import dataclass

import numpy as np
from pcst_fast import pcst_fast

# pcst_fast arguments used by the G-Retriever code: unrooted, one tree, GW pruning.
UNROOTED = -1
NUM_TREES = 1
PRUNING = "gw"
QUIET = 0


@dataclass(frozen=True)
class QuerySubgraph:
    """concepts: sorted concept rows of S_q; prizes: the prize of every concept row."""

    concepts: np.ndarray  # int64
    tree_pairs: np.ndarray  # int64, shape (num_tree_edges, 2), edges of the tree
    prizes: np.ndarray  # float64, shape (num_concepts,)


def undirected_pairs(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """(a, b) with a < b, one per unordered pair, sorted. PCST works on undirected edges."""
    pairs = set()
    for a, b in zip(source.tolist(), target.tolist(), strict=True):
        if a != b:
            pairs.add((min(a, b), max(a, b)))
    return np.array(sorted(pairs), dtype=np.int64).reshape(-1, 2)


def unit_rows(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.linalg.norm(matrix, axis=-1, keepdims=True)


def rank_prizes(similarity: np.ndarray, k: int) -> np.ndarray:
    """Prize k for the most similar concept, k-1 for the next, ..., 0 outside the top k.

    Ties are broken by concept row (stable sort) so the result is deterministic.
    """
    order = np.argsort(-similarity, kind="stable")
    prizes = np.zeros(len(similarity), dtype=np.float64)
    top = order[: min(k, len(similarity))]
    prizes[top] = np.arange(len(top), 0, -1, dtype=np.float64)
    return prizes


def query_prizes(query: np.ndarray, concept_vectors: np.ndarray, k: int) -> np.ndarray:
    similarity = unit_rows(concept_vectors) @ (query / np.linalg.norm(query))
    return rank_prizes(similarity, k)


def query_subgraph(
    query: np.ndarray, concept_vectors: np.ndarray, pairs: np.ndarray, k: int, edge_cost: float
) -> QuerySubgraph:
    """S_q: the PCST tree on the concept graph for this query."""
    prizes = query_prizes(query, concept_vectors, k)
    costs = np.full(len(pairs), edge_cost, dtype=np.float64)
    vertices, edge_rows = pcst_fast(pairs, prizes, costs, UNROOTED, NUM_TREES, PRUNING, QUIET)
    tree_pairs = pairs[edge_rows]
    # As in the G-Retriever code: the subgraph is the chosen vertices plus both ends of
    # every chosen edge.
    concepts = np.unique(np.concatenate([vertices, tree_pairs.ravel()])).astype(np.int64)
    return QuerySubgraph(concepts, tree_pairs.astype(np.int64), prizes)


def no_tree_subgraph(query: np.ndarray, concept_vectors: np.ndarray, k: int) -> QuerySubgraph:
    """Ablation 1 (ADR-0051): only the prized concepts, no bridges, no tree."""
    prizes = query_prizes(query, concept_vectors, k)
    concepts = np.flatnonzero(prizes > 0).astype(np.int64)
    return QuerySubgraph(concepts, np.zeros((0, 2), dtype=np.int64), prizes)
