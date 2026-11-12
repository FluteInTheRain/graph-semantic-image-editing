"""Re-rank CLIP's top-k candidates with a graph score (PDF section 4.2, ADR-0046).

    final = alpha * cos(q, x_i) + (1 - alpha) * graph_score(q, i)

Both scores are cosines, so they are on the same scale. alpha = 1 is plain CLIP,
alpha = 0 is the graph score alone. Needs numpy and FAISS only.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from mmxai.eval.metrics import compute_metrics, first_relevant_rank
from mmxai.eval.relevance import same_text_relevant_images
from mmxai.index.image_index import build_image_index, search_images


@dataclass(frozen=True)
class Candidates:
    """CLIP's top-k images for every query of one split, best first."""

    query_embeddings: np.ndarray  # (num_queries, dim)
    candidate_ids: np.ndarray  # (num_queries, top_k) image ids
    clip_scores: np.ndarray  # (num_queries, top_k) cos(q, x_i), decreasing along a row
    relevant: list[set[int]]  # per query, the images that count as correct (ADR-0043)


def retrieve_candidates(
    image_embeddings: np.ndarray,
    image_ids: np.ndarray,
    text_embeddings: np.ndarray,
    text_image_ids: np.ndarray,
    eval_image_ids: np.ndarray,
    top_k: int,
    same_text_threshold: float,
) -> Candidates:
    """Search the split's images with the split's texts, as the CLIP baseline does (ADR-0043).

    Plain CLIP and CLIP + Graph both call this function, so they re-rank exactly the
    same candidates and are judged by exactly the same relevant images.
    """
    missing = np.setdiff1d(eval_image_ids, image_ids)
    if len(missing) > 0:
        raise ValueError(f"{len(missing)} split images have no embedding, e.g. {missing[0]}")

    # Both the gallery and the queries come from the evaluated split only.
    image_rows = np.isin(image_ids, eval_image_ids)
    text_rows = np.isin(text_image_ids, eval_image_ids)
    query_embeddings = text_embeddings[text_rows]
    query_image_ids = text_image_ids[text_rows]

    index = build_image_index(image_embeddings[image_rows], image_ids[image_rows])
    clip_scores, candidate_ids = search_images(index, query_embeddings, top_k)
    relevant = same_text_relevant_images(query_embeddings, query_image_ids, same_text_threshold)
    return Candidates(query_embeddings, candidate_ids, clip_scores, relevant)


def candidate_graph_scores(
    candidates: Candidates,
    image_h: np.ndarray,
    h_image_ids: np.ndarray,
    chunk_size: int = 512,
) -> np.ndarray:
    """Return cos(q, h_i) for every candidate: shape (num_queries, top_k).

    image_h row r is the graph vector of image h_image_ids[r].
    """
    row_of_id = {}
    for row, image_id in enumerate(h_image_ids.tolist()):
        row_of_id[image_id] = row
    missing = set(np.unique(candidates.candidate_ids).tolist()) - set(row_of_id)
    if missing:
        raise ValueError(f"{len(missing)} candidates have no graph vector, e.g. {min(missing)}")

    unit_h = image_h / np.linalg.norm(image_h, axis=1, keepdims=True)
    queries = candidates.query_embeddings
    unit_q = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    candidate_rows = np.vectorize(row_of_id.__getitem__)(candidates.candidate_ids)

    scores = np.empty(candidate_rows.shape, dtype=np.float32)
    # Chunks: all queries x 100 candidates x 512 numbers at once would take gigabytes.
    for start in range(0, len(candidate_rows), chunk_size):
        stop = start + chunk_size
        chunk_h = unit_h[candidate_rows[start:stop]]  # (chunk, top_k, dim)
        scores[start:stop] = np.einsum("qkd,qd->qk", chunk_h, unit_q[start:stop])
    return scores


def fuse_scores(clip_scores: np.ndarray, graph_scores: np.ndarray, alpha: float) -> np.ndarray:
    """final = alpha * clip + (1 - alpha) * graph."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
    return alpha * clip_scores + (1.0 - alpha) * graph_scores


def rerank_ranks(candidates: Candidates, final_scores: np.ndarray) -> list[int | None]:
    """Sort each query's candidates by final score and return the first relevant rank.

    The sort is stable, so candidates with equal final scores keep CLIP's order.
    With alpha = 1 the order is therefore exactly CLIP's.
    """
    ranks = []
    for query in range(len(final_scores)):
        order = np.argsort(-final_scores[query], kind="stable")
        retrieved = candidates.candidate_ids[query, order]
        ranks.append(first_relevant_rank(retrieved.tolist(), candidates.relevant[query]))
    return ranks


def sweep_alpha(
    candidates: Candidates,
    graph_scores: np.ndarray,
    alpha_grid: Sequence[float],
    recall_ks: Sequence[int],
) -> dict[float, dict]:
    """Return the metrics of every alpha in the grid."""
    top_k = candidates.candidate_ids.shape[1]
    results = {}
    for alpha in alpha_grid:
        final = fuse_scores(candidates.clip_scores, graph_scores, float(alpha))
        results[float(alpha)] = compute_metrics(rerank_ranks(candidates, final), recall_ks, top_k)
    return results


def choose_alpha(sweep: dict[float, dict], metric: str) -> float:
    """Return the alpha with the best metric; on a tie, the larger alpha.

    Decision (ADR-0047): on a tie, prefer the alpha closer to plain CLIP.
      Why: without evidence that the graph helps, do not move away from the baseline.
      Decided by: Claude Code proposal, 2026-11-12; not yet approved by the team.
    """
    best_alpha = None
    for alpha in sorted(sweep):
        if best_alpha is None or sweep[alpha][metric] >= sweep[best_alpha][metric]:
            best_alpha = alpha
    if best_alpha is None:
        raise ValueError("the alpha sweep is empty")
    return best_alpha
