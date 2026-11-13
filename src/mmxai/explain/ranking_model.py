"""Explain one candidate of one query on the trained model that ranked it (ADR-0016).

Faithful by construction of the score, not of the explanation:
    - graph_scores_with_edges re-runs the trained model (image_vectors, eval mode,
      the same SAGEConv path used for ranking) on the split graph with some edges
      removed, then takes cos(q, h_i) as candidate_graph_scores does.
    - explain_candidate first checks that, with every edge kept, this gives the very
      graph scores the ranking used; then Occlusion removes one edge at a time.
    - The deletion check removes the top supporting edges together and re-ranks the
      query's candidates with the saved alpha: the intervention of fidelity (ADR-0016 §4).

Only the one-layer ADR-0046 model is supported: its edges are concept -> image only
(one direction, ADR-0046 §5), so an image's score depends on its own edges alone.
"""

import dataclasses
from dataclasses import dataclass

import numpy as np
import torch

from mmxai.eval.graph_rerank import SplitGraph, image_vectors
from mmxai.eval.query_ranks import rank_of_column
from mmxai.explain.intervention import mark_remaining_edges
from mmxai.explain.occlusion import EdgeContribution, occlusion
from mmxai.models.sage import ConceptToImageSage
from mmxai.rerank.fusion import fuse_scores

# The scores of the full graph must match the ranking's up to float32 rounding.
SCORE_TOLERANCE = 1e-5


@dataclass(frozen=True)
class DeletionCheck:
    """The candidate after removing its top supporting edges together."""

    removed: list[EdgeContribution]
    graph_score_after: float
    final_score_after: float
    rank_after: int


@dataclass(frozen=True)
class CandidateExplanation:
    """Why one candidate got its graph score, and what happens without its best edges."""

    image_id: int
    clip_rank: int
    rank: int  # under CLIP + Graph
    clip_score: float
    graph_score: float
    final_score: float
    contributions: list[EdgeContribution]  # every edge into the image, largest drop first
    deletion: DeletionCheck


def graph_scores_with_edges(
    model: ConceptToImageSage,
    split_graph: SplitGraph,
    remaining: np.ndarray,
    query: np.ndarray,
    image_rows: np.ndarray,
) -> np.ndarray:
    """cos(q, h_i) for the given image rows, computed with only the remaining edges.

    remaining marks, for each column of split_graph.edge_index, whether the edge is kept.
    With every edge kept this is the graph score used for ranking.
    """
    kept_columns = torch.from_numpy(remaining)
    kept_graph = dataclasses.replace(
        split_graph, edge_index=split_graph.edge_index[:, kept_columns]
    )
    h = image_vectors(model, kept_graph)[image_rows]
    unit_h = h / np.linalg.norm(h, axis=1, keepdims=True)
    unit_q = query / np.linalg.norm(query)
    return unit_h @ unit_q


def explain_candidate(
    model: ConceptToImageSage,
    split_graph: SplitGraph,
    query: np.ndarray,
    candidate_rows: np.ndarray,
    clip_scores: np.ndarray,
    ranking_graph_scores: np.ndarray,
    alpha: float,
    column: int,
    num_edges: int,
) -> CandidateExplanation:
    """Explain candidate `column` of one query with Occlusion, then run the deletion check.

    candidate_rows: graph rows of the query's candidates, in CLIP's order.
    clip_scores, ranking_graph_scores: that query's rows of the scores the ranking used.
    num_edges: how many top supporting edges the deletion check removes.
    Raises ValueError if the model does not reproduce the ranking's graph scores.
    """
    if not isinstance(model, ConceptToImageSage):
        raise ValueError("only the one-layer ConceptToImageSage model (ADR-0046) is supported")
    if num_edges < 1:
        raise ValueError(f"num_edges must be at least 1, got {num_edges}")

    edge_index = split_graph.edge_index.numpy()
    every_edge = np.ones(edge_index.shape[1], dtype=bool)
    graph_before = graph_scores_with_edges(model, split_graph, every_edge, query, candidate_rows)
    # Guard: if this is not the score that ranked, the explanation would explain something else.
    if not np.allclose(graph_before, ranking_graph_scores, atol=SCORE_TOLERANCE):
        raise ValueError("the model does not reproduce the graph scores used for ranking")
    final_before = fuse_scores(clip_scores, graph_before, alpha)

    image_row = int(candidate_rows[column])

    def score_image(remaining: np.ndarray) -> float:
        one_row = np.array([image_row])
        return float(graph_scores_with_edges(model, split_graph, remaining, query, one_row)[0])

    # One direction: removing a concept -> image edge removes exactly one column (ADR-0046 §5).
    contributions = occlusion(
        edge_index, image_row, model.num_layers, both_directions=False, score_image=score_image
    )

    supporting = []
    for edge in contributions:
        if edge.contribution > 0 and len(supporting) < num_edges:
            supporting.append(edge)
    removed_pairs = [(edge.source, edge.target) for edge in supporting]
    remaining = mark_remaining_edges(edge_index, removed_pairs, both_directions=False)
    # Re-score every candidate of the query, not only this one, and re-rank with the saved alpha.
    graph_after = graph_scores_with_edges(model, split_graph, remaining, query, candidate_rows)
    final_after = fuse_scores(clip_scores, graph_after, alpha)

    deletion = DeletionCheck(
        removed=supporting,
        graph_score_after=float(graph_after[column]),
        final_score_after=float(final_after[column]),
        rank_after=rank_of_column(final_after, column),
    )
    return CandidateExplanation(
        image_id=int(split_graph.image_ids[image_row]),
        clip_rank=column + 1,  # candidates are stored in CLIP's order
        rank=rank_of_column(final_before, column),
        clip_score=float(clip_scores[column]),
        graph_score=float(graph_before[column]),
        final_score=float(final_before[column]),
        contributions=contributions,
        deletion=deletion,
    )
