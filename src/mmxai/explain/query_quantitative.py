"""Level-3 explanation of one free-text query, measured like the level-3 evaluation.

For each of the first k images CLIP + Graph returns (PDF section 5, ADR-0016):
    1. The main explainer chosen on val (configs/explain_vg.yaml: Integrated Gradients)
       orders the image's concept -> image edges on the trained model; the top m
       (top_m, also chosen on val) are the explanation.
    2. fidelity.evaluate_explanation removes them and re-scores and re-ranks: fidelity+/-,
       rank drop, the same-size random control, sparsity and validity.

Both steps are the functions scripts/evaluate_explanations.py uses (order_relations,
evaluate_explanation), so the demo shows what the reported numbers measure. Nothing is
chosen here: explainer and m come from the explain config.
"""

from dataclasses import dataclass

import numpy as np

from mmxai.explain.config import ExplainConfig
from mmxai.explain.evaluation import order_relations
from mmxai.explain.fidelity import ExplanationReport, evaluate_explanation
from mmxai.explain.query_path import QueryEngine, rank_candidates
from mmxai.explain.sage_scoring import candidate_scores, image_scorer


@dataclass(frozen=True)
class QuantifiedImage:
    """One result image, its explanation edges and the measures of that explanation."""

    rank: int  # under CLIP + Graph, 1 = best
    image_id: int
    graph_score: float
    num_edges: int  # concept -> image edges of the image
    explanation: list[tuple[int, int]]  # (concept row, image row), strongest first
    values: list[float]  # the explainer's value of each explanation edge
    report: ExplanationReport | None  # None when alpha = 1 or m covers every edge


def explain_quantitatively(
    engine: QueryEngine, query: np.ndarray, k: int, config: ExplainConfig, seed: int
) -> list[QuantifiedImage]:
    """Explain and measure the first k images of CLIP + Graph for this query.

    seed drives the random control and robust fidelity, as in the evaluation.
    """
    if config.main_explainer is None or config.top_m is None:
        raise ValueError("main_explainer and top_m must be chosen on val first")
    if not 1 <= k <= engine.top_k:
        raise ValueError(f"k must be between 1 and top_k={engine.top_k}, got {k}")
    ranked = rank_candidates(engine, query)

    results = []
    for rank, column in enumerate(ranked.order[:k].tolist(), start=1):
        scorer = image_scorer(
            engine.model, engine.split_graph, int(ranked.image_rows[column]), query
        )
        ranked_score = float(ranked.graph_scores[column])
        ordered = order_relations(config.main_explainer, scorer, ranked_score, config, seed)
        explanation = ordered.relations[: config.top_m]

        report = None
        # As in the evaluation: an explanation that keeps every edge has fidelity- = 0 by
        # construction and says nothing, so it is not measured.
        if config.top_m < len(ordered.relations):
            report = evaluate_explanation(
                scorer.edge_index,
                scorer.image_row,
                column,
                explanation,
                candidate_scores(scorer, ranked.graph_scores, column),
                ranked.graph_scores,
                ranked.clip_scores,
                engine.alpha,
                config,
                seed,
            )
        results.append(
            QuantifiedImage(
                rank=rank,
                image_id=int(ranked.candidate_ids[column]),
                graph_score=ranked_score,
                num_edges=len(ordered.relations),
                explanation=explanation,
                values=ordered.values[: config.top_m],
                report=report,
            )
        )
    return results
