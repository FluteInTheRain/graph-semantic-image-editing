"""Evaluation reports shared by plain CLIP and CLIP + Graph (spec 2026-11-13).

One place builds the numbers of a run: Recall, MRR, nDCG, the per-group tables and the
image -> text block. Both pipelines hand in their final scores, so both are measured
by exactly the same code.
"""

from pathlib import Path

import numpy as np

from mmxai.data.visual_genome import load_region_texts
from mmxai.eval.image_to_text import TextCandidates, ranked_text_rows
from mmxai.eval.metrics import compute_metrics, ranked_metrics
from mmxai.eval.query_groups import (
    Predicates,
    metrics_by_group,
    relation_groups,
    specificity_groups,
)
from mmxai.rerank.fusion import Candidates, fuse_scores, rerank_ranks


def check_extension_settings(
    ndcg_k: int | None,
    image_to_text: bool | None,
    vg_data_dir: str | None,
    predicates_file: str | None,
) -> None:
    """Reject settings that ask for a report part whose inputs are missing."""
    wants_groups = vg_data_dir is not None or predicates_file is not None
    if wants_groups and (vg_data_dir is None or predicates_file is None):
        raise ValueError("groups need both vg_data_dir and predicates_file")
    if (wants_groups or image_to_text) and ndcg_k is None:
        raise ValueError("groups and image_to_text need ndcg_k")


def ranked_image_ids(candidates: Candidates, final_scores: np.ndarray) -> list[list[int]]:
    """Return each query's candidate image ids sorted by final score, best first.

    The sort is stable like in rerank_ranks, so equal scores keep CLIP's order.
    """
    rankings = []
    for query in range(len(final_scores)):
        order = np.argsort(-final_scores[query], kind="stable")
        rankings.append(candidates.candidate_ids[query, order].tolist())
    return rankings


def text_to_image_report(
    candidates: Candidates,
    final_scores: np.ndarray,
    groupings: dict[str, list[str]] | None,
    recall_ks: list[int],
    top_k: int,
    ndcg_k: int,
) -> dict:
    """Metrics of a text -> image ranking, plus per-group metrics when groupings are given."""
    rankings = ranked_image_ids(candidates, final_scores)
    report = ranked_metrics(rankings, candidates.relevant, recall_ks, top_k, ndcg_k)

    # Safety net: the report must agree with the ranks the rest of the pipeline uses.
    old_style = compute_metrics(rerank_ranks(candidates, final_scores), recall_ks, top_k)
    for key, value in old_style.items():
        if report[key] != value:
            raise ValueError(f"report and pipeline disagree on {key}: {report[key]} vs {value}")

    if groupings is not None:
        report["groups"] = {}
        for name, labels in groupings.items():
            report["groups"][name] = metrics_by_group(
                rankings, candidates.relevant, labels, recall_ks, top_k, ndcg_k
            )
    return report


def image_to_text_report(
    candidates: TextCandidates,
    final_scores: np.ndarray,
    recall_ks: list[int],
    top_k: int,
    ndcg_k: int,
) -> dict:
    """Metrics of an image -> text ranking (the queries are images, the gallery is texts)."""
    rankings = ranked_text_rows(candidates, final_scores)
    return ranked_metrics(rankings, candidates.relevant, recall_ks, top_k, ndcg_k)


def sweep_alpha_image_to_text(
    candidates: TextCandidates,
    graph_scores: np.ndarray,
    alpha_grid: list[float],
    recall_ks: list[int],
    top_k: int,
    ndcg_k: int,
) -> dict[float, dict]:
    """Return the image -> text metrics of every alpha in the grid (like sweep_alpha)."""
    results = {}
    for alpha in alpha_grid:
        final = fuse_scores(candidates.clip_scores, graph_scores, float(alpha))
        results[float(alpha)] = image_to_text_report(candidates, final, recall_ks, top_k, ndcg_k)
    return results


def query_groupings(
    candidates: Candidates,
    text_ids: np.ndarray,
    vg_data_dir: str,
    predicates: Predicates,
) -> dict[str, list[str]]:
    """Label every text -> image query by specificity and by relation type.

    text_ids are the region ids of the split's queries, in the order of candidates.
    The cache keeps only ids, so the sentences come from the raw Visual Genome file.
    """
    region_texts = {}
    for record in load_region_texts(Path(vg_data_dir)):
        region_texts[record.text_id] = record.text
    return {
        "specificity": specificity_groups(candidates.relevant),
        "relation": relation_groups(text_ids.tolist(), region_texts, predicates),
    }
