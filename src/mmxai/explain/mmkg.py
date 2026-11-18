"""Level-3 explanation evaluation on the Vietnamese MMKG (ADR-0054).

The same measures as on Visual Genome (mmxai.explain.evaluation), on the MMKG graph of
ADR-0053, with every choice taken from VG val: the one-layer VG model and its alpha,
Integrated Gradients, m = 10. Per seed:
    1. Rank all queries; stop unless the MRR equals the committed transfer run (rule 6).
    2. For every query, explain the image ranked first and measure the explanation
       (fidelity+/-, rank drop vs the random control, robust fidelity, sparsity, validity).
Then summarize overall and per query group, mean +- std over seeds, Wilcoxon vs random.
"""

from dataclasses import asdict

import numpy as np

from mmxai.eval.metrics import compute_metrics
from mmxai.eval.mmkg_transfer import (
    TransferGraph,
    build_transfer_graph,
    clip_candidates,
    image_graph_vectors,
    query_groups,
)
from mmxai.eval.query_ranks import ranked_columns
from mmxai.explain.config import ExplainConfig
from mmxai.explain.evaluation import order_relations, summarize
from mmxai.explain.fidelity import evaluate_explanation
from mmxai.explain.sage_scoring import candidate_scores, image_scorer
from mmxai.rerank.fusion import candidate_graph_scores, fuse_scores, rerank_ranks

# Tolerance of the reproduction check: the transfer run stored MRR as a float in JSON.
REPRODUCE_TOLERANCE = 1e-9


def check_reproduces_transfer(ranks: list[int | None], top_k: int, expected_mrr: float) -> None:
    """Stop if this ranking is not the one the transfer run measured (ADR-0054 rule 6).

    Explaining a different ranking than the reported one would explain the wrong model.
    """
    mrr = compute_metrics(ranks, [1], top_k)[f"MRR@{top_k}"]
    if abs(mrr - expected_mrr) > REPRODUCE_TOLERANCE:
        raise ValueError(f"ranking gives MRR {mrr}, the transfer run recorded {expected_mrr}")


def explain_seed(
    model,
    alpha: float,
    seed: int,
    graph: TransferGraph,
    candidates,
    queries: list[dict],
    concept_keys: list[str],
    config: ExplainConfig,
    expected_mrr: float | None,
    log=print,
) -> list[dict]:
    """One record per query: the rank-1 image's IG explanation and its measures."""
    num_images = len(graph.image_ids)
    row_of_id = {image_id: row for row, image_id in enumerate(graph.image_ids.tolist())}
    h = image_graph_vectors(model, graph)
    graph_scores = candidate_graph_scores(candidates, h, graph.image_ids)
    final = fuse_scores(candidates.clip_scores, graph_scores, alpha)
    if expected_mrr is not None:
        top_k = candidates.candidate_ids.shape[1]
        check_reproduces_transfer(rerank_ranks(candidates, final), top_k, expected_mrr)

    labels = query_groups(queries)
    records = []
    for query_row, query in enumerate(queries):
        column = int(ranked_columns(final[query_row])[0])  # the image shown first
        image_id = int(candidates.candidate_ids[query_row, column])
        scorer = image_scorer(
            model, graph, row_of_id[image_id], candidates.query_embeddings[query_row]
        )
        ranked_score = float(graph_scores[query_row, column])
        ordered = order_relations(config.main_explainer, scorer, ranked_score, config, seed)
        if config.top_m >= len(ordered.relations):
            # Same rule as on VG: an explanation must leave some edge out.
            log(f"seed {seed}, {query['qid']}: m={config.top_m} covers every edge, skipped")
            continue
        explanation = ordered.relations[: config.top_m]
        report = evaluate_explanation(
            scorer.edge_index,
            scorer.image_row,
            column,
            explanation,
            candidate_scores(scorer, graph_scores[query_row], column),
            graph_scores[query_row],
            candidates.clip_scores[query_row],
            alpha,
            config,
            seed,
        )
        if report is None:  # alpha = 1: the graph plays no part in the ranking
            continue
        records.append(
            {
                "seed": seed,
                "query_row": query_row,
                "qid": query["qid"],
                "group": labels["type_language"][query_row],
                "image_id": image_id,
                "image_is_correct": image_id in candidates.relevant[query_row],
                "clip_rank": column + 1,  # candidates are stored in CLIP's order
                "alpha": alpha,
                "explainer": config.main_explainer,
                "top_m": config.top_m,
                "relations": [list(edge) for edge in explanation],
                "concepts": [concept_keys[source - num_images] for source, _ in explanation],
                "explainer_values": ordered.values[: config.top_m],
                "completeness_gap": ordered.completeness_gap,
                **asdict(report),
            }
        )
    return records


def summarize_mmkg(records: list[dict], config: ExplainConfig) -> dict:
    """The VG summary, overall and for every query group (direct/indirect x en/vi)."""
    explainers = [config.main_explainer]
    top_m_options = [config.top_m]
    result = {"all": summarize(records, "test", explainers, top_m_options, config.seeds)}
    for group in sorted({record["group"] for record in records}):
        subset = [record for record in records if record["group"] == group]
        result[group] = summarize(subset, "test", explainers, top_m_options, config.seeds)
    return result


def table_rows(summary: dict, run_name: str) -> list[dict]:
    """Flatten the summary into rows for results/tables/xai_mmkg.csv."""
    rows = []
    for group, group_summary in summary.items():
        for row in group_summary["rows"]:
            flat = {
                "group": group,
                "explainer": row["explainer"],
                "top_m": row["top_m"],
                "num_records": row["num_records"],
            }
            for name, value in row.items():
                if isinstance(value, dict):
                    flat[f"{name}_mean"] = value["mean"]
                    flat[f"{name}_std"] = value["std"]
                elif name not in flat:
                    flat[name] = value
            flat["run"] = run_name
            rows.append(flat)
    return rows


def mmkg_graph_and_candidates(
    transfer_config: dict, arrays: dict[str, np.ndarray], queries: list[dict], triples
):
    """The ADR-0053 graph and CLIP candidates, built exactly as the transfer run did."""
    if [query["qid"] for query in queries] != arrays["query_qids"].tolist():
        raise ValueError("the feature cache was built from a different query file")
    image_ids = arrays["image_ids"]
    id_of_key = dict(zip(arrays["image_keys"].tolist(), image_ids.tolist(), strict=True))
    relevant = [{id_of_key[key] for key in query["relevant"]} for query in queries]
    candidates = clip_candidates(
        arrays["query_features"],
        arrays["image_features"],
        image_ids,
        relevant,
        transfer_config["top_k"],
    )
    graph = build_transfer_graph(
        image_ids,
        arrays["image_features"],
        arrays["concept_keys"].tolist(),
        arrays["concept_features"],
        triples,
        transfer_config["clip_topk_k"],
    )
    return graph, candidates
