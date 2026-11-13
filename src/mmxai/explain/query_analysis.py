"""5 success + 5 failure queries on VAL, each explained on the trained model (PDF section 7).

Steps (logic of scripts/explain.py):
    1. Load the weights and the alpha that training saved (selection.json). Choose nothing.
    2. Score val exactly as training did: split_candidates, image_vectors,
       candidate_graph_scores, then fuse with the saved alpha.
    3. Per query: rank of the first relevant image under CLIP and under CLIP + Graph.
    4. Pick 5 success and 5 failure queries (rule: mmxai.eval.query_ranks, fixed beforehand).
    5. Explain the first relevant image in the CLIP + Graph order and, for a failure, also
       the image CLIP + Graph put first: Occlusion on the trained model, then the deletion
       check (mmxai.explain.ranking_model).

Decision (ADR-0009): val only. Test runs once per experiment and seed, in
scripts/evaluate_graph_rerank.py; re-scoring it here would need a team decision.
"""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from mmxai.data.visual_genome import load_region_texts
from mmxai.eval.graph_rerank import (
    MODEL_FILE,
    SELECTION_FILE,
    GraphRerankConfig,
    SplitGraph,
    build_split_graph,
    image_vectors,
    library_versions,
    load_inputs,
    make_model,
    seed_model_dir,
    split_candidates,
)
from mmxai.eval.metrics import compute_metrics
from mmxai.eval.query_ranks import clip_ranks, fused_ranks, ranked_columns, select_queries
from mmxai.explain.occlusion import EdgeContribution
from mmxai.explain.presentation import Explanation, Relation, build_explanation
from mmxai.explain.ranking_model import CandidateExplanation, explain_candidate
from mmxai.rerank.fusion import Candidates, candidate_graph_scores, fuse_scores
from mmxai.utils.runs import collect_env, make_run_dir

QUERIES_PER_GROUP = 5
# A concept -> image edge is the contains edge (ADR-0045) read from the concept's side.
RELATION_NAME = "in"
ANALYSIS_JSON = "analysis.json"
ANALYSIS_MD = "analysis.md"


def load_query_texts(vg_data_dir: str | Path) -> dict[int, str]:
    """region_id -> phrase, from Visual Genome's region_descriptions.json.

    The feature cache keeps only region ids ("text_ids"), not the phrases, so the
    text has to be read from the raw file (several GB).
    """
    texts = {}
    for record in load_region_texts(Path(vg_data_dir)):
        texts[record.text_id] = record.text
    return texts


def analyse_split(
    config: GraphRerankConfig,
    seed: int,
    split: str,
    num_edges: int,
    vg_data_dir: str | Path | None,
    log=print,
) -> dict:
    """Return every number of the analysis as a JSON-ready dict.

    vg_data_dir: folder with region_descriptions.json for the query text, or None.
    """
    if split != "val":
        raise ValueError(
            f"--split {split} is not allowed: only val. Test runs once, in "
            "scripts/evaluate_graph_rerank.py (ADR-0009); explaining test needs a team decision"
        )
    if config.concept_edges_dir is not None:
        raise ValueError("only the one-layer model (concept_edges_dir: null) can be explained")

    query_texts = None
    if vg_data_dir is not None:
        query_texts = load_query_texts(vg_data_dir)

    # 1. What training saved: weights and alpha. Nothing is chosen here.
    model_dir = seed_model_dir(config, seed)
    with open(model_dir / SELECTION_FILE, encoding="utf-8") as file:
        alpha = float(json.load(file)["alpha"])
    inputs = load_inputs(config)
    split_graph = build_split_graph(inputs, split)
    model = make_model(config, split_graph.x.shape[1])
    model.load_state_dict(torch.load(model_dir / MODEL_FILE, weights_only=True))

    # 2. The same scores as train_and_select / evaluate_test.
    candidates = split_candidates(config, inputs, split)
    h = image_vectors(model, split_graph)
    graph_scores = candidate_graph_scores(candidates, h, split_graph.image_ids)
    final_scores = fuse_scores(candidates.clip_scores, graph_scores, alpha)

    # 3. and 4. Per-query ranks and the pre-declared pick.
    clip = clip_ranks(candidates)
    fused = fused_ranks(candidates, graph_scores, alpha)
    success, failure = select_queries(clip, fused, QUERIES_PER_GROUP)
    for group, picked in (("success", success), ("failure", failure)):
        if len(picked) < QUERIES_PER_GROUP:
            log(f"warning: only {len(picked)} {group} queries exist (wanted {QUERIES_PER_GROUP})")

    if "text_ids" not in inputs.texts:
        raise ValueError(f"{config.features_dir} has no text_ids (Visual Genome region ids)")
    # Query rows: the same mask retrieve_candidates uses, so row k is query k.
    is_split_text = np.isin(inputs.texts["image_ids"], inputs.split[split])
    query_text_ids = inputs.texts["text_ids"][is_split_text]
    query_image_ids = inputs.texts["image_ids"][is_split_text]

    row_of_id = {}
    for row, image_id in enumerate(split_graph.image_ids.tolist()):
        row_of_id[image_id] = row

    # 5. Explanations.
    queries = []
    for group, picked in (("success", success), ("failure", failure)):
        for query in picked:
            order = ranked_columns(final_scores[query])
            columns = [("first relevant image", first_relevant_column(order, candidates, query))]
            if group == "failure":
                columns.append(("rank 1 under CLIP+Graph", int(order[0])))
            candidate_rows = np.array(
                [row_of_id[image_id] for image_id in candidates.candidate_ids[query].tolist()]
            )

            explained = []
            for role, column in columns:
                log(f"{group} query {query}: explaining {role}")
                result = explain_candidate(
                    model,
                    split_graph,
                    candidates.query_embeddings[query],
                    candidate_rows,
                    candidates.clip_scores[query],
                    graph_scores[query],
                    alpha,
                    column,
                    num_edges,
                )
                explanation = build_explanation(
                    result.contributions,
                    num_edges,
                    node_names(result.contributions, split_graph, inputs.graph.concept_keys),
                    relation_names(result.contributions),
                    alpha,
                    min_contribution=0.0,
                )
                explained.append(candidate_record(role, result, explanation))

            text_id = int(query_text_ids[query])
            record = {
                "group": group,
                "query_index": query,
                "text_id": text_id,
                "text": query_text(query_texts, text_id),
                "source_image_id": int(query_image_ids[query]),
                "relevant_image_ids": sorted(candidates.relevant[query]),
                "clip_rank": clip[query],
                "graph_rank": fused[query],
                "rank_change": clip[query] - fused[query],
                "explained": explained,
            }
            queries.append(record)

    return {
        "settings": {
            "experiment": config.experiment,
            "seed": seed,
            "split": split,
            "alpha": alpha,
            "num_edges": num_edges,
            "queries_per_group": QUERIES_PER_GROUP,
            "explainer": "occlusion",
            "selection_rule": selection_rule_text(),
        },
        "summary": rank_summary(config, clip, fused),
        "queries": queries,
    }


def query_text(query_texts: dict[int, str] | None, text_id: int) -> str | None:
    """The region phrase, or None when the raw text was not loaded."""
    if query_texts is None:
        return None
    return query_texts.get(text_id)


def first_relevant_column(order: np.ndarray, candidates: Candidates, query: int) -> int:
    """The candidate column of the first relevant image in this order."""
    for column in order.tolist():
        if int(candidates.candidate_ids[query, column]) in candidates.relevant[query]:
            return column
    raise ValueError(f"query {query} has no relevant candidate")


def node_names(
    contributions: list[EdgeContribution], split_graph: SplitGraph, concept_keys: np.ndarray
) -> dict[int, str]:
    """Names of the nodes in these edges: "image <id>" or the concept's synset."""
    num_images = len(split_graph.image_ids)
    names = {}
    for edge in contributions:
        for node in (edge.source, edge.target):
            if node < num_images:
                names[node] = f"image {int(split_graph.image_ids[node])}"
            else:
                names[node] = str(concept_keys[node - num_images])  # concepts come after images
    return names


def relation_names(contributions: list[EdgeContribution]) -> dict[tuple[int, int], str]:
    names = {}
    for edge in contributions:
        names[(edge.source, edge.target)] = RELATION_NAME
    return names


def relation_record(relation: Relation) -> dict:
    return {
        "concept": relation.source,
        "relation": relation.relation,
        "image": relation.target,
        "effect_on_final_score": relation.contribution,
    }


def candidate_record(role: str, result: CandidateExplanation, explanation: Explanation) -> dict:
    """One explained image as plain numbers and names."""
    return {
        "role": role,
        "image_id": result.image_id,
        "clip_rank": result.clip_rank,
        "graph_rank": result.rank,
        "clip_score": result.clip_score,
        "graph_score": result.graph_score,
        "final_score": result.final_score,
        # Sparsity: the explanation shows num_edges of this many edges.
        "num_edges_into_image": len(result.contributions),
        "graph_score_drop_per_edge": [
            {"source": edge.source, "target": edge.target, "drop": edge.contribution}
            for edge in result.contributions
        ],
        "supporting": [relation_record(item) for item in explanation.supporting],
        "opposing": [relation_record(item) for item in explanation.opposing],
        "deletion_check": {
            "removed": [
                {"source": edge.source, "target": edge.target} for edge in result.deletion.removed
            ],
            "graph_score_before": result.graph_score,
            "graph_score_after": result.deletion.graph_score_after,
            "final_score_before": result.final_score,
            "final_score_after": result.deletion.final_score_after,
            "graph_rank_before": result.rank,
            "graph_rank_after": result.deletion.rank_after,
        },
    }


def rank_summary(config: GraphRerankConfig, clip: list, fused: list) -> dict:
    """How many queries moved up, down or not at all, plus both models' metrics."""
    improved = 0
    worsened = 0
    unchanged = 0
    no_candidate = 0
    for clip_rank, fused_rank in zip(clip, fused, strict=True):
        if clip_rank is None:
            no_candidate += 1
        elif fused_rank < clip_rank:
            improved += 1
        elif fused_rank > clip_rank:
            worsened += 1
        else:
            unchanged += 1
    return {
        "num_queries": len(clip),
        "relevant_not_in_candidates": no_candidate,
        "improved": improved,
        "worsened": worsened,
        "unchanged": unchanged,
        "clip_metrics": compute_metrics(clip, config.recall_ks, config.top_k),
        "graph_metrics": compute_metrics(fused, config.recall_ks, config.top_k),
    }


def selection_rule_text() -> str:
    return (
        "Only queries whose relevant image is in CLIP's top-k. change = CLIP rank - "
        "CLIP+Graph rank of the first relevant image. Success: change > 0, largest first; "
        "failure: change < 0, most negative first; ties: smaller query index first. "
        f"{QUERIES_PER_GROUP} of each, fewer if not enough exist."
    )


def format_markdown(analysis: dict) -> str:
    """The analysis as a page a person can read; every number is also in analysis.json."""
    settings = analysis["settings"]
    summary = analysis["summary"]
    lines = [
        f"# Query analysis: {settings['experiment']}, seed {settings['seed']}, {settings['split']}",
        "",
        f"- alpha (chosen on val by training): {settings['alpha']}",
        f"- explainer: {settings['explainer']} on the trained model; "
        f"top {settings['num_edges']} relations per side",
        f"- selection rule: {settings['selection_rule']}",
        f"- queries: {summary['num_queries']}; improved {summary['improved']}, "
        f"worsened {summary['worsened']}, unchanged {summary['unchanged']}, "
        f"relevant image not in candidates {summary['relevant_not_in_candidates']}",
        "",
        "Contributions are the change of the final score when one edge is removed",
        "((1 - alpha) x drop of cos(q, h_i)). The deletion check removes the top supporting",
        "edges together and re-ranks the query's candidates.",
    ]
    if settings["alpha"] == 1.0:
        lines += ["", "alpha = 1: the graph plays no part, so no relation explains the ranking."]
    for group in ("success", "failure"):
        lines += ["", f"## {group.capitalize()} queries", ""]
        group_queries = [item for item in analysis["queries"] if item["group"] == group]
        if not group_queries:
            lines.append("(none)")
        for item in group_queries:
            lines += format_query(item)
    return "\n".join(lines) + "\n"


def format_query(item: dict) -> list[str]:
    text = item["text"]
    if text is None:
        text = f"(text not loaded; region id {item['text_id']})"
    lines = [
        f'### Query {item["query_index"]}: "{text}"',
        "",
        f"Source image {item['source_image_id']}. First relevant image: CLIP rank "
        f"{item['clip_rank']} -> CLIP+Graph rank {item['graph_rank']} "
        f"(change {item['rank_change']:+d}).",
    ]
    for candidate in item["explained"]:
        deletion = candidate["deletion_check"]
        lines += [
            "",
            f"**{candidate['role']}: image {candidate['image_id']}** "
            f"(CLIP rank {candidate['clip_rank']}, CLIP+Graph rank {candidate['graph_rank']}; "
            f"clip {candidate['clip_score']:.4f}, graph {candidate['graph_score']:.4f}, "
            f"final {candidate['final_score']:.4f}; {candidate['num_edges_into_image']} edges)",
            "",
            "Pushed the image up:",
        ]
        lines += format_relations(candidate["supporting"])
        lines.append("Pulled the image down:")
        lines += format_relations(candidate["opposing"])
        lines.append(
            f"Deletion check (remove {len(deletion['removed'])} supporting edges): graph score "
            f"{deletion['graph_score_before']:.4f} -> {deletion['graph_score_after']:.4f}, "
            f"CLIP+Graph rank {deletion['graph_rank_before']} -> {deletion['graph_rank_after']}."
        )
    lines.append("")
    return lines


def format_relations(relations: list[dict]) -> list[str]:
    if not relations:
        return ["- (none)"]
    return [
        f"- {item['concept']} -[{item['relation']}]-> {item['image']} "
        f"({item['effect_on_final_score']:+.4f})"
        for item in relations
    ]


def write_analysis(
    config: GraphRerankConfig, analysis: dict, cli_settings: dict, now: datetime
) -> Path:
    """Write results/runs/<YYYYMMDD-HHMM>_<experiment>_explain_seed<k>/.

    No metrics.json on purpose: scripts/summarize_runs.py reads every folder that has
    one, and this val folder would look like a second val run of the same seed.
    """
    seed = analysis["settings"]["seed"]
    run_dir = make_run_dir(config.output_dir, f"{config.experiment}_explain", seed, now)
    run_config = asdict(config)
    run_config["explain"] = cli_settings
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(run_config, file, sort_keys=False)
    with open(run_dir / "env.json", "w", encoding="utf-8") as file:
        json.dump(collect_env(library_versions()), file, indent=2)
    with open(run_dir / ANALYSIS_JSON, "w", encoding="utf-8") as file:
        json.dump(analysis, file, indent=2)
    with open(run_dir / ANALYSIS_MD, "w", encoding="utf-8") as file:
        file.write(format_markdown(analysis))
    return run_dir
