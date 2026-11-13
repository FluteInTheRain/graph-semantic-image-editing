"""Level-3 explanation evaluation on the chosen CLIP + Graph model (PDF section 5, ADR-0016).

Per seed, with that seed's trained weights and val-chosen alpha:
    1. Rank every query of the split exactly as the pipeline does. On val, stop unless the
       ranks give the val metric training saved (query_analysis.check_reproduces_training).
    2. For each picked query, take the image CLIP + Graph ranks first.
    3. Each explainer orders that image's edges on the trained model: Occlusion by score
       drop, Integrated Gradients by attribution, GNNExplainer by mask value. The top m
       edges are the explanation, for every m in top_m_options.
    4. fidelity.evaluate_explanation measures it: fidelity+/-, rank drop, robust fidelity,
       the same-size random control, sparsity and validity.
Then summarize per (explainer, m): mean ± std over seeds of the per-seed mean, and a
one-sided Wilcoxon test "explanation beats the random control" over queries. On val the
(explainer, m) with the best characterization is proposed; test reuses it, once.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from mmxai.eval.graph_rerank import (
    MODEL_FILE,
    SELECTION_FILE,
    GraphRerankConfig,
    build_split_graph,
    image_vectors,
    library_versions,
    load_graph_rerank_config,
    load_inputs,
    make_model,
    seed_model_dir,
    split_candidates,
)
from mmxai.eval.query_ranks import fused_ranks, ranked_columns
from mmxai.explain.config import ExplainConfig
from mmxai.explain.fidelity import evaluate_explanation
from mmxai.explain.gnn_explainer import gnn_explainer
from mmxai.explain.integrated_gradients import integrated_gradients
from mmxai.explain.occlusion import occlusion
from mmxai.explain.query_analysis import check_reproduces_training
from mmxai.explain.sage_scoring import ImageScorer, candidate_scores, image_scorer
from mmxai.explain.statistics import choose_on_validation, mean_and_std, wilcoxon_greater
from mmxai.models.sage import ConceptToImageSage
from mmxai.rerank.fusion import candidate_graph_scores, fuse_scores
from mmxai.utils.runs import collect_env

EVALUATION_JSON = "evaluation.json"
SUMMARY_MD = "summary.md"
# Measures averaged in the summary; larger is better except fidelity_minus/robust_minus.
SUMMARY_MEASURES = (
    "fidelity_plus",
    "random_plus_mean",
    "rank_drop",
    "random_rank_drop_mean",
    "fidelity_minus",
    "characterization",
    "robust_plus",
    "robust_minus",
    "sparsity",
    "connected_share",
)


@dataclass(frozen=True)
class OrderedRelations:
    """One explainer's order of an image's edges, best first, with its own values."""

    relations: list[tuple[int, int]]
    values: list[float]  # score drop, attribution or mask value, in the same order
    completeness_gap: float | None  # Integrated Gradients only


def pick_query_rows(num_split_queries: int, count: int, sample_seed: int) -> np.ndarray:
    """Query rows to explain: `count` rows drawn at random once, sorted.

    The rule is fixed before any result is seen and does not look at ranks, so the
    queries are not picked to make the explanations look good.
    """
    if not 1 <= count <= num_split_queries:
        raise ValueError(f"cannot pick {count} of {num_split_queries} queries")
    rng = np.random.default_rng(sample_seed)
    return np.sort(rng.choice(num_split_queries, size=count, replace=False))


def order_relations(
    explainer: str,
    scorer: ImageScorer,
    ranked_graph_score: float,
    config: ExplainConfig,
    seed: int,
) -> OrderedRelations:
    """Run one explainer on the trained model for one image."""
    common = (scorer.edge_index, scorer.image_row, config.num_layers, config.both_directions)
    if explainer == "occlusion":
        # Occlusion does not take the ranked score; evaluate_explanation checks the same
        # score_image against it right after, before anything is recorded.
        contributions = occlusion(*common, scorer.score_image)
        gap = None
    elif explainer == "integrated_gradients":
        contributions, gap = integrated_gradients(
            *common,
            scorer.score_image,
            scorer.score_and_gradient,
            ranked_graph_score,
            config.ig_steps,
        )
    elif explainer == "gnn_explainer":
        masks = gnn_explainer(
            *common,
            scorer.score_image,
            scorer.score_and_gradient,
            ranked_graph_score,
            config.gnn_explainer_epochs,
            config.gnn_explainer_learning_rate,
            config.gnn_explainer_size_weight,
            config.gnn_explainer_entropy_weight,
            seed,
        )
        # The mask only orders the edges; fidelity measures their real effect.
        return OrderedRelations(
            [(edge.source, edge.target) for edge in masks], [edge.mask for edge in masks], None
        )
    else:
        raise ValueError(f"unknown explainer {explainer!r}")
    return OrderedRelations(
        [(edge.source, edge.target) for edge in contributions],
        [edge.contribution for edge in contributions],
        gap,
    )


def load_seed(
    ranking: GraphRerankConfig, split_graph, seed: int
) -> tuple[ConceptToImageSage, float, dict]:
    """The weights and alpha that training saved for this seed. Nothing is chosen here."""
    model_dir = seed_model_dir(ranking, seed)
    with open(model_dir / SELECTION_FILE, encoding="utf-8") as file:
        selection = json.load(file)
    model = make_model(ranking, split_graph.x.shape[1])
    model.load_state_dict(torch.load(model_dir / MODEL_FILE, weights_only=True))
    model.eval()
    return model, float(selection["alpha"]), selection


def check_settings(config: ExplainConfig, ranking: GraphRerankConfig) -> None:
    """Stop if the explain config does not describe the explained model (ADR-0046)."""
    if ranking.concept_edges_dir is not None or not ranking.use_graph:
        raise ValueError("only the one-layer CLIP + Graph model (ADR-0046) can be explained")
    if config.num_layers != ConceptToImageSage.num_layers:
        raise ValueError(f"num_layers must be {ConceptToImageSage.num_layers} for this model")
    if config.both_directions:
        raise ValueError("this model stores concept -> image edges in one direction only")


def evaluate_split(config: ExplainConfig, log=print) -> dict:
    """Every per-explanation record and the summary of one split, JSON-ready."""
    ranking = load_graph_rerank_config(config.ranking_config)
    check_settings(config, ranking)
    split = config.eval_split
    explainers = config.explainers
    top_m_options = config.top_m_options
    if split == "test":
        # load_explain_config already refuses test without a val choice; use only it.
        explainers = [config.main_explainer]
        top_m_options = [config.top_m]

    inputs = load_inputs(ranking)
    split_graph = build_split_graph(inputs, split)
    candidates = split_candidates(ranking, inputs, split)
    num_images = len(split_graph.image_ids)
    row_of_id = {image_id: row for row, image_id in enumerate(split_graph.image_ids.tolist())}
    query_rows = pick_query_rows(
        len(candidates.candidate_ids), config.num_queries, config.query_sample_seed
    )
    is_split_text = np.isin(inputs.texts["image_ids"], inputs.split[split])
    query_text_ids = inputs.texts["text_ids"][is_split_text] if "text_ids" in inputs.texts else None

    def concept_name(node: int) -> str:
        return str(inputs.graph.concept_keys[node - num_images])  # concepts come after images

    records = []
    alphas = {}
    for seed in config.seeds:
        model, alpha, selection = load_seed(ranking, split_graph, seed)
        alphas[seed] = alpha
        h = image_vectors(model, split_graph)
        graph_scores = candidate_graph_scores(candidates, h, split_graph.image_ids)
        if split == "val":
            check_reproduces_training(
                ranking, selection, alpha, fused_ranks(candidates, graph_scores, alpha)
            )
        final = fuse_scores(candidates.clip_scores, graph_scores, alpha)

        for query in query_rows.tolist():
            column = int(ranked_columns(final[query])[0])  # the image shown first
            image_id = int(candidates.candidate_ids[query, column])
            scorer = image_scorer(
                model, split_graph, row_of_id[image_id], candidates.query_embeddings[query]
            )
            ranked_score = float(graph_scores[query, column])
            log(f"seed {seed}, query {query}: explaining image {image_id}")
            for explainer in explainers:
                ordered = order_relations(explainer, scorer, ranked_score, config, seed)
                for top_m in top_m_options:
                    # An explanation must leave some edge out: with every edge kept,
                    # fidelity- is 0 and characterization 1 by construction, which would
                    # make the val choice always pick the largest m.
                    if top_m >= len(ordered.relations):
                        log(f"  {explainer}: m={top_m} covers all {len(ordered.relations)} edges")
                        continue
                    explanation = ordered.relations[:top_m]
                    report = evaluate_explanation(
                        scorer.edge_index,
                        scorer.image_row,
                        column,
                        explanation,
                        candidate_scores(scorer, graph_scores[query], column),
                        graph_scores[query],
                        candidates.clip_scores[query],
                        alpha,
                        config,
                        seed,
                    )
                    if report is None:  # alpha = 1: the graph plays no part
                        continue
                    records.append(
                        {
                            "seed": seed,
                            "query_row": query,
                            "text_id": None
                            if query_text_ids is None
                            else int(query_text_ids[query]),
                            "image_id": image_id,
                            "clip_rank": column + 1,  # candidates are stored in CLIP's order
                            "alpha": alpha,
                            "explainer": explainer,
                            "top_m": top_m,
                            # Node rows of the split graph, so a reader can redo the removal.
                            "relations": [list(edge) for edge in explanation],
                            "concepts": [concept_name(source) for source, _ in explanation],
                            "explainer_values": ordered.values[:top_m],
                            "completeness_gap": ordered.completeness_gap,
                            **asdict(report),
                        }
                    )

    return {
        "settings": {
            "experiment": config.experiment,
            "ranking_experiment": ranking.experiment,
            "split": split,
            "seeds": config.seeds,
            "alpha_per_seed": alphas,
            "query_rows": query_rows.tolist(),
            "explained_image": "rank 1 under CLIP + Graph",
            "explainers": explainers,
            "top_m_options": top_m_options,
        },
        "summary": summarize(records, split, explainers, top_m_options, config.seeds),
        "records": records,
    }


def summarize(
    records: list[dict],
    split: str,
    explainers: list[str],
    top_m_options: list[int],
    seeds: list[int],
) -> dict:
    """Per (explainer, m): mean ± std over seeds, Wilcoxon vs random, and the val choice."""
    rows = []
    characterizations = {}
    for explainer in explainers:
        for top_m in top_m_options:
            group = [r for r in records if r["explainer"] == explainer and r["top_m"] == top_m]
            if not group:
                continue
            row = {"explainer": explainer, "top_m": top_m, "num_records": len(group)}
            for measure in SUMMARY_MEASURES:
                row[measure] = _mean_std_over_seeds(group, measure, seeds)
            row["all_in_reach_share"] = float(np.mean([r["all_in_reach"] for r in group]))
            row["p_fidelity_plus_vs_random"] = wilcoxon_greater(
                _per_query_mean(group, lambda r: r["fidelity_plus"] - r["random_plus_mean"])
            )
            row["p_rank_drop_vs_random"] = wilcoxon_greater(
                _per_query_mean(group, lambda r: r["rank_drop"] - r["random_rank_drop_mean"])
            )
            rows.append(row)
            characterizations[(explainer, top_m)] = [r["characterization"] for r in group]

    proposal = None
    if split == "val" and characterizations:
        explainer, top_m = choose_on_validation(characterizations, split)
        proposal = {"main_explainer": explainer, "top_m": top_m, "by": "mean characterization"}
    return {"rows": rows, "val_proposal": proposal}


def _mean_std_over_seeds(group: list[dict], measure: str, seeds: list[int]) -> dict | None:
    """Mean over queries per seed, then mean ± std (ddof=1) of those seed means."""
    seed_means = []
    for seed in seeds:
        values = [r[measure] for r in group if r["seed"] == seed and r[measure] is not None]
        if values:
            seed_means.append(float(np.mean(values)))
    if len(seed_means) < 2:
        return None
    mean, std = mean_and_std(seed_means)
    return {"mean": mean, "std": std}


def _per_query_mean(group: list[dict], difference) -> list[float]:
    """One value per query (averaged over seeds), so the test pairs independent queries."""
    by_query = {}
    for record in group:
        by_query.setdefault(record["query_row"], []).append(difference(record))
    return [float(np.mean(values)) for _, values in sorted(by_query.items())]


def make_explain_run_dir(
    output_dir: str | Path, experiment: str, split: str, now: datetime
) -> Path:
    """results/runs/<YYYYMMDD-HHMM>_<experiment>_<split>/; test only once (ADR-0009).

    One folder covers every seed, because the summary compares seeds. It holds no
    metrics.json, so scripts/summarize_runs.py does not count it as a model run.
    """
    output_dir = Path(output_dir)
    if split == "test" and any(output_dir.glob(f"*_{experiment}_test/{EVALUATION_JSON}")):
        raise ValueError(f"{experiment} was already evaluated on test; test runs once")
    run_dir = output_dir / f"{now:%Y%m%d-%H%M}_{experiment}_{split}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_evaluation(run_dir: Path, evaluation: dict, config: ExplainConfig) -> None:
    """Write evaluation.json, summary.md, config.yaml and env.json."""
    with open(run_dir / EVALUATION_JSON, "w", encoding="utf-8") as file:
        json.dump(evaluation, file, indent=2)
    with open(run_dir / SUMMARY_MD, "w", encoding="utf-8") as file:
        file.write(format_summary(evaluation))
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(asdict(config), file, sort_keys=False)
    with open(run_dir / "env.json", "w", encoding="utf-8") as file:
        json.dump(collect_env(library_versions()), file, indent=2)


def format_summary(evaluation: dict) -> str:
    """A Markdown table of the summary rows (mean ± std over seeds)."""
    settings = evaluation["settings"]
    lines = [
        f"# Explanation evaluation: {settings['ranking_experiment']}, {settings['split']}",
        "",
        f"Seeds {settings['seeds']}, {len(settings['query_rows'])} queries, "
        f"explained image: {settings['explained_image']}. Mean ± std over seeds.",
        "",
        "| explainer | m | Fid+ | random Fid+ | rank drop | random rank drop | Fid- "
        "| characterization | robust Fid+ | sparsity | p (Fid+ > random) | p (rank > random) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in evaluation["summary"]["rows"]:
        cells = [row["explainer"], str(row["top_m"])]
        for measure in (
            "fidelity_plus",
            "random_plus_mean",
            "rank_drop",
            "random_rank_drop_mean",
            "fidelity_minus",
            "characterization",
            "robust_plus",
            "sparsity",
        ):
            cells.append(_format_mean_std(row[measure]))
        cells.append(f"{row['p_fidelity_plus_vs_random']:.3g}")
        cells.append(f"{row['p_rank_drop_vs_random']:.3g}")
        lines.append("| " + " | ".join(cells) + " |")
    proposal = evaluation["summary"]["val_proposal"]
    if proposal is not None:
        lines += [
            "",
            f"Val proposal (to write into the explain config before test): main_explainer "
            f"{proposal['main_explainer']}, top_m {proposal['top_m']}.",
        ]
    return "\n".join(lines) + "\n"


def _format_mean_std(value: dict | None) -> str:
    if value is None:
        return "–"
    return f"{value['mean']:.4f} ± {value['std']:.4f}"
