"""CLIP + PCST subgraph reranker: choose on val, test once (ADR-0051, stage 1).

Two separate steps, so test can never feed back into a choice:
    select_on_val   grid over prize_k, edge_cost, alpha on VAL, save the choice
    evaluate_test   read the saved choice, score TEST once, choose nothing

There is no training: the score is the overlap between an image's concepts and the
PCST subgraph of the query (mmxai.models.subgraph_scorer).
"""

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from mmxai.eval.graph_rerank import (
    Inputs,
    library_versions,
    load_inputs,
    split_candidates,
)
from mmxai.eval.metrics import compute_metrics
from mmxai.graph.contains import split_subgraph
from mmxai.graph.pcst import QuerySubgraph, no_tree_subgraph, query_subgraph, undirected_pairs
from mmxai.graph.shuffle import shuffle_concept_pairs
from mmxai.models.subgraph_scorer import image_concept_sets, overlap_scores
from mmxai.rerank.fusion import (
    Candidates,
    choose_alpha,
    fuse_scores,
    rerank_ranks,
    sweep_alpha,
)
from mmxai.utils.runs import (
    collect_env,
    file_sha256,
    git_state,
    make_run_dir,
    test_run_exists,
    write_run_files,
)
from mmxai.utils.yaml_config import load_yaml_dataclass

SELECTION_FILE = "selection.json"
SUBGRAPH_MODES = ("pcst", "top_k_only")
LOG_EVERY = 1000


@dataclass(frozen=True)
class PcstRerankConfig:
    """Settings of one G-Retriever stage-1 experiment (configs/clip_pcst_vg*.yaml).

    graph_rerank.load_inputs reads dataset, features_dir, graph_dir, split_file and
    concept_edges_dir; split_candidates reads top_k and same_text_threshold. Nothing else.
    """

    experiment: str
    dataset: str
    split_file: str
    features_dir: str
    graph_dir: str
    concept_edges_dir: str
    subgraph_mode: str  # "pcst" (main) or "top_k_only" (ablation 1)
    shuffle_concept_edges: bool  # ablation 2; the shuffle seed is the run seed
    top_k: int
    recall_ks: list[int]
    same_text_threshold: float
    prize_k_grid: list[int]
    edge_cost_grid: list[float]
    alpha_grid: list[float]
    selection_metric: str
    selection_dir: str  # where selection.json is written (artifacts/, gitignored)
    output_dir: str


def load_pcst_rerank_config(path: str | Path) -> PcstRerankConfig:
    """Load the YAML config; reject an unknown subgraph mode or a metric we do not report."""
    config = load_yaml_dataclass(path, PcstRerankConfig)
    if config.subgraph_mode not in SUBGRAPH_MODES:
        raise ValueError(f"subgraph_mode must be one of {SUBGRAPH_MODES}")
    if config.selection_metric != f"MRR@{config.top_k}":
        raise ValueError(f"selection_metric must be MRR@{config.top_k}")
    return config


def concept_pairs(inputs: Inputs, config: PcstRerankConfig, seed: int) -> np.ndarray:
    """Undirected concept pairs for PCST, shuffled (degree-preserving) in ablation 2."""
    edges = inputs.concept_edges
    if edges is None:
        raise ValueError("the PCST reranker needs concept_edges_dir")
    pairs = undirected_pairs(edges.edge_source, edges.edge_target)
    if config.shuffle_concept_edges:
        pairs = shuffle_concept_pairs(pairs, seed)
    return pairs


def split_subgraphs(
    queries: np.ndarray,
    concept_vectors: np.ndarray,
    pairs: np.ndarray,
    mode: str,
    prize_k: int,
    edge_cost: float,
    log=print,
) -> list[QuerySubgraph]:
    """One S_q per query; slow on ~25,000 queries, so report progress."""
    subgraphs = []
    started = time.time()
    for number, query in enumerate(queries, start=1):
        if mode == "pcst":
            subgraphs.append(query_subgraph(query, concept_vectors, pairs, prize_k, edge_cost))
        else:
            subgraphs.append(no_tree_subgraph(query, concept_vectors, prize_k))
        if number % LOG_EVERY == 0:
            log(f"  subgraphs {number}/{len(queries)} after {time.time() - started:.0f}s")
    return subgraphs


def grid_settings(config: PcstRerankConfig) -> list[tuple[int, float]]:
    """(prize_k, edge_cost) pairs to try. Without a tree the cost is unused, so try one."""
    costs = config.edge_cost_grid
    if config.subgraph_mode == "top_k_only":
        costs = costs[:1]
    return [(k, cost) for k in config.prize_k_grid for cost in costs]


def grid_search(
    config: PcstRerankConfig,
    candidates: Candidates,
    sets: dict[int, set[int]],
    concept_vectors: np.ndarray,
    pairs: np.ndarray,
    log=print,
) -> list[dict]:
    """For each (prize_k, edge_cost): its best alpha and metrics on the given candidates."""
    results = []
    for prize_k, edge_cost in grid_settings(config):
        log(f"prize_k={prize_k} edge_cost={edge_cost}")
        subgraphs = split_subgraphs(
            candidates.query_embeddings,
            concept_vectors,
            pairs,
            config.subgraph_mode,
            prize_k,
            edge_cost,
            log,
        )
        scores = overlap_scores(candidates.candidate_ids, subgraphs, sets)
        sweep = sweep_alpha(candidates, scores, config.alpha_grid, config.recall_ks)
        alpha = choose_alpha(sweep, config.selection_metric)
        metrics = sweep[alpha]
        results.append(
            {
                "prize_k": prize_k,
                "edge_cost": edge_cost,
                "alpha": alpha,
                "metric": metrics[config.selection_metric],
                "metrics": metrics,
            }
        )
    return results


def choose_setting(results: list[dict]) -> dict:
    """Best metric; ties prefer larger alpha, then smaller prize_k, then larger edge_cost.

    Decision (ADR-0051): on a tie, prefer the setting closest to plain CLIP.
      Why: a larger alpha, a smaller prize_k and a larger edge_cost all give a smaller
        subgraph that moves the ranking less; without evidence the graph helps, stay near CLIP.
      Decided by: Claude Code proposal, 2026-11-15; not yet approved by the team.
    """
    if not results:
        raise ValueError("no setting to choose from")
    return max(
        results,
        key=lambda r: (r["metric"], r["alpha"], -r["prize_k"], r["edge_cost"]),
    )


def run_facts(config: PcstRerankConfig, inputs: Inputs, split_name: str, seed: int) -> dict:
    """What every metrics.json must say about where its number came from (ADR-0009, ADR-0010)."""
    return {
        "split": split_name,
        "split_file": config.split_file,
        "split_sha256": file_sha256(config.split_file),
        "seed": seed,
        "subgraph_mode": config.subgraph_mode,
        "shuffle_concept_edges": config.shuffle_concept_edges,
        "graph_dir": config.graph_dir,
        "concept_edges_dir": config.concept_edges_dir,
        "graph_git_commit": inputs.graph_meta["git"]["git_commit"],
        "features_code_hash": inputs.feature_meta["code_hash"],
    }


def write_run(config: PcstRerankConfig, metrics: dict, seed: int) -> Path:
    run_dir = make_run_dir(config.output_dir, config.experiment, seed, datetime.now())
    write_run_files(run_dir, asdict(config), metrics, collect_env(library_versions()))
    return run_dir


def selection_path(config: PcstRerankConfig, seed: int) -> Path:
    return Path(config.selection_dir) / f"seed{seed}" / SELECTION_FILE


def select_on_val(config: PcstRerankConfig, seed: int, log=print) -> Path:
    """Choose prize_k, edge_cost and alpha on VAL only. Never reads test."""
    path = selection_path(config, seed)
    if path.exists():
        raise FileExistsError(f"{path} exists; remove it on purpose to choose again")
    inputs = load_inputs(config)
    candidates = split_candidates(config, inputs, "val")
    sets = image_concept_sets(split_subgraph(inputs.graph, "val"))
    pairs = concept_pairs(inputs, config, seed)

    results = grid_search(config, candidates, sets, inputs.concept_features, pairs, log)
    chosen = choose_setting(results)
    log(f"seed {seed} chosen on val: {chosen}")

    selection = {
        "prize_k": chosen["prize_k"],
        "edge_cost": chosen["edge_cost"],
        "alpha": chosen["alpha"],
        "val_metric": chosen["metric"],
        "split_sha256": file_sha256(config.split_file),
        "git": git_state(),
        "grid": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(selection, file, indent=2)

    metrics = dict(chosen["metrics"])
    metrics.update(run_facts(config, inputs, "val", seed))
    metrics.update(prize_k=chosen["prize_k"], edge_cost=chosen["edge_cost"], alpha=chosen["alpha"])
    metrics["grid"] = results
    return write_run(config, metrics, seed)


def has_test_run(config: PcstRerankConfig, seed: int) -> bool:
    """True if results/runs already holds a test run of this experiment and seed."""
    return test_run_exists(config.output_dir, config.experiment, seed)


def evaluate_test(config: PcstRerankConfig, seed: int, log=print) -> Path:
    """Score TEST once with the setting chosen on val. Chooses nothing.

    Decision (ADR-0009, ADR-0047): refuse a second test run of the same experiment and seed.
      Why: test runs once with the locked config; a repeat means test could steer choices.
      Decided by: FluteInTheRain, ADR-0009; same check as graph_rerank.evaluate_test.
    """
    if has_test_run(config, seed):
        raise FileExistsError(f"test of {config.experiment} seed {seed} was already run")
    with open(selection_path(config, seed), encoding="utf-8") as file:
        selection = json.load(file)

    inputs = load_inputs(config)
    candidates = split_candidates(config, inputs, "test")
    sets = image_concept_sets(split_subgraph(inputs.graph, "test"))
    pairs = concept_pairs(inputs, config, seed)

    subgraphs = split_subgraphs(
        candidates.query_embeddings,
        inputs.concept_features,
        pairs,
        config.subgraph_mode,
        selection["prize_k"],
        selection["edge_cost"],
        log,
    )
    scores = overlap_scores(candidates.candidate_ids, subgraphs, sets)
    final = fuse_scores(candidates.clip_scores, scores, selection["alpha"])
    metrics = compute_metrics(rerank_ranks(candidates, final), config.recall_ks, config.top_k)
    metrics.update(run_facts(config, inputs, "test", seed))
    metrics.update(
        prize_k=selection["prize_k"], edge_cost=selection["edge_cost"], alpha=selection["alpha"]
    )
    metrics["selection_git_commit"] = selection["git"]
    log(f"seed {seed} test: {metrics}")
    return write_run(config, metrics, seed)
