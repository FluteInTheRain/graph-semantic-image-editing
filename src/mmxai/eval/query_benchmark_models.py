"""Rank the verified query benchmark with trained re-rankers (ADR-0057).

Evaluation only (PDF section 6, ADR-0009): nothing here trains or tunes. Each model
uses the checkpoint and alpha its own selection.json holds, which were chosen on val.

Every ranker works like the CLIP + Graph protocol of ADR-0043:
    1. exact cosine search of the 5000 test images -> CLIP's top_k candidates
    2. the model scores those candidates, final = alpha * clip + (1 - alpha) * graph
    3. the candidates are re-ordered by the final score

A ranker only receives the query TEXTS and their CLIP embeddings (QueryRanker protocol),
never hidden_concept, kg_evidence, path_hint or relevant.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from mmxai.data.splits import load_split
from mmxai.eval.graph_rerank import (
    MODEL_FILE,
    SELECTION_FILE,
    Inputs,
    SplitGraph,
    build_split_graph,
    image_vectors,
    load_graph_rerank_config,
    load_inputs,
    make_model,
    seed_model_dir,
)
from mmxai.eval.pcst_rerank import concept_pairs, load_checked_inputs
from mmxai.eval.pcst_sage_rerank import (
    PcstSageConfig,
    compute_subgraph_rows,
    graph_scores,
    load_pcst_sage_config,
    split_scoring,
)
from mmxai.eval.query_benchmark import (
    encode_query_texts,
    evaluate_benchmark,
    load_verified_benchmark,
)
from mmxai.index.vector_store import select_split_images
from mmxai.models.subgraph_sage import SubgraphSage
from mmxai.rerank.fusion import Candidates, candidate_graph_scores, fuse_scores
from mmxai.utils.runs import (
    collect_env,
    file_sha256,
    make_run_dir,
    test_run_exists,
    write_run_files,
)
from mmxai.utils.yaml_config import load_yaml_dataclass

MODEL_KINDS = ("graph_rerank", "pcst_sage")


def clip_candidates(
    test_embeddings: np.ndarray,
    test_image_ids: np.ndarray,
    query_embeddings: np.ndarray,
    top_k: int,
) -> Candidates:
    """CLIP's exact top_k test images for every query, best first.

    Embeddings are L2-normalised, so the dot product is the cosine. `relevant` stays
    empty: the metrics are computed later from the benchmark's own answers.
    """
    if top_k > len(test_image_ids):
        raise ValueError(f"top_k={top_k} is larger than the {len(test_image_ids)} test images")
    scores = query_embeddings @ test_embeddings.T  # (num_queries, num_images)
    # Stable sort: equal scores keep the gallery order, so the result is deterministic.
    order = np.argsort(-scores, axis=1, kind="stable")[:, :top_k]
    clip_scores = np.take_along_axis(scores, order, axis=1)
    relevant = [set() for _ in range(len(query_embeddings))]
    return Candidates(query_embeddings, test_image_ids[order], clip_scores, relevant)


def reranked_ids(candidates: Candidates, final_scores: np.ndarray) -> np.ndarray:
    """Re-order each query's candidate ids by final score, highest first.

    The sort is stable, so equal scores keep CLIP's order (alpha = 1 gives CLIP exactly).
    """
    order = np.argsort(-final_scores, axis=1, kind="stable")
    return np.take_along_axis(candidates.candidate_ids, order, axis=1)


class GraphRerankRanker:
    """CLIP + Graph (graph_rerank models): score = cos(q, h_i) with h computed once."""

    def __init__(
        self,
        image_h: np.ndarray,
        h_image_ids: np.ndarray,
        test_embeddings: np.ndarray,
        test_image_ids: np.ndarray,
        alpha: float,
        top_k: int,
    ) -> None:
        self.image_h = image_h
        self.h_image_ids = h_image_ids
        self.test_embeddings = test_embeddings
        self.test_image_ids = test_image_ids
        self.alpha = alpha
        self.top_k = top_k

    def rank(self, texts: list[str], query_embeddings: np.ndarray, top_k: int) -> np.ndarray:
        candidates = clip_candidates(
            self.test_embeddings, self.test_image_ids, query_embeddings, top_k
        )
        graph = candidate_graph_scores(candidates, self.image_h, self.h_image_ids)
        final = fuse_scores(candidates.clip_scores, graph, self.alpha)
        return reranked_ids(candidates, final)


class PcstSageRanker:
    """G-Retriever stage 2: score each candidate on the query's own PCST subgraph S_q."""

    def __init__(
        self,
        config: PcstSageConfig,
        inputs: Inputs,
        pairs: np.ndarray,
        test_graph: SplitGraph,
        model: SubgraphSage,
        alpha: float,
        test_embeddings: np.ndarray,
        test_image_ids: np.ndarray,
    ) -> None:
        self.config = config
        self.inputs = inputs
        self.pairs = pairs
        self.test_graph = test_graph
        self.model = model
        self.alpha = alpha
        self.test_embeddings = test_embeddings
        self.test_image_ids = test_image_ids

    def rank(self, texts: list[str], query_embeddings: np.ndarray, top_k: int) -> np.ndarray:
        candidates = clip_candidates(
            self.test_embeddings, self.test_image_ids, query_embeddings, top_k
        )
        # No cache: the benchmark has a few dozen queries, so PCST takes seconds.
        offsets, values = compute_subgraph_rows(
            self.config, self.inputs, self.pairs, candidates.query_embeddings
        )
        scoring = split_scoring(self.inputs, "test", self.test_graph, candidates, offsets, values)
        graph = graph_scores(self.config, self.model, scoring)
        final = fuse_scores(candidates.clip_scores, graph, self.alpha)
        return reranked_ids(candidates, final)


@dataclass(frozen=True)
class ModelBenchmarkEvalConfig:
    """Settings of one trained-model evaluation; dataset, split and top_k come from model_config."""

    experiment: str
    model_kind: str  # "graph_rerank" or "pcst_sage"
    model_config: str  # the YAML the model was trained with
    benchmark_path: str
    model: str
    pretrained: str
    device: str
    text_batch_size: int
    recall_ks: list[int]
    output_dir: str


def load_model_benchmark_eval_config(path: str | Path) -> ModelBenchmarkEvalConfig:
    config = load_yaml_dataclass(path, ModelBenchmarkEvalConfig)
    if config.model_kind not in MODEL_KINDS:
        raise ValueError(f"model_kind must be one of {MODEL_KINDS}, got {config.model_kind!r}")
    return config


def has_benchmark_run(config: ModelBenchmarkEvalConfig, seed: int) -> bool:
    """True if output_dir already holds a run of this experiment and seed."""
    return test_run_exists(config.output_dir, config.experiment, seed)


def load_model_config(config: ModelBenchmarkEvalConfig):
    """The config the model was trained with (GraphRerankConfig or PcstSageConfig)."""
    if config.model_kind == "graph_rerank":
        return load_graph_rerank_config(config.model_config)
    return load_pcst_sage_config(config.model_config)


def check_selection_split(selection: dict, split_file: str) -> None:
    """Refuse a selection.json that was made against another split file."""
    # graph_rerank selections do not record split_sha256 (pcst_sage ones do), so for those
    # the check is skipped here; load_inputs still refuses a graph built on another split.
    if "split_sha256" not in selection:
        return
    if selection["split_sha256"] != file_sha256(split_file):
        raise ValueError("selection.json was made with a different split file than the config")


def check_feature_model(config: ModelBenchmarkEvalConfig, feature_meta: dict) -> None:
    """Query and image vectors must come from the same CLIP, or cosine means nothing."""
    if feature_meta["model"] != config.model or feature_meta["pretrained"] != config.pretrained:
        raise ValueError(
            f"features use {feature_meta['model']}/{feature_meta['pretrained']}, "
            f"config asks {config.model}/{config.pretrained}"
        )


def run_model_benchmark_evaluation(config: ModelBenchmarkEvalConfig, seed: int, log=print) -> Path:
    """Score the verified benchmark with a saved model and write a run folder (ADR-0010).

    Decision (ADR-0009, ADR-0057): refuse a second run of the same experiment and seed.
      Why: the benchmark is read once with the locked model; a repeat could steer choices.
      Decided by: same rule as graph_rerank.evaluate_test; applied here by Claude Code.
    """
    import torch

    if has_benchmark_run(config, seed):
        raise FileExistsError(f"{config.experiment} seed {seed} was already run")
    model_config = load_model_config(config)

    split = load_split(model_config.split_file)
    test_ids = split["test"]
    test_id_strings = {str(image_id) for image_id in test_ids.tolist()}
    queries = load_verified_benchmark(config.benchmark_path, test_id_strings)

    # Both model kinds keep their files in <model_dir>/seed<k>/.
    model_dir = seed_model_dir(model_config, seed)
    with open(model_dir / SELECTION_FILE, encoding="utf-8") as file:
        selection = json.load(file)
    check_selection_split(selection, model_config.split_file)

    if config.model_kind == "graph_rerank":
        inputs = load_inputs(model_config)
    else:
        inputs = load_checked_inputs(model_config)
    check_feature_model(config, inputs.feature_meta)

    test_graph = build_split_graph(inputs, "test")
    test_embeddings, test_image_ids = select_split_images(
        inputs.image_embeddings, inputs.image_ids, test_ids
    )
    state_dict = torch.load(model_dir / MODEL_FILE, weights_only=True)
    # encode_query_texts only reads model, pretrained, device and text_batch_size.
    query_embeddings = encode_query_texts(config, [query["text"] for query in queries])

    if config.model_kind == "graph_rerank":
        model = make_model(model_config, test_graph.x.shape[1])
        model.load_state_dict(state_dict)
        image_h = image_vectors(model, test_graph)
        ranker = GraphRerankRanker(
            image_h,
            test_graph.image_ids,
            test_embeddings,
            test_image_ids,
            selection["alpha"],
            model_config.top_k,
        )
    else:
        pairs = concept_pairs(inputs, model_config, seed)
        model = SubgraphSage(test_graph.x.shape[1])
        model.load_state_dict(state_dict)
        ranker = PcstSageRanker(
            model_config,
            inputs,
            pairs,
            test_graph,
            model,
            selection["alpha"],
            test_embeddings,
            test_image_ids,
        )

    metrics = evaluate_benchmark(
        queries, query_embeddings, ranker, config.recall_ks, model_config.top_k
    )
    metrics["split"] = "test"
    metrics["num_images"] = len(test_image_ids)
    metrics["split_file"] = model_config.split_file
    metrics["split_sha256"] = file_sha256(model_config.split_file)
    metrics["benchmark_sha256"] = file_sha256(config.benchmark_path)
    metrics["seed"] = seed
    metrics["alpha"] = selection["alpha"]
    metrics["best_epoch"] = selection["best_epoch"]
    metrics["model_sha256"] = file_sha256(model_dir / MODEL_FILE)
    metrics["model_git_commit"] = selection["git"]
    metrics["model_experiment"] = model_config.experiment
    log(f"seed {seed}: {config.experiment} alpha {selection['alpha']}")

    run_dir = make_run_dir(config.output_dir, config.experiment, seed, datetime.now())
    env = collect_env({"numpy": np.__version__, "torch": torch.__version__})
    write_run_files(run_dir, asdict(config), metrics, env)
    return run_dir


SUMMARY_METRICS = ("R@1", "R@5", "R@10", "MRR@100")


def load_run(run_dir: Path) -> tuple[str, dict]:
    """Return (experiment, metrics) of one run folder; the name is in config.yaml."""
    with open(run_dir / "config.yaml", encoding="utf-8") as file:
        experiment = yaml.safe_load(file)["experiment"]
    with open(run_dir / "metrics.json", encoding="utf-8") as file:
        metrics = json.load(file)
    return experiment, metrics


def scope_metrics(metrics: dict) -> dict[str, dict | None]:
    """Flatten a metrics.json into {scope name: metric dict}, e.g. "activity_direct"."""
    scopes = {name: metrics.get(name) for name in ("all", "direct", "indirect")}
    for group, by_type in metrics.get("by_query_group", {}).items():
        for query_type, group_metrics in by_type.items():
            scopes[f"{group}_{query_type}"] = group_metrics
    return scopes


def mean_and_std(values: list[float]) -> tuple[float, float | None]:
    """Mean and sample std (ddof=1, ADR-0011); the std of a single seed is None."""
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, None
    return mean, float(np.std(values, ddof=1))


def summarize_benchmark_runs(runs_dir: str | Path) -> list[dict]:
    """One row per experiment: mean and std over seeds of every scope's metrics.

    Refuses a repeated seed and runs made on different benchmark files, because every
    model must be judged on the same queries (ADR-0057).
    """
    runs: dict[str, list[dict]] = {}
    for run_dir in sorted(Path(runs_dir).glob("*/metrics.json")):
        experiment, metrics = load_run(run_dir.parent)
        runs.setdefault(experiment, []).append(metrics)
    if not runs:
        raise ValueError(f"no runs found in {runs_dir}")

    benchmark_hashes = {m["benchmark_sha256"] for group in runs.values() for m in group}
    if len(benchmark_hashes) > 1:
        raise ValueError("runs were made on different benchmark files (benchmark_sha256)")

    rows = []
    for experiment in sorted(runs):
        group = runs[experiment]
        seeds = sorted(m["seed"] for m in group)
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"{experiment} has a duplicated seed: {seeds}")
        row = {
            "experiment": experiment,
            "num_seeds": len(seeds),
            "seeds": " ".join(str(seed) for seed in seeds),
            "num_queries": group[0]["all"]["num_queries"],
        }
        # Scope names come from the first run; all runs share one benchmark file.
        for scope in scope_metrics(group[0]):
            for metric in SUMMARY_METRICS:
                values = []
                for metrics in group:
                    scope_values = scope_metrics(metrics).get(scope)
                    if scope_values is not None:
                        values.append(scope_values[metric])
                # A scope that is None in the runs stays empty.
                if len(values) == len(group):
                    mean, std = mean_and_std(values)
                else:
                    mean, std = None, None
                row[f"{scope}_{metric}_mean"] = mean
                row[f"{scope}_{metric}_std"] = std
        rows.append(row)
    return rows
