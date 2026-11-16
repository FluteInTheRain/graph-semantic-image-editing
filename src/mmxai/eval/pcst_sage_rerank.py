"""G-Retriever stage 2: GraphSAGE on the query's PCST subgraph (ADR-0051).

Three separate steps, so test can never feed back into a choice:
    prepare_caches    compute S_q of every train and val text once (never test)
    train_and_select  train on train, early-stop on VAL, choose alpha on VAL, save the model
    evaluate_test     load the saved model and alpha, score TEST once, choose nothing

prize_k and edge_cost are fixed in the config (no grid): one PCST pass over the train
texts takes about an hour, so a grid would be too slow.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from mmxai.eval.graph_rerank import (
    MODEL_FILE,
    SELECTION_FILE,
    Inputs,
    SplitGraph,
    build_split_graph,
    build_training_data,
    split_candidates,
)
from mmxai.eval.metrics import compute_metrics
from mmxai.eval.pcst_rerank import (
    SUBGRAPH_MODES,
    concept_pairs,
    has_test_run,
    load_checked_inputs,
    run_facts,
    split_subgraphs,
    top_concepts_per_query,
    write_run,
)
from mmxai.graph.contains import split_subgraph
from mmxai.graph.pcst import edge_costs
from mmxai.graph.pcst_cache import load_cache, pack_subgraphs, padded_rows, save_cache
from mmxai.models.subgraph_sage import (
    SubgraphSage,
    SubgraphTrainingData,
    candidate_subgraph_scores,
    image_concept_matrix,
    subgraph_batch_loss,
)
from mmxai.models.train import TrainingData, fit
from mmxai.rerank.fusion import (
    Candidates,
    choose_alpha,
    fuse_scores,
    rerank_ranks,
    sweep_alpha,
)
from mmxai.utils.runs import collect_env, file_sha256
from mmxai.utils.seed import set_seed
from mmxai.utils.yaml_config import load_yaml_dataclass

SHARED_CACHE = "shared"


@dataclass(frozen=True)
class PcstSageConfig:
    """Settings of one G-Retriever stage-2 experiment (configs/clip_pcst_sage_vg*.yaml).

    The reused graph_rerank functions read these fields only (duck typing):
      load_inputs          dataset, features_dir, graph_dir, split_file, concept_edges_dir
      build_training_data  top_k, same_text_threshold
      split_candidates     top_k, same_text_threshold
    The reused pcst_rerank functions read split_file, subgraph_mode, shuffle_concept_edges,
    graph_dir, concept_edges_dir, experiment and output_dir.
    """

    experiment: str
    dataset: str
    split_file: str
    features_dir: str
    graph_dir: str
    concept_edges_dir: str
    subgraph_mode: str  # "pcst" (main) or "top_k_only" (ablation 1)
    shuffle_concept_edges: bool  # ablation 2; the shuffle seed is the run seed
    prize_k: int
    edge_cost: float
    cache_dir: str  # packed subgraphs S_q (artifacts/, gitignored)
    model_dir: str  # model.pt and selection.json, one folder per seed
    top_k: int
    recall_ks: list[int]
    same_text_threshold: float
    num_hard_negatives: int
    temperature: float
    learning_rate: float
    batch_size: int
    max_epochs: int
    patience: int
    score_chunk: int  # memory setting only: it never changes a score
    alpha_grid: list[float]
    selection_metric: str
    output_dir: str


def load_pcst_sage_config(path: str | Path) -> PcstSageConfig:
    """Load the YAML config; reject an unknown subgraph mode or a metric we do not report."""
    config = load_yaml_dataclass(path, PcstSageConfig)
    if config.subgraph_mode not in SUBGRAPH_MODES:
        raise ValueError(f"subgraph_mode must be one of {SUBGRAPH_MODES}")
    if config.selection_metric != f"MRR@{config.top_k}":
        raise ValueError(f"selection_metric must be MRR@{config.top_k}")
    return config


def cache_path(config: PcstSageConfig, seed: int) -> Path:
    """Folder of the cached subgraphs.

    Without shuffling, S_q does not depend on the seed, so all seeds share one cache and
    the hour of PCST is paid once. The shuffle seed changes the PMI pairs, so the shuffled
    ablation needs one cache per seed.
    """
    if config.shuffle_concept_edges:
        return Path(config.cache_dir) / f"seed{seed}"
    return Path(config.cache_dir) / SHARED_CACHE


def cache_meta(
    config: PcstSageConfig, inputs: Inputs, seed: int, split: str, num_texts: int
) -> dict:
    """What a cache must say about how it was made; load_cache refuses a mismatch."""
    shuffle_seed = None
    if config.shuffle_concept_edges:
        shuffle_seed = seed
    return {
        "prize_k": config.prize_k,
        "edge_cost": config.edge_cost,
        "subgraph_mode": config.subgraph_mode,
        "shuffle_concept_edges": config.shuffle_concept_edges,
        "shuffle_seed": shuffle_seed,
        "split": split,
        "num_texts": num_texts,
        "split_sha256": file_sha256(config.split_file),
        "graph_git_commit": inputs.graph_meta["git"]["git_commit"],
        "concept_edges_dir": config.concept_edges_dir,
    }


def compute_subgraph_rows(
    config: PcstSageConfig, inputs: Inputs, pairs: np.ndarray, queries: np.ndarray, log=print
) -> tuple[np.ndarray, np.ndarray]:
    """S_q of every query, packed as (offsets, values) for pcst_cache."""
    top = top_concepts_per_query(queries, inputs.concept_features, config.prize_k)
    subgraphs = split_subgraphs(
        top,
        len(inputs.concept_features),
        pairs,
        config.subgraph_mode,
        config.prize_k,
        edge_costs(pairs, config.edge_cost),
        log,
    )
    return pack_subgraphs([subgraph.concepts for subgraph in subgraphs])


def prepare_caches(config: PcstSageConfig, seed: int, log=print) -> None:
    """Cache S_q for the train texts stage 2 trains on and for all val texts. Never test.

    Only the train texts that build_training_data keeps are cached (ADR-0047 filter):
    the others are never used by the loss, so PCST on them would be wasted time.
    """
    folder = cache_path(config, seed)
    missing = [split for split in ("train", "val") if not (folder / f"{split}.npz").exists()]
    if not missing:
        log(f"caches in {folder} already exist; nothing to do")
        return
    inputs = load_checked_inputs(config)
    pairs = concept_pairs(inputs, config, seed)

    if "train" in missing:
        train_graph = build_split_graph(inputs, "train")
        training = build_training_data(config, inputs, train_graph, log)
        queries = training.text_x.numpy()
        offsets, values = compute_subgraph_rows(config, inputs, pairs, queries, log)
        meta = cache_meta(config, inputs, seed, "train", len(queries))
        save_cache(folder, "train", offsets, values, meta)
        log(f"saved train cache: {len(queries)} texts")
    else:
        log("train cache already exists; skipping")

    if "val" in missing:
        candidates = split_candidates(config, inputs, "val")
        queries = candidates.query_embeddings
        offsets, values = compute_subgraph_rows(config, inputs, pairs, queries, log)
        meta = cache_meta(config, inputs, seed, "val", len(queries))
        save_cache(folder, "val", offsets, values, meta)
        log(f"saved val cache: {len(queries)} texts")
    else:
        log("val cache already exists; skipping")


def estimate_pcst_time(config: PcstSageConfig, seed: int, num_texts: int, log=print) -> float:
    """Time PCST on the first num_texts train texts; write nothing. Returns seconds per text.

    It uses all train texts, not only the ones build_training_data keeps, so it avoids the
    slow hard-negative search; the kept texts are most of them, so the projection is close.
    """
    inputs = load_checked_inputs(config)
    pairs = concept_pairs(inputs, config, seed)
    train_graph = build_split_graph(inputs, "train")
    is_train_text = np.isin(inputs.texts["image_ids"], train_graph.image_ids)
    all_queries = inputs.texts["embeddings"][is_train_text]
    queries = all_queries[:num_texts]

    started = time.time()
    compute_subgraph_rows(config, inputs, pairs, queries, log)
    seconds = time.time() - started
    per_text = seconds / max(len(queries), 1)
    projected = per_text * len(all_queries)
    log(
        f"{len(queries)} texts in {seconds:.1f}s = {per_text:.4f}s/text; "
        f"projected for {len(all_queries)} train texts: {projected / 60:.0f} min"
    )
    return per_text


def build_subgraph_training_data(
    training: TrainingData,
    inputs: Inputs,
    train_graph: SplitGraph,
    text_subgraphs: np.ndarray,
) -> SubgraphTrainingData:
    """Stage-1 training data plus each image's concept rows and each text's subgraph."""
    contains = split_subgraph(inputs.graph, "train")
    # image_concepts rows must line up with image_x rows, or h would mix up images.
    if not np.array_equal(contains.image_ids, train_graph.image_ids):
        raise ValueError("train graph and contains graph list the images in a different order")
    num_images = len(train_graph.image_ids)
    return SubgraphTrainingData(
        image_x=training.image_x,
        image_concepts=torch.from_numpy(image_concept_matrix(contains)),
        concept_x=train_graph.x[num_images:],
        text_x=training.text_x,
        text_image_row=training.text_image_row,
        candidate_rows=training.candidate_rows,
        is_negative=training.is_negative,
        text_subgraphs=torch.from_numpy(text_subgraphs),
    )


@dataclass(frozen=True)
class SplitScoring:
    """Everything candidate_subgraph_scores needs for the candidates of one split."""

    query_x: torch.Tensor
    query_subgraphs: torch.Tensor
    candidate_rows: torch.Tensor
    image_x: torch.Tensor
    image_concepts: torch.Tensor
    concept_x: torch.Tensor


def split_scoring(
    inputs: Inputs,
    split_name: str,
    split_graph: SplitGraph,
    candidates: Candidates,
    offsets: np.ndarray,
    values: np.ndarray,
) -> SplitScoring:
    """Tensors to score one split's candidates; candidate ids become rows of the split."""
    contains = split_subgraph(inputs.graph, split_name)
    if not np.array_equal(contains.image_ids, split_graph.image_ids):
        raise ValueError(f"{split_name} graphs list the images in a different order")
    # Graph image ids are sorted, so searchsorted gives each candidate's row.
    rows = np.searchsorted(split_graph.image_ids, candidates.candidate_ids)
    if not np.array_equal(split_graph.image_ids[rows], candidates.candidate_ids):
        raise ValueError(f"a {split_name} candidate is not an image of the split")
    num_images = len(split_graph.image_ids)
    return SplitScoring(
        query_x=torch.from_numpy(candidates.query_embeddings),
        query_subgraphs=torch.from_numpy(padded_rows(offsets, values)),
        candidate_rows=torch.from_numpy(rows),
        image_x=split_graph.x[:num_images],
        image_concepts=torch.from_numpy(image_concept_matrix(contains)),
        concept_x=split_graph.x[num_images:],
    )


def graph_scores(config: PcstSageConfig, model: SubgraphSage, scoring: SplitScoring) -> np.ndarray:
    """(num_queries, top_k) graph score of every candidate."""
    return candidate_subgraph_scores(
        model,
        scoring.query_x,
        scoring.query_subgraphs,
        scoring.candidate_rows,
        scoring.image_x,
        scoring.image_concepts,
        scoring.concept_x,
        config.score_chunk,
    )


def seed_model_dir(config: PcstSageConfig, seed: int) -> Path:
    return Path(config.model_dir) / f"seed{seed}"


def train_and_select(config: PcstSageConfig, seed: int, log=print) -> Path:
    """Train one seed, early-stop and choose alpha on val, save the model, write a val run."""
    model_dir = seed_model_dir(config, seed)
    if model_dir.exists():
        raise FileExistsError(f"{model_dir} exists; delete it to train this seed again")
    set_seed(seed)

    inputs = load_checked_inputs(config)
    train_graph = build_split_graph(inputs, "train")
    training = build_training_data(config, inputs, train_graph, log)
    folder = cache_path(config, seed)
    train_meta = cache_meta(config, inputs, seed, "train", len(training.text_x))
    train_offsets, train_values = load_cache(folder, "train", train_meta)
    data = build_subgraph_training_data(
        training, inputs, train_graph, padded_rows(train_offsets, train_values)
    )

    val_graph = build_split_graph(inputs, "val")
    val_candidates = split_candidates(config, inputs, "val")
    val_meta = cache_meta(config, inputs, seed, "val", len(val_candidates.query_embeddings))
    val_offsets, val_values = load_cache(folder, "val", val_meta)
    val_scoring = split_scoring(inputs, "val", val_graph, val_candidates, val_offsets, val_values)

    def graph_only_val_score(model: SubgraphSage) -> float:
        # Early stopping looks at the graph score alone (alpha = 0), as in ADR-0047: it is
        # what the loss trains, and alpha is chosen afterwards, both on val only.
        scores = graph_scores(config, model, val_scoring)
        ranks = rerank_ranks(val_candidates, scores)
        return compute_metrics(ranks, config.recall_ks, config.top_k)[config.selection_metric]

    model = SubgraphSage(data.text_x.shape[1])
    result = fit(
        model,
        data,
        graph_only_val_score,
        config.learning_rate,
        config.batch_size,
        config.num_hard_negatives,
        config.temperature,
        config.max_epochs,
        config.patience,
        torch.Generator().manual_seed(seed),
        log,
        loss_fn=subgraph_batch_loss,
    )
    model.load_state_dict(result.best_state)

    scores = graph_scores(config, model, val_scoring)
    sweep = sweep_alpha(val_candidates, scores, config.alpha_grid, config.recall_ks)
    alpha = choose_alpha(sweep, config.selection_metric)
    log(f"seed {seed}: best epoch {result.best_epoch}, alpha {alpha}")

    model_dir.mkdir(parents=True)
    torch.save(result.best_state, model_dir / MODEL_FILE)
    selection = {
        "alpha": alpha,
        "best_epoch": result.best_epoch,
        "history": result.history,
        "alpha_sweep": {str(key): value for key, value in sweep.items()},
        "git": collect_env({})["git_commit"],
        "split_sha256": file_sha256(config.split_file),
        "cache_meta": {"train": train_meta, "val": val_meta},
    }
    with open(model_dir / SELECTION_FILE, "w", encoding="utf-8") as file:
        json.dump(selection, file, indent=2)

    metrics = dict(sweep[alpha])
    metrics.update(run_facts(config, inputs, "val", seed))
    metrics.update(
        prize_k=config.prize_k,
        edge_cost=config.edge_cost,
        best_epoch=result.best_epoch,
        alpha=alpha,
    )
    metrics["alpha_sweep"] = selection["alpha_sweep"]
    metrics["history"] = result.history
    return write_run(config, metrics, seed)


def evaluate_test(config: PcstSageConfig, seed: int, log=print) -> Path:
    """Score TEST once with the saved model and the alpha chosen on val. Chooses nothing.

    Decision (ADR-0009, ADR-0051): refuse a second test run of the same experiment and seed.
      Why: test runs once with the locked config; a repeat means test could steer choices.
      Decided by: FluteInTheRain, ADR-0009; same check as graph_rerank.evaluate_test.
    The test subgraph cache is created here and nowhere else.
    """
    if has_test_run(config, seed):
        raise FileExistsError(f"test of {config.experiment} seed {seed} was already run")
    model_dir = seed_model_dir(config, seed)
    with open(model_dir / SELECTION_FILE, encoding="utf-8") as file:
        selection = json.load(file)
    # The choice was made against one split file; test must be scored against the same one.
    if selection["split_sha256"] != file_sha256(config.split_file):
        raise ValueError("selection.json was made with a different split file than the config")

    folder = cache_path(config, seed)
    if (folder / "test.npz").exists():
        raise FileExistsError(f"{folder / 'test.npz'} exists; delete it on purpose to redo")

    inputs = load_checked_inputs(config)
    pairs = concept_pairs(inputs, config, seed)
    test_graph = build_split_graph(inputs, "test")
    candidates = split_candidates(config, inputs, "test")
    queries = candidates.query_embeddings
    offsets, values = compute_subgraph_rows(config, inputs, pairs, queries, log)
    meta = cache_meta(config, inputs, seed, "test", len(queries))
    save_cache(folder, "test", offsets, values, meta)

    model = SubgraphSage(test_graph.x.shape[1])
    model.load_state_dict(torch.load(model_dir / MODEL_FILE, weights_only=True))
    scoring = split_scoring(inputs, "test", test_graph, candidates, offsets, values)
    scores = graph_scores(config, model, scoring)
    final = fuse_scores(candidates.clip_scores, scores, selection["alpha"])
    metrics = compute_metrics(rerank_ranks(candidates, final), config.recall_ks, config.top_k)
    metrics.update(run_facts(config, inputs, "test", seed))
    metrics.update(
        prize_k=config.prize_k,
        edge_cost=config.edge_cost,
        best_epoch=selection["best_epoch"],
        alpha=selection["alpha"],
    )
    metrics["model_sha256"] = file_sha256(model_dir / MODEL_FILE)
    metrics["model_git_commit"] = selection["git"]
    log(f"seed {seed} test: {metrics}")
    return write_run(config, metrics, seed)
