"""CLIP + Graph on Visual Genome: train on train, choose on val, test once (ADR-0046, ADR-0047).

Two separate steps, so test can never feed back into a choice:
    train_and_select  train each seed, early-stop and choose alpha on VAL, save the model
    evaluate_test     load a saved model and its alpha, score TEST once, choose nothing

Every split is scored on its own subgraph (split_subgraph). This is exact for both
models: an image's score depends only on its own contains edges and, for the ADR-0048
model, on the concept -> concept edges, which are counted on train and shared by all.
"""

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch_geometric

from mmxai.data.splits import load_split
from mmxai.eval.metrics import compute_metrics
from mmxai.features.cache import load_image_features, load_meta, load_text_features
from mmxai.graph.contains import ContainsGraph, load_contains_graph, split_subgraph
from mmxai.graph.cooccurrence import ConceptEdges, load_concept_edges
from mmxai.graph.pyg import to_pyg_data
from mmxai.models.sage import (
    ConceptToImageSage,
    CooccurrenceSage,
    concept_to_image_edges,
    neighbor_mean,
    second_hop_mean,
)
from mmxai.models.train import (
    TrainingData,
    fit,
    hard_negative_pool,
    texts_with_positive_in_candidates,
)
from mmxai.rerank.fusion import (
    Candidates,
    candidate_graph_scores,
    choose_alpha,
    fuse_scores,
    rerank_ranks,
    retrieve_candidates,
    sweep_alpha,
)
from mmxai.utils.runs import (
    collect_env,
    file_sha256,
    make_run_dir,
    test_run_exists,
    write_run_files,
)
from mmxai.utils.seed import set_seed
from mmxai.utils.yaml_config import load_yaml_dataclass

MODEL_FILE = "model.pt"
SELECTION_FILE = "selection.json"


@dataclass(frozen=True)
class GraphRerankConfig:
    """Settings of one CLIP + Graph experiment (configs/clip_graph_vg.yaml)."""

    experiment: str
    dataset: str
    split_file: str
    features_dir: str
    graph_dir: str
    # Concept -> concept co-occurrence edges (ADR-0048); null = the one-layer ADR-0046 model.
    concept_edges_dir: str | None
    model_dir: str
    use_graph: bool
    top_k: int
    recall_ks: list[int]
    same_text_threshold: float
    num_hard_negatives: int
    temperature: float
    learning_rate: float
    batch_size: int
    max_epochs: int
    patience: int
    alpha_grid: list[float]
    selection_metric: str
    output_dir: str


def load_graph_rerank_config(path: str | Path) -> GraphRerankConfig:
    """Load the YAML config; reject a selection metric that is not reported."""
    config = load_yaml_dataclass(path, GraphRerankConfig)
    if config.selection_metric != f"MRR@{config.top_k}":
        raise ValueError(f"selection_metric must be MRR@{config.top_k}")
    if config.concept_edges_dir is not None and not config.use_graph:
        # The no-graph control has no concept messages, so concept edges would do nothing.
        raise ValueError("concept_edges_dir needs use_graph: true")
    return config


Model = ConceptToImageSage | CooccurrenceSage


def make_model(config: GraphRerankConfig, dim: int) -> Model:
    """The model the config asks for, untrained (it scores exactly like CLIP)."""
    if config.concept_edges_dir is None:
        return ConceptToImageSage(dim, use_graph=config.use_graph)
    return CooccurrenceSage(dim)


@dataclass(frozen=True)
class SplitGraph:
    """One split's images, in graph row order, with the concept -> image edges."""

    image_ids: np.ndarray
    x: torch.Tensor  # images first, then concepts
    edge_index: torch.Tensor
    # Concept -> concept edges in concept rows (ADR-0048), or None for the one-layer model.
    concept_edge_index: torch.Tensor | None


@dataclass(frozen=True)
class Inputs:
    """Everything loaded from disk once and shared by every seed."""

    graph: ContainsGraph
    concept_features: np.ndarray
    graph_meta: dict
    image_embeddings: np.ndarray
    image_ids: np.ndarray
    texts: dict[str, np.ndarray]
    feature_meta: dict
    split: dict[str, np.ndarray]
    concept_edges: ConceptEdges | None


def load_concept_edges_for(config: GraphRerankConfig, graph: ContainsGraph) -> ConceptEdges | None:
    """Load the co-occurrence edges and check they belong to this graph and split."""
    if config.concept_edges_dir is None:
        return None
    edges, meta = load_concept_edges(config.concept_edges_dir)
    if meta["split_sha256"] != file_sha256(config.split_file):
        raise ValueError(f"{config.concept_edges_dir} was counted on a different split file")
    # Edge rows are concept rows, so both files must list the same concepts in the same order.
    if not np.array_equal(edges.concept_keys, graph.concept_keys):
        raise ValueError(f"{config.concept_edges_dir} does not use the concepts of the graph")
    return edges


def load_inputs(config: GraphRerankConfig) -> Inputs:
    """Load features, graph and split, and check they belong together."""
    feature_meta = load_meta(config.features_dir)
    if feature_meta["dataset"] != config.dataset or feature_meta["is_partial"]:
        raise ValueError(f"{config.features_dir} is not a complete {config.dataset} feature set")
    graph, _, concept_features, graph_meta = load_contains_graph(config.graph_dir)
    # The graph records each image's split; it must be the split we evaluate with.
    if graph_meta["split_sha256"] != file_sha256(config.split_file):
        raise ValueError(f"{config.graph_dir} was built with a different split file")
    image_embeddings, image_ids = load_image_features(config.features_dir)
    return Inputs(
        graph=graph,
        concept_features=concept_features,
        graph_meta=graph_meta,
        image_embeddings=image_embeddings,
        image_ids=image_ids,
        texts=load_text_features(config.features_dir),
        feature_meta=feature_meta,
        split=load_split(config.split_file),
        concept_edges=load_concept_edges_for(config, graph),
    )


def build_split_graph(inputs: Inputs, split_name: str) -> SplitGraph:
    """The images of one split, their concepts and only the concept -> image edges."""
    subgraph = split_subgraph(inputs.graph, split_name)
    data = to_pyg_data(
        subgraph,
        inputs.concept_features,
        inputs.image_embeddings,
        inputs.image_ids,
        inputs.graph_meta,
        inputs.feature_meta,
    )
    concept_edge_index = None
    if inputs.concept_edges is not None:
        edges = inputs.concept_edges
        concept_edge_index = torch.from_numpy(np.stack([edges.edge_source, edges.edge_target]))
    return SplitGraph(subgraph.image_ids, data.x, concept_to_image_edges(data), concept_edge_index)


def image_vectors(model: Model, split_graph: SplitGraph) -> np.ndarray:
    """h of every image of the split, with the full SAGEConv (the path explain/ uses)."""
    num_images = len(split_graph.image_ids)
    model.eval()
    with torch.no_grad():
        if split_graph.concept_edge_index is None:
            h = model(split_graph.x, split_graph.edge_index)
        else:
            h = model(
                split_graph.x, split_graph.edge_index, split_graph.concept_edge_index, num_images
            )
    return h[:num_images].numpy()


def build_training_data(
    config: GraphRerankConfig, inputs: Inputs, train_graph: SplitGraph, log=print
) -> TrainingData:
    """Train images, their concept means, train texts and their hard negatives."""
    num_images = len(train_graph.image_ids)
    image_x = train_graph.x[:num_images]
    mean = neighbor_mean(train_graph.x, train_graph.edge_index, num_images)
    second_mean = None
    if train_graph.concept_edge_index is not None:
        second_mean = second_hop_mean(
            train_graph.x, train_graph.edge_index, train_graph.concept_edge_index, num_images
        )

    # Only texts of train images; graph image ids are sorted, so searchsorted gives rows.
    is_train_text = np.isin(inputs.texts["image_ids"], train_graph.image_ids)
    text_x = inputs.texts["embeddings"][is_train_text]
    text_image_row = np.searchsorted(
        train_graph.image_ids, inputs.texts["image_ids"][is_train_text]
    )

    candidate_rows, is_negative = hard_negative_pool(
        image_x.numpy(), text_x, text_image_row, config.top_k, config.same_text_threshold
    )
    keep = texts_with_positive_in_candidates(text_image_row, candidate_rows)
    num_kept = int(keep.sum())
    log(f"{num_kept} of {len(keep)} train texts have their image in CLIP's top {config.top_k}")
    return TrainingData(
        image_x=image_x,
        neighbor_mean=mean,
        text_x=torch.from_numpy(text_x[keep]),
        text_image_row=torch.from_numpy(text_image_row[keep]),
        candidate_rows=torch.from_numpy(candidate_rows[keep]),
        is_negative=torch.from_numpy(is_negative[keep]),
        second_hop_mean=second_mean,
    )


def split_candidates(config: GraphRerankConfig, inputs: Inputs, split_name: str) -> Candidates:
    """CLIP's top-k for every text of the split (the same as the CLIP baseline)."""
    return retrieve_candidates(
        inputs.image_embeddings,
        inputs.image_ids,
        inputs.texts["embeddings"],
        inputs.texts["image_ids"],
        inputs.split[split_name],
        config.top_k,
        config.same_text_threshold,
    )


@dataclass(frozen=True)
class Prepared:
    """The slow, seed-independent part: computed once, then reused for every seed."""

    inputs: Inputs
    training_data: TrainingData
    val_graph: SplitGraph
    val_candidates: Candidates


def prepare(config: GraphRerankConfig, log=print) -> Prepared:
    started = time.time()
    inputs = load_inputs(config)
    train_graph = build_split_graph(inputs, "train")
    log(
        f"train graph: {len(train_graph.image_ids)} images, {train_graph.edge_index.shape[1]} edges"
    )
    training_data = build_training_data(config, inputs, train_graph, log)
    log(
        f"hard negatives for {len(training_data.text_x)} train texts ({time.time() - started:.0f}s)"
    )
    val_graph = build_split_graph(inputs, "val")
    val_candidates = split_candidates(config, inputs, "val")
    log(f"val: {len(val_candidates.candidate_ids)} queries ({time.time() - started:.0f}s)")
    return Prepared(inputs, training_data, val_graph, val_candidates)


def graph_only_val_score(config: GraphRerankConfig, model: Model, prepared: Prepared) -> float:
    """Val MRR of the graph score alone (alpha = 0), the early-stopping signal.

    Decision (ADR-0047): early-stop on the graph score alone, then choose alpha.
      Why: it is what the loss trains; choosing epochs on the fused score would mix
      the two choices. Both are made on val only.
      Decided by: Claude Code proposal, 2026-11-12; not yet approved by the team.
    """
    h = image_vectors(model, prepared.val_graph)
    scores = candidate_graph_scores(prepared.val_candidates, h, prepared.val_graph.image_ids)
    ranks = rerank_ranks(prepared.val_candidates, scores)
    return compute_metrics(ranks, config.recall_ks, config.top_k)[config.selection_metric]


def library_versions() -> dict[str, str]:
    import faiss

    return {
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "faiss": faiss.__version__,
    }


def seed_model_dir(config: GraphRerankConfig, seed: int) -> Path:
    return Path(config.model_dir) / f"seed{seed}"


def train_and_select(config: GraphRerankConfig, prepared: Prepared, seed: int, log=print) -> Path:
    """Train one seed, choose epoch and alpha on val, save the model, write a val run."""
    model_dir = seed_model_dir(config, seed)
    if model_dir.exists():
        raise FileExistsError(f"{model_dir} exists; delete it to train this seed again")

    set_seed(seed)
    model = make_model(config, prepared.training_data.text_x.shape[1])
    result = fit(
        model,
        prepared.training_data,
        lambda current: graph_only_val_score(config, current, prepared),
        config.learning_rate,
        config.batch_size,
        config.num_hard_negatives,
        config.temperature,
        config.max_epochs,
        config.patience,
        torch.Generator().manual_seed(seed),
        log,
    )
    model.load_state_dict(result.best_state)

    h = image_vectors(model, prepared.val_graph)
    scores = candidate_graph_scores(prepared.val_candidates, h, prepared.val_graph.image_ids)
    sweep = sweep_alpha(prepared.val_candidates, scores, config.alpha_grid, config.recall_ks)
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
    }
    with open(model_dir / SELECTION_FILE, "w", encoding="utf-8") as file:
        json.dump(selection, file, indent=2)

    metrics = dict(sweep[alpha])
    metrics.update(run_facts(config, prepared.inputs, "val", seed, alpha, result.best_epoch))
    metrics["alpha_sweep"] = selection["alpha_sweep"]
    metrics["history"] = result.history
    return write_run(config, metrics, seed)


def run_facts(
    config: GraphRerankConfig, inputs: Inputs, split_name: str, seed: int, alpha: float, epoch: int
) -> dict:
    """What every metrics.json must say about where its number came from (ADR-0009, ADR-0010)."""
    return {
        "split": split_name,
        "split_file": config.split_file,
        "split_sha256": file_sha256(config.split_file),
        "seed": seed,
        "alpha": alpha,
        "best_epoch": epoch,
        "use_graph": config.use_graph,
        "graph_dir": config.graph_dir,
        "concept_edges_dir": config.concept_edges_dir,
        "graph_edge_source": inputs.graph_meta["edge_source"],
        "graph_git_commit": inputs.graph_meta["git"]["git_commit"],
        "features_code_hash": inputs.feature_meta["code_hash"],
    }


def write_run(config: GraphRerankConfig, metrics: dict, seed: int) -> Path:
    run_dir = make_run_dir(config.output_dir, config.experiment, seed, datetime.now())
    write_run_files(run_dir, asdict(config), metrics, collect_env(library_versions()))
    return run_dir


def has_test_run(config: GraphRerankConfig, seed: int) -> bool:
    """True if results/runs already holds a test run of this experiment and seed."""
    return test_run_exists(config.output_dir, config.experiment, seed)


def evaluate_test(config: GraphRerankConfig, seed: int, log=print) -> Path:
    """Score TEST once with the saved model and the alpha chosen on val. Chooses nothing.

    Decision (ADR-0009, ADR-0047): refuse a second test run of the same experiment and seed.
      Why: test runs once with the locked config; a repeat means test could steer choices.
      Decided by: FluteInTheRain, ADR-0009; this check proposed by Claude Code, 2026-11-12.
    """
    if has_test_run(config, seed):
        raise FileExistsError(f"test of {config.experiment} seed {seed} was already run")
    model_dir = seed_model_dir(config, seed)
    with open(model_dir / SELECTION_FILE, encoding="utf-8") as file:
        selection = json.load(file)

    inputs = load_inputs(config)
    test_graph = build_split_graph(inputs, "test")
    model = make_model(config, test_graph.x.shape[1])
    model.load_state_dict(torch.load(model_dir / MODEL_FILE, weights_only=True))

    candidates = split_candidates(config, inputs, "test")
    h = image_vectors(model, test_graph)
    scores = candidate_graph_scores(candidates, h, test_graph.image_ids)
    final = fuse_scores(candidates.clip_scores, scores, selection["alpha"])
    metrics = compute_metrics(rerank_ranks(candidates, final), config.recall_ks, config.top_k)
    metrics.update(
        run_facts(config, inputs, "test", seed, selection["alpha"], selection["best_epoch"])
    )
    metrics["model_sha256"] = file_sha256(model_dir / MODEL_FILE)
    metrics["model_git_commit"] = selection["git"]
    log(f"seed {seed} test: {metrics}")
    return write_run(config, metrics, seed)
