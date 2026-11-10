"""Train the one-layer GNN to re-rank CLIP's top 100 (ADR-0046, section 3).

For each region description t of a TRAIN image: candidates are its own image (correct)
plus n_hard wrong images sampled from CLIP's top 100 of t, searched among TRAIN images
only. Loss: cross-entropy over cos(t, h) / tau of those candidates (listwise InfoNCE).
The epoch with the best val MRR@100 of the graph score alone is kept (early stopping).
"""

import copy
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

from mmxai.eval.relevance import same_text_relevant_images
from mmxai.features.cache import load_image_features, load_meta, load_text_features
from mmxai.graph.contains import load_contains_graph, train_subgraph
from mmxai.graph.pyg import to_pyg_data
from mmxai.index.image_index import build_image_index, search_images
from mmxai.models.one_layer_sage import OneLayerSage, graph_scores
from mmxai.rerank.combine import rerank_metrics
from mmxai.utils.runs import git_state
from mmxai.utils.tracking import start_run
from mmxai.utils.yaml_config import load_yaml_dataclass

CHECKPOINT_FILE = "model.pt"
CHECKPOINT_META = "meta.json"
# Batch loss goes to W&B every this many steps; one value per step would flood the chart.
LOG_EVERY_STEPS = 50
# The alpha chosen on val for this model, written by the val re-ranking run.
ALPHA_FILE = "chosen_alpha.json"


@dataclass(frozen=True)
class GnnConfig:
    """Settings for training the GNN and re-ranking with it; see configs/gnn_vg.yaml."""

    experiment: str
    dataset: str
    features_dir: str
    graph_dir: str
    top_k: int
    n_hard: int
    temperature: float
    learning_rate: float
    batch_size: int
    max_epochs: int
    patience: int
    val_max_queries: int
    same_text_threshold: float
    recall_ks: list[int]
    alpha_grid: list[float]
    alpha: float | None
    device: str
    checkpoint_dir: str
    output_dir: str
    # W&B project for the training logs (ADR-0047); null = no logging.
    wandb_project: str | None = None


def load_gnn_config(path: str | Path) -> GnnConfig:
    return load_yaml_dataclass(path, GnnConfig)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@dataclass(frozen=True)
class BipartiteGraph:
    """The concept -> image graph as the model reads it. Image rows follow image_ids."""

    image_ids: np.ndarray
    image_split: np.ndarray
    concept_names: np.ndarray
    image_x: torch.Tensor  # (num_images, d)
    concept_x: torch.Tensor  # (num_concepts, d)
    edge_index: torch.Tensor  # (2, E): row 0 = concept row, row 1 = image row
    split_sha256: str


def load_bipartite(config: GnnConfig, train_only: bool, device: torch.device) -> BipartiteGraph:
    """Load the saved graph with its CLIP vectors; train_only keeps train images only."""
    graph, concept_names, concept_features, graph_meta = load_contains_graph(config.graph_dir)
    if train_only:
        graph = train_subgraph(graph)
    feature_meta = load_meta(Path(config.features_dir))
    if feature_meta["dataset"] != config.dataset or feature_meta["is_partial"]:
        raise ValueError(f"{config.features_dir} is not a complete {config.dataset} feature set")
    embeddings, feature_ids = load_image_features(Path(config.features_dir))
    data = to_pyg_data(graph, concept_features, embeddings, feature_ids, graph_meta, feature_meta)

    num_images = len(graph.image_ids)
    # to_pyg_data stores every edge both ways; the model only reads concept -> image.
    from_concept = data.edge_index[0] >= num_images
    edge_index = data.edge_index[:, from_concept].clone()
    edge_index[0] -= num_images
    return BipartiteGraph(
        image_ids=graph.image_ids,
        image_split=graph.image_split,
        concept_names=concept_names,
        image_x=data.x[:num_images].to(device),
        concept_x=data.x[num_images:].to(device),
        edge_index=edge_index.to(device),
        split_sha256=graph_meta["split_sha256"],
    )


def load_texts(
    config: GnnConfig, graph: BipartiteGraph
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (text vectors, graph row of each text's image, text ids) for `graph`'s images."""
    texts = load_text_features(Path(config.features_dir))
    row_of_id = {image_id: row for row, image_id in enumerate(graph.image_ids.tolist())}
    keep = np.array([image_id in row_of_id for image_id in texts["image_ids"].tolist()])
    rows = [row_of_id[image_id] for image_id in texts["image_ids"][keep].tolist()]
    vectors = texts["embeddings"][keep].astype(np.float32)
    return vectors, np.array(rows, dtype=np.int64), texts["text_ids"][keep]


def texts_of_image_matrix(text_image_rows: np.ndarray, num_images: int) -> np.ndarray:
    """(num_images, max texts per image) text rows of each image, padded with -1."""
    counts = np.bincount(text_image_rows, minlength=num_images)
    matrix = np.full((num_images, max(int(counts.max()), 1)), -1, dtype=np.int64)
    order = np.argsort(text_image_rows, kind="stable")
    # Slot of each text among its image's texts: 0, 1, 2, ... per image.
    first_of_image = np.repeat(np.cumsum(counts) - counts, counts)
    slot = np.arange(len(order)) - first_of_image
    matrix[text_image_rows[order], slot] = order
    return matrix


def embed_images(model: OneLayerSage, graph: BipartiteGraph, rows: torch.Tensor) -> torch.Tensor:
    """h of the given (distinct) image rows.

    With one layer and edges concept -> image, h_i needs only the edges into i, so
    running on those edges gives exactly the h of the whole graph, much faster.
    """
    position = torch.full((len(graph.image_x),), -1, dtype=torch.long, device=rows.device)
    position[rows] = torch.arange(len(rows), device=rows.device)
    targets = position[graph.edge_index[1]]
    keep = targets >= 0
    edge_index = torch.stack([graph.edge_index[0, keep], targets[keep]])
    return model(graph.concept_x, graph.image_x[rows], edge_index)


def sample_hard_negatives(
    query: torch.Tensor,
    positive_rows: torch.Tensor,
    image_x: torch.Tensor,
    text_x: torch.Tensor,
    texts_of_image: torch.Tensor,
    top_k: int,
    n_hard: int,
    same_text_threshold: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample n_hard wrong images per query from its CLIP top_k (all rows of image_x).

    Returns (rows, valid), both (B, n_hard). The query's own image and every image owning
    the same sentence (ADR-0043) are never negatives; if fewer than n_hard images are
    left, the missing slots are marked valid = False.
    """
    top = (query @ image_x.T).topk(top_k, dim=1).indices  # (B, K)

    candidate_texts = texts_of_image[top]  # (B, K, T), -1 = no text
    text_vectors = text_x[candidate_texts.clamp(min=0)]  # (B, K, T, d)
    similarity = torch.einsum("bd,bktd->bkt", query, text_vectors)
    similarity = similarity.masked_fill(candidate_texts < 0, -1.0)
    same_text = (similarity >= same_text_threshold).any(dim=2)

    allowed = (top != positive_rows[:, None]) & ~same_text
    # A random key per candidate; disallowed ones get -1 so they are picked last.
    keys = torch.rand(top.shape, generator=generator, device=generator.device).to(top.device)
    keys = keys.masked_fill(~allowed, -1.0)
    picked = keys.topk(n_hard, dim=1).indices
    return top.gather(1, picked), allowed.gather(1, picked)


def listwise_loss(
    model: OneLayerSage,
    graph: BipartiteGraph,
    query: torch.Tensor,
    positive_rows: torch.Tensor,
    negative_rows: torch.Tensor,
    negative_valid: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Cross-entropy over cos(q, h) / tau of {correct image, hard negatives}; correct = 0."""
    candidates = torch.cat([positive_rows[:, None], negative_rows], dim=1)
    unique_rows, inverse = torch.unique(candidates, return_inverse=True)
    h = embed_images(model, graph, unique_rows)[inverse]  # (B, 1 + n_hard, d)
    logits = graph_scores(query[:, None, :], h) / temperature
    valid = torch.cat([torch.ones_like(negative_valid[:, :1]), negative_valid], dim=1)
    logits = logits.masked_fill(~valid, float("-inf"))
    target = torch.zeros(len(query), dtype=torch.long, device=query.device)
    return nn.functional.cross_entropy(logits, target)


@torch.no_grad()
def score_candidates(
    model: OneLayerSage,
    graph: BipartiteGraph,
    gallery_rows: np.ndarray,
    query_x: np.ndarray,
    top_k: int,
    query_chunk: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CLIP top_k of each query in the gallery, with cosine and graph score of each.

    Candidates come from the same exact FAISS search as the CLIP baseline (ADR-0043,
    ADR-0046 section 4). Returns (candidate_rows, cosine, graph), all (Q, top_k).
    """
    model.eval()
    image_x = graph.image_x.cpu().numpy()
    index = build_image_index(image_x[gallery_rows], gallery_rows)
    cosine, candidate_rows = search_images(index, query_x, top_k)

    rows = torch.from_numpy(np.asarray(gallery_rows, dtype=np.int64)).to(graph.image_x.device)
    h = embed_images(model, graph, rows)
    position_of_row = {row: position for position, row in enumerate(gallery_rows.tolist())}
    positions = np.vectorize(position_of_row.__getitem__)(candidate_rows)

    graph_score = np.empty(candidate_rows.shape, dtype=np.float32)
    # Chunks keep memory at query_chunk x top_k x d.
    for start in range(0, len(query_x), query_chunk):
        stop = start + query_chunk
        chunk_h = h[torch.from_numpy(positions[start:stop]).to(h.device)]
        chunk_q = torch.from_numpy(query_x[start:stop]).to(h.device)[:, None, :]
        graph_score[start:stop] = graph_scores(chunk_q, chunk_h).cpu().numpy()
    return candidate_rows, cosine, graph_score


def split_queries(
    config: GnnConfig, graph: BipartiteGraph, split_name: str
) -> tuple[np.ndarray, np.ndarray, list[set[int]], np.ndarray]:
    """Gallery rows, query vectors, relevant rows and text ids of one split (ADR-0043)."""
    gallery_rows = np.flatnonzero(graph.image_split == split_name)
    text_x, text_image_rows, text_ids = load_texts(config, graph)
    in_split = np.flatnonzero(graph.image_split[text_image_rows] == split_name)
    query_x = text_x[in_split]
    relevant = same_text_relevant_images(
        query_x, text_image_rows[in_split], config.same_text_threshold
    )
    return gallery_rows, query_x, relevant, text_ids[in_split]


def checkpoint_folder(config: GnnConfig, seed: int) -> Path:
    return Path(config.checkpoint_dir) / f"{config.experiment}_seed{seed}"


def save_chosen_alpha(config: GnnConfig, seed: int, alpha: float) -> None:
    """Store the alpha chosen on val next to the model it was chosen for."""
    with open(checkpoint_folder(config, seed) / ALPHA_FILE, "w", encoding="utf-8") as file:
        json.dump({"alpha": alpha, "chosen_on": "val"}, file, indent=2)


def load_chosen_alpha(config: GnnConfig, seed: int) -> float:
    path = checkpoint_folder(config, seed) / ALPHA_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run scripts/evaluate_rerank.py --split val --seed {seed} first"
        )
    with open(path, encoding="utf-8") as file:
        return float(json.load(file)["alpha"])


def train_gnn(config: GnnConfig, seed: int) -> Path:
    """Train with early stopping on val, save the best weights, return the checkpoint folder."""
    output_dir = checkpoint_folder(config, seed)
    if output_dir.exists():
        raise FileExistsError(f"{output_dir} already exists; delete it to train again")
    torch.manual_seed(seed)
    device = resolve_device(config.device)
    generator = torch.Generator(device=device).manual_seed(seed)

    # PDF section 6, ADR-0046: the training graph holds train images only.
    train_graph = load_bipartite(config, train_only=True, device=device)
    if set(train_graph.image_split.tolist()) != {"train"}:
        raise ValueError("the training graph contains non-train images")
    text_x_np, text_rows_np, _ = load_texts(config, train_graph)
    text_x = torch.from_numpy(text_x_np).to(device)
    text_image_rows = torch.from_numpy(text_rows_np).to(device)
    texts_of_image = torch.from_numpy(
        texts_of_image_matrix(text_rows_np, len(train_graph.image_ids))
    ).to(device)

    # Val: a fixed subset of queries, the same for every seed, so seeds compare fairly.
    val_graph = load_bipartite(config, train_only=False, device=device)
    val_gallery, val_query_x, val_relevant, _ = split_queries(config, val_graph, "val")
    subset = np.sort(
        np.random.default_rng(0).permutation(len(val_query_x))[: config.val_max_queries]
    )
    val_query_x = val_query_x[subset]
    val_relevant = [val_relevant[i] for i in subset]

    model = OneLayerSage(train_graph.image_x.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    mrr_name = f"MRR@{config.top_k}"

    def val_mrr() -> float:
        candidates, cosine, graph = score_candidates(
            model, val_graph, val_gallery, val_query_x, config.top_k
        )
        # alpha = 0: the graph score alone decides, so the metric measures the GNN.
        metrics, _ = rerank_metrics(candidates, cosine, graph, 0.0, val_relevant, config.recall_ks)
        return metrics[mrr_name]

    # Epoch 0 = untrained model = plain CLIP; training must beat it to be kept.
    best_mrr = val_mrr()
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    history = [{"epoch": 0, "train_loss": None, f"val_{mrr_name}": best_mrr}]
    epochs_without_gain = 0
    run = start_run(
        config.wandb_project, f"{config.experiment}_seed{seed}", asdict(config) | {"seed": seed}
    )
    run.log({"epoch": 0, f"val/{mrr_name}": best_mrr})
    step = 0
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        order = torch.randperm(len(text_x), generator=generator, device=device)
        losses = []
        for start in range(0, len(order), config.batch_size):
            batch = order[start : start + config.batch_size]
            query = text_x[batch]
            positive_rows = text_image_rows[batch]
            negative_rows, negative_valid = sample_hard_negatives(
                query,
                positive_rows,
                train_graph.image_x,
                text_x,
                texts_of_image,
                config.top_k,
                config.n_hard,
                config.same_text_threshold,
                generator,
            )
            loss = listwise_loss(
                model,
                train_graph,
                query,
                positive_rows,
                negative_rows,
                negative_valid,
                config.temperature,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            step += 1
            if step % LOG_EVERY_STEPS == 0:
                run.log({"train/batch_loss": loss.item(), "step": step})

        mrr = val_mrr()
        history.append(
            {"epoch": epoch, "train_loss": float(np.mean(losses)), f"val_{mrr_name}": mrr}
        )
        print(f"epoch {epoch}: train loss {np.mean(losses):.4f}, val {mrr_name} {mrr:.4f}")
        run.log({"epoch": epoch, "train/loss": float(np.mean(losses)), f"val/{mrr_name}": mrr})
        if mrr > best_mrr:
            best_mrr, best_epoch, epochs_without_gain = mrr, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= config.patience:
                break

    output_dir.mkdir(parents=True)
    torch.save(best_state, output_dir / CHECKPOINT_FILE)
    meta = {
        "seed": seed,
        "best_epoch": best_epoch,
        f"best_val_{mrr_name}": best_mrr,
        "history": history,
        "split_sha256": train_graph.split_sha256,
        "git": git_state(),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": asdict(config),
    }
    with open(output_dir / CHECKPOINT_META, "w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2)
    run.summary["best_epoch"] = best_epoch
    run.summary[f"best_val/{mrr_name}"] = best_mrr
    run.finish()
    return output_dir


def load_trained_model(
    config: GnnConfig, seed: int, dim: int, device: torch.device
) -> OneLayerSage:
    """Load the weights saved by train_gnn for this experiment and seed, in eval mode."""
    path = checkpoint_folder(config, seed) / CHECKPOINT_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run scripts/train_gnn.py with --seed {seed}")
    model = OneLayerSage(dim).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model
