"""Score the Visual Genome models on the Vietnamese MMKG without retraining them (ADR-0053).

    VG weights (W_self, W_nb, ...) + VG alpha   ->  applied to a new graph:
        image nodes   = 380 MMKG images (frozen CLIP vectors)
        concept nodes = 119 MMKG entities ("a photo of a {name_en}")
        contains      = each image -> its k closest entities by CLIP cosine (never entity_id)
        concept edges = MMKG triples, both directions (two-layer model only)

The weights are 512 x 512 maps on frozen CLIP vectors, not tied to VG's concept list,
so they run on any graph. Nothing here is chosen on MMKG: k, alpha and the queries
are fixed before any run (ADR-0053). Pure numpy/torch; CLIP encoding is in the script.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from mmxai.data.records import ImageRecord
from mmxai.eval.metrics import compute_metrics
from mmxai.graph.clip_edges import clip_topk_concepts
from mmxai.models.sage import ConceptToImageSage, CooccurrenceSage
from mmxai.rerank.fusion import Candidates, candidate_graph_scores, fuse_scores, rerank_ranks


@dataclass(frozen=True)
class TransferGraph:
    """The MMKG graph in the node layout of mmxai.graph.pyg: images first, then concepts."""

    image_ids: np.ndarray  # (num_images,) integer ids, row order of x
    x: torch.Tensor  # (num_images + num_concepts, dim)
    edge_index: torch.Tensor  # concept -> image, in node rows
    concept_edge_index: torch.Tensor  # concept -> concept, in CONCEPT rows (0 = first concept)


def contains_edge_index(
    image_ids: np.ndarray,
    image_features: np.ndarray,
    concept_keys: list[str],
    concept_features: np.ndarray,
    k: int,
) -> torch.Tensor:
    """Concept -> image edges from CLIP top-k, in node rows (concept row = num_images + j).

    Reads only CLIP vectors: the image's entity_id (the answer) is never an input.
    """
    nearest = clip_topk_concepts(image_ids, image_features, concept_keys, concept_features, k)
    concept_row = {key: row for row, key in enumerate(concept_keys)}
    num_images = len(image_ids)
    sources = []
    targets = []
    for image_row, image_id in enumerate(image_ids.tolist()):
        # Sorted so the edge order, and so any float sum over it, is always the same.
        for key in sorted(nearest[image_id]):
            sources.append(num_images + concept_row[key])
            targets.append(image_row)
    return torch.tensor([sources, targets], dtype=torch.long)


def triple_edge_index(triples: list[dict], concept_keys: list[str]) -> torch.Tensor:
    """Every triple as two concept -> concept edges (head <-> tail), in concept rows.

    Decision (ADR-0053 rule 4): both directions, relation name ignored.
      Why: the VG co-occurrence edges the two-layer model was trained on are symmetric
      and unlabelled, so this is the closest MMKG equivalent.
      Decided by: qphulong, 2026-11-18 (proposed by Claude Code).
    """
    row_of = {key: row for row, key in enumerate(concept_keys)}
    sources = []
    targets = []
    for triple in triples:
        head = row_of[triple["head_id"]]
        tail = row_of[triple["tail_id"]]
        sources.extend([head, tail])
        targets.extend([tail, head])
    return torch.tensor([sources, targets], dtype=torch.long)


def build_transfer_graph(
    image_ids: np.ndarray,
    image_features: np.ndarray,
    concept_keys: list[str],
    concept_features: np.ndarray,
    triples: list[dict],
    k: int,
) -> TransferGraph:
    x = torch.from_numpy(np.concatenate([image_features, concept_features]).astype(np.float32))
    return TransferGraph(
        image_ids=image_ids,
        x=x,
        edge_index=contains_edge_index(
            image_ids, image_features, concept_keys, concept_features, k
        ),
        concept_edge_index=triple_edge_index(triples, concept_keys),
    )


def load_vg_model(model_file: str, two_layer: bool, dim: int) -> torch.nn.Module:
    """The trained VG model, unchanged: its weights do not depend on the concept list."""
    model = CooccurrenceSage(dim) if two_layer else ConceptToImageSage(dim)
    model.load_state_dict(torch.load(model_file, weights_only=True))
    model.eval()
    return model


def image_graph_vectors(model: torch.nn.Module, graph: TransferGraph) -> np.ndarray:
    """h of every MMKG image, with the same forward pass as on VG test."""
    num_images = len(graph.image_ids)
    with torch.no_grad():
        if isinstance(model, CooccurrenceSage):
            h = model(graph.x, graph.edge_index, graph.concept_edge_index, num_images)
        else:
            h = model(graph.x, graph.edge_index)
    return h[:num_images].numpy()


def clip_candidates(
    query_features: np.ndarray,
    image_features: np.ndarray,
    image_ids: np.ndarray,
    relevant: list[set[int]],
    top_k: int,
) -> Candidates:
    """CLIP's top_k images of every query, best first, with the given correct images.

    MMKG has 380 images, so an exact dot product replaces the FAISS index; the
    relevant sets come from the query file (ADR-0053 rule 7), not from equal sentences.
    """
    top_k = min(top_k, len(image_ids))
    similarity = query_features @ image_features.T
    # Stable sort on the negated score: ties keep image order, so runs are repeatable.
    order = np.argsort(-similarity, axis=1, kind="stable")[:, :top_k]
    clip_scores = np.take_along_axis(similarity, order, axis=1).astype(np.float32)
    return Candidates(query_features, image_ids[order], clip_scores, relevant)


def group_metrics(
    ranks: list[int | None],
    groups: dict[str, list[str]],
    recall_ks: list[int],
    top_k: int,
) -> dict:
    """{grouping: {group: metrics}} in the format scripts/summarize_runs.py --kind groups reads.

    groups maps a grouping name to one label per query, e.g. "type": ["direct", ...].
    """
    result = {}
    for grouping, labels in groups.items():
        result[grouping] = {}
        for label in sorted(set(labels)):
            group_ranks = [rank for rank, own in zip(ranks, labels, strict=True) if own == label]
            metrics = compute_metrics(group_ranks, recall_ks, top_k)
            metrics["fraction_queries"] = len(group_ranks) / len(ranks)
            result[grouping][label] = metrics
    return result


def transfer_ranks(
    candidates: Candidates,
    image_h: np.ndarray | None,
    image_ids: np.ndarray,
    alpha: float,
) -> list[int | None]:
    """First-relevant rank of every query; image_h=None is plain CLIP (CLIP's own order)."""
    if image_h is None:
        return rerank_ranks(candidates, candidates.clip_scores)
    graph_scores = candidate_graph_scores(candidates, image_h, image_ids)
    return rerank_ranks(candidates, fuse_scores(candidates.clip_scores, graph_scores, alpha))


# --- Features: encode once, cache, reuse for every experiment and seed. ----------------------

FEATURE_FILE = "features.npz"
FEATURE_META_FILE = "meta.json"


def image_records(images_dir: Path, image_keys: list[str]) -> list[ImageRecord]:
    """One record per MMKG image; the integer id is the row in sorted image_keys.

    The GNN code needs integer image ids; the MMKG ids ("Q36160_8498a0eb") are kept
    next to them in the cache, so every row can be traced back.
    """
    records = []
    for number, key in enumerate(image_keys):
        matches = sorted(images_dir.glob(f"{key}.*"))
        if len(matches) != 1:
            raise FileNotFoundError(f"expected one file for image {key} in {images_dir}")
        records.append(ImageRecord(number, matches[0]))
    return records


def encode_mmkg(
    config: dict, entities: dict, queries: list[dict], image_keys: list[str]
) -> dict[str, np.ndarray]:
    """CLIP vectors of the images, the entities (concept_prompt + name_en) and the queries."""
    # Imported here so the scoring functions above need no open_clip.
    from mmxai.features.clip_encoder import encode_images, encode_texts, load_frozen_clip

    bundle = load_frozen_clip(config["model"], config["pretrained"], config["device"])
    records = image_records(Path(config["images_dir"]), image_keys)
    image_features, image_ids, failed = encode_images(
        bundle, records, config["batch_size"], num_workers=0
    )
    if failed:
        raise ValueError(f"{len(failed)} MMKG images could not be read, e.g. row {failed[0]}")

    concept_keys = sorted(entities)
    prompts = [config["concept_prompt"].format(entities[key].name_en) for key in concept_keys]
    return {
        "image_keys": np.array(image_keys),
        "image_ids": image_ids,
        "image_features": image_features,
        "concept_keys": np.array(concept_keys),
        "concept_features": encode_texts(bundle, prompts, config["batch_size"]),
        "query_qids": np.array([query["qid"] for query in queries]),
        "query_features": encode_texts(
            bundle, [query["text"] for query in queries], config["batch_size"]
        ),
    }


def save_features(arrays: dict[str, np.ndarray], meta: dict, features_dir: Path) -> None:
    features_dir.mkdir(parents=True, exist_ok=True)
    np.savez(features_dir / FEATURE_FILE, **arrays)
    with open(features_dir / FEATURE_META_FILE, "w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2)


def load_features(features_dir: Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(features_dir / FEATURE_FILE) as data:
        arrays = {name: data[name] for name in data.files}
    with open(features_dir / FEATURE_META_FILE, encoding="utf-8") as file:
        meta = json.load(file)
    return arrays, meta


# --- One run: one experiment, one seed. -------------------------------------------------------


def query_groups(queries: list[dict]) -> dict[str, list[str]]:
    """Group labels per query (ADR-0053 rule 8): type x language, and indirect specificity."""
    groups: dict[str, list[str]] = {"type": [], "language": [], "type_language": [], "indirect": []}
    for query in queries:
        groups["type"].append(query["type"])
        groups["language"].append(query["lang"])
        groups["type_language"].append(f"{query['type']}_{query['lang']}")
        if query["type"] == "direct":
            groups["indirect"].append("direct")
        else:
            # Specific = exactly one item fits the description; generic = several do.
            how = "specific" if len(query["relevant_items"]) == 1 else "generic"
            groups["indirect"].append(f"indirect_{how}_{query['lang']}")
    return groups


def run_transfer(
    config: dict,
    experiment: dict,
    seed: int,
    arrays: dict[str, np.ndarray],
    queries: list[dict],
    triples: list[dict],
) -> dict:
    """Metrics of one experiment and seed on MMKG. Chooses nothing: alpha comes from VG val."""
    if [query["qid"] for query in queries] != arrays["query_qids"].tolist():
        raise ValueError("the feature cache was built from a different query file")
    image_ids = arrays["image_ids"]
    id_of_key = dict(zip(arrays["image_keys"].tolist(), image_ids.tolist(), strict=True))
    relevant = [{id_of_key[key] for key in query["relevant"]} for query in queries]
    candidates = clip_candidates(
        arrays["query_features"], arrays["image_features"], image_ids, relevant, config["top_k"]
    )

    facts = {"seed": seed, "experiment": experiment["name"]}
    if experiment["model_dir"] is None:
        alpha = 1.0
        ranks = transfer_ranks(candidates, None, image_ids, alpha)
    else:
        seed_dir = Path(experiment["model_dir"]) / f"seed{seed}"
        with open(seed_dir / "selection.json", encoding="utf-8") as file:
            selection = json.load(file)
        alpha = selection["alpha"]
        graph = build_transfer_graph(
            image_ids,
            arrays["image_features"],
            arrays["concept_keys"].tolist(),
            arrays["concept_features"],
            triples,
            config["clip_topk_k"],
        )
        model = load_vg_model(
            seed_dir / "model.pt", experiment["two_layer"], arrays["image_features"].shape[1]
        )
        h = image_graph_vectors(model, graph)
        ranks = transfer_ranks(candidates, h, image_ids, alpha)
        facts["model_git_commit"] = selection.get("git")
        facts["vg_best_epoch"] = selection["best_epoch"]
        facts["num_contains_edges"] = int(graph.edge_index.shape[1])
        facts["num_concept_edges"] = int(graph.concept_edge_index.shape[1])
    facts["alpha"] = alpha

    top_k = candidates.candidate_ids.shape[1]
    metrics = compute_metrics(ranks, config["recall_ks"], top_k)
    metrics["groups"] = group_metrics(ranks, query_groups(queries), config["recall_ks"], top_k)
    metrics.update(facts)
    return metrics


def query_spread(arrays: dict[str, np.ndarray], queries: list[dict]) -> list[dict]:
    """Per (type, lang) group: how spread out CLIP's query vectors and top-1 images are.

    A text encoder that cannot read a language maps all its sentences close together, so
    they share one top-1 image. Plain CLIP only; deterministic, so no seed.
    """
    if [query["qid"] for query in queries] != arrays["query_qids"].tolist():
        raise ValueError("the feature cache was built from a different query file")
    top1 = np.argmax(arrays["query_features"] @ arrays["image_features"].T, axis=1)
    labels = [f"{query['type']}_{query['lang']}" for query in queries]
    rows = []
    for label in sorted(set(labels)):
        members = [row for row, own in enumerate(labels) if own == label]
        vectors = arrays["query_features"][members]
        pairs = np.triu_indices(len(members), k=1)
        top_images = top1[members].tolist()
        rows.append(
            {
                "group": label,
                "num_queries": len(members),
                "distinct_top1_images": len(set(top_images)),
                "largest_top1_share": max(top_images.count(i) for i in set(top_images))
                / len(members),
                # A group of one query has no pair: NaN instead of a misleading number.
                "mean_query_cosine": (
                    float((vectors @ vectors.T)[pairs].mean()) if len(members) > 1 else float("nan")
                ),
            }
        )
    return rows
