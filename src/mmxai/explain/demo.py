"""Retrieve images for a few queries and explain each top result (xAI levels 1-2).

Modes:
    clip        plain CLIP: rank by cosine only (alpha = 1). The graph plays no part,
                so the explanation says so and lists no relation.
    clip_graph  CLIP top 100 re-ranked with the trained GNN and the alpha chosen on val.
                The top result is explained by Occlusion on the model itself (ADR-0016):
                remove each concept -> image edge of that image, re-run the model, and
                report how much its graph score drops.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from mmxai.explain.occlusion import occlusion
from mmxai.explain.presentation import Explanation, build_explanation
from mmxai.index.image_index import build_image_index, search_images
from mmxai.models.one_layer_sage import OneLayerSage, graph_scores
from mmxai.models.training import (
    BipartiteGraph,
    GnnConfig,
    load_bipartite,
    load_chosen_alpha,
    load_texts,
    load_trained_model,
    resolve_device,
    score_candidates,
)
from mmxai.rerank.combine import final_scores

MODES = ("clip", "clip_graph")
# The one-layer GNN of ADR-0046: an image's score depends on its own edges only.
NUM_LAYERS = 1


@dataclass(frozen=True)
class QueryResult:
    """What one query returned and why its top image is there."""

    query: str
    relevant_image_ids: set[int] | None  # None for a typed query: no known answer
    top_image_ids: list[int]
    top_scores: list[float]
    first_relevant_rank: int | None  # among all top_k candidates, 1 = first
    explanation: Explanation


def explain_image(
    model: OneLayerSage,
    graph: BipartiteGraph,
    query: torch.Tensor,
    image_row: int,
    alpha: float,
    top_m: int,
    min_contribution: float,
) -> Explanation:
    """Occlusion over the concept edges of one image, shown as named relations.

    Local numbering for explain/: node 0 is the image, node 1 + j its j-th concept.
    The model only reads edges into this image, so this small graph gives exactly
    the score used for ranking.
    """
    concept_rows = graph.edge_index[0, graph.edge_index[1] == image_row]
    num_edges = len(concept_rows)
    edge_index = np.stack([np.arange(1, num_edges + 1), np.zeros(num_edges, dtype=np.int64)])
    concept_x = graph.concept_x[concept_rows]
    image_x = graph.image_x[image_row : image_row + 1]

    @torch.no_grad()
    def score_image(remaining: np.ndarray) -> float:
        kept = torch.from_numpy(np.flatnonzero(remaining)).to(image_x.device)
        local_edges = torch.stack([kept, torch.zeros_like(kept)])  # concept j -> image 0
        h = model(concept_x, image_x, local_edges)
        return float(graph_scores(query, h[0]))

    contributions = occlusion(
        edge_index, 0, NUM_LAYERS, both_directions=False, score_image=score_image
    )
    node_names = {0: f"image {int(graph.image_ids[image_row])}"}
    relation_names = {}
    for j, concept_row in enumerate(concept_rows.tolist()):
        node_names[1 + j] = str(graph.concept_names[concept_row])
        relation_names[(1 + j, 0)] = "in"
    return build_explanation(
        contributions, top_m, node_names, relation_names, alpha, min_contribution
    )


def pick_split_queries(
    config: GnnConfig, graph: BipartiteGraph, split_name: str, num_queries: int, sample_seed: int
) -> tuple[np.ndarray, np.ndarray, list[set[int]]]:
    """Sample region descriptions of the split as queries.

    Returns (query vectors, text ids, relevant image rows). Relevant = the source image
    plus every split image owning the same sentence (ADR-0043).
    """
    text_x, text_image_rows, text_ids = load_texts(config, graph)
    in_split = np.flatnonzero(graph.image_split[text_image_rows] == split_name)
    rng = np.random.default_rng(sample_seed)
    chosen = np.sort(rng.choice(in_split, size=num_queries, replace=False))

    relevant = []
    for text_row in chosen.tolist():
        similarity = text_x[in_split] @ text_x[text_row]
        same = in_split[similarity >= config.same_text_threshold]
        relevant.append({int(text_image_rows[text_row])} | set(text_image_rows[same].tolist()))
    return text_x[chosen], text_ids[chosen], relevant


def encode_typed_queries(config: GnnConfig, texts: list[str]) -> np.ndarray:
    """Encode typed sentences with the frozen CLIP that made the cached features."""
    # Imported here so the rest of the module runs without open_clip.
    from mmxai.features.cache import load_meta
    from mmxai.features.clip_encoder import encode_texts, load_frozen_clip

    meta = load_meta(Path(config.features_dir))
    bundle = load_frozen_clip(meta["model"], meta["pretrained"], config.device)
    return encode_texts(bundle, texts, batch_size=64)


def sentences_of(text_ids: np.ndarray, data_dir: str | None) -> list[str]:
    """The region sentences, if region_descriptions.json is there; otherwise their ids."""
    path = None if data_dir is None else Path(data_dir) / "region_descriptions.json"
    if path is None or not path.exists():
        return [
            f"(region {int(text_id)}; add region_descriptions.json to see the sentence)"
            for text_id in text_ids
        ]
    from mmxai.data.visual_genome import load_region_texts

    wanted = set(text_ids.tolist())
    by_id = {
        region.text_id: region.text
        for region in load_region_texts(Path(data_dir))
        if region.text_id in wanted
    }
    return [by_id.get(int(text_id), f"(region {int(text_id)})") for text_id in text_ids]


def run_demo(
    config: GnnConfig,
    mode: str,
    seed: int,
    split_name: str,
    num_queries: int,
    typed_texts: list[str],
    sample_seed: int,
    show_top: int,
    top_m: int,
    min_contribution: float,
    data_dir: str | None,
) -> list[QueryResult]:
    """Run retrieval (and re-ranking) for the queries and explain each top result."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    device = resolve_device(config.device)
    graph = load_bipartite(config, train_only=False, device=device)
    gallery_rows = np.flatnonzero(graph.image_split == split_name)

    if typed_texts:
        query_x = encode_typed_queries(config, typed_texts)
        sentences = list(typed_texts)
        relevant: list[set[int] | None] = [None] * len(typed_texts)
    else:
        query_x, text_ids, relevant_rows = pick_split_queries(
            config, graph, split_name, num_queries, sample_seed
        )
        sentences = sentences_of(text_ids, data_dir)
        relevant = list(relevant_rows)

    if mode == "clip":
        model = None
        alpha = 1.0
        index = build_image_index(graph.image_x.cpu().numpy()[gallery_rows], gallery_rows)
        cosine, candidates = search_images(index, query_x, config.top_k)
        graph_score = np.zeros_like(cosine)  # unused: alpha = 1
    else:
        model = load_trained_model(config, seed, graph.image_x.shape[1], device)
        alpha = load_chosen_alpha(config, seed)
        candidates, cosine, graph_score = score_candidates(
            model, graph, gallery_rows, query_x, config.top_k
        )

    scores = final_scores(cosine, graph_score, alpha)
    # Same stable sort as mmxai.rerank.combine.rerank: ties keep CLIP's order.
    order = np.argsort(-scores, axis=1, kind="stable")
    ranked_rows = np.take_along_axis(candidates, order, axis=1)
    ranked_scores = np.take_along_axis(scores, order, axis=1)

    results = []
    for q in range(len(query_x)):
        rows = ranked_rows[q]
        first_rank = None
        if relevant[q] is not None:
            for position, row in enumerate(rows.tolist(), start=1):
                if row in relevant[q]:
                    first_rank = position
                    break
        if model is None:
            explanation = build_explanation([], top_m, {}, {}, alpha, min_contribution)
        else:
            query = torch.from_numpy(query_x[q]).to(device)
            explanation = explain_image(
                model, graph, query, int(rows[0]), alpha, top_m, min_contribution
            )
        results.append(
            QueryResult(
                query=sentences[q],
                relevant_image_ids=None
                if relevant[q] is None
                else {int(graph.image_ids[row]) for row in relevant[q]},
                top_image_ids=[int(graph.image_ids[row]) for row in rows[:show_top]],
                top_scores=[float(value) for value in ranked_scores[q, :show_top]],
                first_relevant_rank=first_rank,
                explanation=explanation,
            )
        )
    return results
