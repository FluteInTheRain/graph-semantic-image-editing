"""Search one free-text query and explain each top-k image as paths (PDF sections 5, 7, 8).

Flow of search_and_explain, with the same functions the evaluation uses:
    1. CLIP top-k over the split's images (search_images, as retrieve_candidates does).
    2. Graph score cos(q, h_i) of every candidate (candidate_graph_scores) and
       final = alpha * clip + (1 - alpha) * graph with the alpha saved by training.
    3. Re-rank by final score (ranked_columns, the evaluation's stable sort).
    4. For each of the first k images: the score split into
           query -> image itself (W_self x_i), bias, query -> concept c -[contains]-> image
       (score_path.decompose_graph_score), next to Occlusion's drop for each edge
       (the score lost when that one edge is really removed).

The engine holds one split only. Decision (ADR-0009): the demo uses val; test is
scored once, in scripts/evaluate_graph_rerank.py.
"""

import json
from dataclasses import dataclass

import numpy as np
import torch

from mmxai.eval.graph_rerank import (
    MODEL_FILE,
    SELECTION_FILE,
    GraphRerankConfig,
    SplitGraph,
    build_split_graph,
    image_vectors,
    load_inputs,
    make_model,
    seed_model_dir,
    split_candidates,
)
from mmxai.eval.query_ranks import fused_ranks, ranked_columns
from mmxai.explain.occlusion import occlusion
from mmxai.explain.query_analysis import check_reproduces_training
from mmxai.explain.ranking_model import SCORE_TOLERANCE
from mmxai.explain.sage_scoring import image_scorer
from mmxai.explain.score_path import ScorePath, decompose_graph_score
from mmxai.features.cache import load_meta
from mmxai.features.clip_encoder import ClipBundle, encode_texts, load_frozen_clip
from mmxai.index.image_index import ImageIndex, build_image_index, search_images
from mmxai.models.sage import ConceptToImageSage
from mmxai.rerank.fusion import Candidates, candidate_graph_scores, fuse_scores

# Decision (ADR-0009): a typed query searches val only; test is scored once elsewhere.
DEMO_SPLIT = "val"


@dataclass(frozen=True)
class QueryEngine:
    """Everything needed to search and explain one split, loaded once.

    image_h[r] is the graph vector of split_graph row r (image_vectors on the full graph).
    image_index holds the same split's CLIP image vectors.
    """

    model: ConceptToImageSage
    split_graph: SplitGraph
    image_h: np.ndarray
    image_index: ImageIndex
    alpha: float  # chosen on val by training (selection.json), never by the demo
    top_k: int  # how many CLIP candidates are re-ranked (config.top_k)
    concept_keys: np.ndarray  # synset of each concept; concept node row r is key r - num_images


@dataclass(frozen=True)
class ImagePath:
    """One result image: its scores, ranks and the paths that make its graph score."""

    rank: int  # under CLIP + Graph, 1 = best
    clip_rank: int
    image_id: int
    clip_score: float
    graph_score: float
    final_score: float
    path: ScorePath
    occlusion_drop: dict[int, float]  # concept row -> graph score lost without that edge


def search_and_explain(engine: QueryEngine, query: np.ndarray, k: int) -> list[ImagePath]:
    """Return the first k images under CLIP + Graph for this query, each with its paths.

    query: the CLIP text vector of the query (length 1, as stored features are).
    """
    if not 1 <= k <= engine.top_k:
        raise ValueError(f"k must be between 1 and top_k={engine.top_k}, got {k}")

    # 1. CLIP top-k, in CLIP's order.
    clip_scores, candidate_ids = search_images(engine.image_index, query[None, :], engine.top_k)
    # relevant is empty: a typed query has no ground truth, nothing is measured here.
    candidates = Candidates(query[None, :], candidate_ids, clip_scores, [set()])

    # 2. The ranking's scores, from the evaluation's own functions.
    graph_scores = candidate_graph_scores(candidates, engine.image_h, engine.split_graph.image_ids)[
        0
    ]
    final_scores = fuse_scores(clip_scores[0], graph_scores, engine.alpha)

    row_of_id = {}
    for row, image_id in enumerate(engine.split_graph.image_ids.tolist()):
        row_of_id[image_id] = row

    # 3. and 4.
    results = []
    order = ranked_columns(final_scores)
    for rank, column in enumerate(order[:k].tolist(), start=1):
        image_row = row_of_id[int(candidate_ids[0, column])]
        path = decompose_graph_score(engine.model, engine.split_graph, image_row, query)
        # Guard: the explained score must be the one that ranked the image.
        if abs(path.graph_score - float(graph_scores[column])) > SCORE_TOLERANCE:
            raise ValueError(
                f"image row {image_row}: path gives graph score {path.graph_score}, "
                f"the ranking used {graph_scores[column]} (image_h from another model?)"
            )
        results.append(
            ImagePath(
                rank=rank,
                clip_rank=column + 1,  # search_images returns CLIP's order
                image_id=int(candidate_ids[0, column]),
                clip_score=float(clip_scores[0, column]),
                graph_score=float(graph_scores[column]),
                final_score=float(final_scores[column]),
                path=path,
                occlusion_drop=occlusion_drops(engine, image_row, query),
            )
        )
    return results


def occlusion_drops(engine: QueryEngine, image_row: int, query: np.ndarray) -> dict[int, float]:
    """Occlusion on the image's own edges: concept row -> drop of the graph score.

    Uses the per-image scorer, which gives the same score as the full graph
    (tests/test_explain_sage_scoring.py) but runs in milliseconds.
    """
    scorer = image_scorer(engine.model, engine.split_graph, image_row, query)
    contributions = occlusion(
        scorer.edge_index,
        image_row,
        engine.model.num_layers,
        both_directions=False,
        score_image=scorer.score_image,
    )
    drops = {}
    for edge in contributions:
        drops[edge.source] = edge.contribution
    return drops


def load_engine(config: GraphRerankConfig, seed: int, log=print) -> QueryEngine:
    """Load the trained model and the val split once, and check they reproduce training.

    Stops (check_reproduces_training) unless every val text, ranked with these weights,
    this graph and the saved alpha, gives exactly the val metric training saved.
    """
    if config.concept_edges_dir is not None:
        raise ValueError("only the one-layer model (concept_edges_dir: null) can be explained")

    model_dir = seed_model_dir(config, seed)
    with open(model_dir / SELECTION_FILE, encoding="utf-8") as file:
        selection = json.load(file)
    alpha = float(selection["alpha"])

    log(f"loading features, graph and model ({DEMO_SPLIT})")
    inputs = load_inputs(config)
    split_graph = build_split_graph(inputs, DEMO_SPLIT)
    model = make_model(config, split_graph.x.shape[1])
    model.load_state_dict(torch.load(model_dir / MODEL_FILE, weights_only=True))
    model.eval()
    image_h = image_vectors(model, split_graph)

    log("checking the model reproduces the val metric saved by training")
    candidates = split_candidates(config, inputs, DEMO_SPLIT)
    graph_scores = candidate_graph_scores(candidates, image_h, split_graph.image_ids)
    check_reproduces_training(
        config, selection, alpha, fused_ranks(candidates, graph_scores, alpha)
    )

    # The same gallery as retrieve_candidates: the split's images only.
    is_split_image = np.isin(inputs.image_ids, inputs.split[DEMO_SPLIT])
    image_index = build_image_index(
        inputs.image_embeddings[is_split_image], inputs.image_ids[is_split_image]
    )
    return QueryEngine(
        model=model,
        split_graph=split_graph,
        image_h=image_h,
        image_index=image_index,
        alpha=alpha,
        top_k=config.top_k,
        concept_keys=inputs.graph.concept_keys,
    )


def load_text_encoder(config: GraphRerankConfig, device: str = "cpu") -> ClipBundle:
    """The frozen CLIP that made the cached features (model name read from their meta.json)."""
    meta = load_meta(config.features_dir)
    return load_frozen_clip(meta["model"], meta["pretrained"], device)


def encode_query(bundle: ClipBundle, text: str) -> np.ndarray:
    """CLIP text vector of one query, length 1 like the cached text features."""
    return encode_texts(bundle, [text], batch_size=1)[0]


def concept_name(engine: QueryEngine, concept_row: int) -> str:
    """Synset of a concept node row, e.g. "dog.n.01" (concepts come after the images)."""
    return str(engine.concept_keys[concept_row - len(engine.split_graph.image_ids)])
