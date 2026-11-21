"""Search one free-text query with G-Retriever stage 2 and explain each result (ADR-0051).

Flow of search_retriever, with the functions the stage-2 evaluation uses:
    1. CLIP top_k over the split's images (clip_candidates, as PcstSageRanker does).
    2. S_q: the prize_k concepts closest to the query get prizes and PCST keeps a
       connected subgraph of the PMI concept graph (split_subgraphs, as for training).
    3. Graph score of every candidate on S_q (split_scoring + graph_scores) and
       final = alpha * clip + (1 - alpha) * graph with the alpha saved by training.
    4. For each of the first k images, the explanation is the concepts the score really
       uses: the image's concepts that lie in S_q. Each is removed from S_q and the image
       re-scored (explain_candidate, exact occlusion); bridge = kept by PCST only to
       connect prized concepts.

No val-metric check at load time, unlike the CLIP + Graph demo: it would need S_q of every
val text (PCST takes about an hour). Instead each explained score must equal the score
that ranked the image. The demo uses val only (ADR-0009).
"""

import json
from dataclasses import dataclass

import numpy as np
import torch

from mmxai.eval.graph_rerank import (
    MODEL_FILE,
    SELECTION_FILE,
    Inputs,
    SplitGraph,
    build_split_graph,
)
from mmxai.eval.pcst_rerank import (
    concept_pairs,
    load_checked_inputs,
    split_subgraphs,
    top_concepts_per_query,
)
from mmxai.eval.pcst_sage_rerank import (
    PcstSageConfig,
    graph_scores,
    seed_model_dir,
    split_scoring,
)
from mmxai.eval.query_benchmark_models import clip_candidates
from mmxai.eval.query_ranks import ranked_columns
from mmxai.explain.ranking_model import SCORE_TOLERANCE
from mmxai.graph.pcst import QuerySubgraph, edge_costs
from mmxai.graph.pcst_cache import pack_subgraphs
from mmxai.index.vector_store import select_split_images
from mmxai.models.subgraph_sage import SubgraphSage, explain_candidate
from mmxai.rerank.fusion import fuse_scores

# Decision (ADR-0009): a typed query searches val only; test is scored once elsewhere.
DEMO_SPLIT = "val"


@dataclass(frozen=True)
class RetrieverEngine:
    """Everything G-Retriever stage 2 needs to search and explain one split, loaded once."""

    config: PcstSageConfig
    inputs: Inputs
    pairs: np.ndarray  # undirected PMI concept pairs PCST runs on
    split_graph: SplitGraph
    model: SubgraphSage
    alpha: float  # chosen on val by training (selection.json), never by the demo
    image_embeddings: np.ndarray  # CLIP vectors of the split's images
    image_ids: np.ndarray


@dataclass(frozen=True)
class MatchedConcept:
    """A concept the image contains and S_q keeps: it enters the image's score."""

    name: str
    bridge: bool  # True: in S_q only to connect prized concepts, not close to the query
    drop: float  # graph score lost when this concept is removed from S_q


@dataclass(frozen=True)
class RetrieverResult:
    """One result image: its scores, ranks and the matched concepts behind its graph score."""

    rank: int  # under G-Retriever, 1 = best
    clip_rank: int
    image_id: int
    clip_score: float
    graph_score: float
    final_score: float
    matched: list[MatchedConcept]


@dataclass(frozen=True)
class RetrieverAnswer:
    """The query's subgraph S_q and the first k images."""

    prized: list[str]  # the prize_k concepts closest to the query
    bridges: list[str]  # concepts PCST added to connect the prized ones
    tree_edges: list[tuple[str, str]]  # PMI edges of the PCST tree
    results: list[RetrieverResult]


def query_subgraph_for(
    engine: RetrieverEngine, query: np.ndarray
) -> tuple[QuerySubgraph, set[int]]:
    """S_q of one query exactly as training built it, and its prized concept rows."""
    config = engine.config
    concepts = engine.inputs.concept_features
    top = top_concepts_per_query(query[None, :], concepts, config.prize_k)
    subgraph = split_subgraphs(
        top,
        len(concepts),
        engine.pairs,
        config.subgraph_mode,
        config.prize_k,
        edge_costs(engine.pairs, config.edge_cost),
        log=lambda message: None,
    )[0]
    return subgraph, set(top[0, : config.prize_k].tolist())


def search_retriever(engine: RetrieverEngine, query: np.ndarray, k: int) -> RetrieverAnswer:
    """Return S_q and the first k images under G-Retriever stage 2, each explained."""
    config = engine.config
    if not 1 <= k <= config.top_k:
        raise ValueError(f"k must be between 1 and top_k={config.top_k}, got {k}")
    keys = engine.inputs.graph.concept_keys

    # 1. and 2.
    candidates = clip_candidates(
        engine.image_embeddings, engine.image_ids, query[None, :], config.top_k
    )
    subgraph, prized_rows = query_subgraph_for(engine, query)
    offsets, values = pack_subgraphs([subgraph.concepts])

    # 3. The ranking's scores, from the evaluation's own functions.
    scoring = split_scoring(
        engine.inputs, DEMO_SPLIT, engine.split_graph, candidates, offsets, values
    )
    graph = graph_scores(config, engine.model, scoring)[0]
    final = fuse_scores(candidates.clip_scores[0], graph, engine.alpha)

    # 4.
    results = []
    query_vector = torch.from_numpy(query)
    for rank, column in enumerate(ranked_columns(final)[:k].tolist(), start=1):
        row = int(scoring.candidate_rows[0, column])
        explained = explain_candidate(
            engine.model,
            query_vector,
            scoring.query_subgraphs[0],
            prized_rows,
            scoring.image_x[row],
            scoring.image_concepts[row],
            scoring.concept_x,
            keys,
        )
        # Guard: the explained score must be the one that ranked the image.
        if abs(explained["score"] - float(graph[column])) > SCORE_TOLERANCE:
            raise ValueError(
                f"image row {row}: explanation gives {explained['score']}, "
                f"the ranking used {graph[column]}"
            )
        matched = []
        for concept in explained["matched"]:
            matched.append(
                MatchedConcept(
                    name=concept["concept"],
                    bridge=concept["bridge"],
                    drop=explained["score"] - concept["score_without"],
                )
            )
        results.append(
            RetrieverResult(
                rank=rank,
                clip_rank=column + 1,  # clip_candidates returns CLIP's order
                image_id=int(candidates.candidate_ids[0, column]),
                clip_score=float(candidates.clip_scores[0, column]),
                graph_score=float(graph[column]),
                final_score=float(final[column]),
                matched=matched,
            )
        )

    in_subgraph = subgraph.concepts.tolist()
    return RetrieverAnswer(
        prized=[str(keys[row]) for row in in_subgraph if row in prized_rows],
        bridges=[str(keys[row]) for row in in_subgraph if row not in prized_rows],
        tree_edges=[(str(keys[a]), str(keys[b])) for a, b in subgraph.tree_pairs.tolist()],
        results=results,
    )


def load_retriever_engine(config: PcstSageConfig, seed: int) -> RetrieverEngine:
    """Load the stage-2 weights and alpha training saved for this seed, and the val split.

    Needs the PMI concept edges (config.concept_edges_dir) and the trained model
    (scripts/train_pcst_sage.py); a missing file raises FileNotFoundError.
    """
    model_dir = seed_model_dir(config, seed)
    with open(model_dir / SELECTION_FILE, encoding="utf-8") as file:
        selection = json.load(file)

    inputs = load_checked_inputs(config)
    split_graph = build_split_graph(inputs, DEMO_SPLIT)
    model = SubgraphSage(split_graph.x.shape[1])
    model.load_state_dict(torch.load(model_dir / MODEL_FILE, weights_only=True))
    model.eval()
    image_embeddings, image_ids = select_split_images(
        inputs.image_embeddings, inputs.image_ids, inputs.split[DEMO_SPLIT]
    )
    return RetrieverEngine(
        config=config,
        inputs=inputs,
        pairs=concept_pairs(inputs, config, seed),
        split_graph=split_graph,
        model=model,
        alpha=float(selection["alpha"]),
        image_embeddings=image_embeddings,
        image_ids=image_ids,
    )
