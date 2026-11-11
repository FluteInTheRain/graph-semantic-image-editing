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
from mmxai.explain.presentation import Explanation, build_explanation, format_explanation
from mmxai.index.image_index import build_image_index, search_images
from mmxai.models.one_layer_sage import OneLayerSage, graph_scores
from mmxai.models.training import (
    BipartiteGraph,
    GnnConfig,
    embed_images,
    load_bipartite,
    load_chosen_alpha,
    load_texts,
    load_trained_model,
    resolve_device,
    score_candidates,
)
from mmxai.rerank.combine import final_scores

MODES = ("clip", "clip_graph")
# How queries are picked from the split (typed --text queries ignore this):
#   random  num_queries region descriptions sampled with sample_seed
#   fixed   every region description is ranked, then the first num_queries successes and
#           the first num_queries failures by text id are kept (see pick_fixed_cases)
PICKS = ("random", "fixed")
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
    explanation: Explanation  # why the top image got its graph score
    case: str | None = None  # "success" / "failure" in fixed mode
    # For the failure analysis: the same query ranked by CLIP alone, so one sees whether
    # the graph moved the correct image up or down.
    clip_rank: int | None = None
    # The image the region description was written for (None for a typed query), its
    # rank after re-ranking, and why it got its graph score (None if it is the top image,
    # whose explanation is `explanation`, or in clip mode).
    source_image_id: int | None = None
    source_rank: int | None = None
    source_explanation: Explanation | None = None


def explain_image(
    model: OneLayerSage,
    graph: BipartiteGraph,
    query: torch.Tensor,
    image_row: int,
    ranked_graph_score: float,
    alpha: float,
    top_m: int,
    min_contribution: float,
) -> Explanation:
    """Occlusion over the concept edges of one image, shown as named relations.

    Local numbering for explain/: node 0 is the image, node 1 + j its j-th concept.
    The model only reads edges into this image, so this small graph gives exactly
    the score used for ranking; occlusion checks it against ranked_graph_score.
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
        edge_index,
        0,
        NUM_LAYERS,
        both_directions=False,
        score_image=score_image,
        ranked_graph_score=ranked_graph_score,
    )
    node_names = {0: f"image {int(graph.image_ids[image_row])}"}
    relation_names = {}
    for j, concept_row in enumerate(concept_rows.tolist()):
        node_names[1 + j] = str(graph.concept_names[concept_row])
        relation_names[(1 + j, 0)] = "in"
    return build_explanation(
        contributions, top_m, node_names, relation_names, alpha, min_contribution
    )


def relevant_rows(
    text_x: np.ndarray,
    text_image_rows: np.ndarray,
    query_rows: np.ndarray,
    split_rows: np.ndarray,
    threshold: float,
    chunk_size: int = 1024,
) -> list[set[int]]:
    """For each query text, its own image plus every split image owning the same sentence.

    Same rule as the evaluation (ADR-0043): same sentence = text cosine >= threshold.
    """
    relevant = []
    # Chunks keep memory at chunk_size x split texts.
    for start in range(0, len(query_rows), chunk_size):
        chunk = query_rows[start : start + chunk_size]
        similarity = text_x[chunk] @ text_x[split_rows].T
        for row, same in zip(chunk.tolist(), similarity >= threshold, strict=True):
            relevant.append(
                {int(text_image_rows[row])} | set(text_image_rows[split_rows[same]].tolist())
            )
    return relevant


def pick_fixed_cases(
    text_ids: np.ndarray, first_ranks: list[int | None], num_cases: int, failure_after: int
) -> tuple[list[int], list[int]]:
    """Return (success positions, failure positions) chosen by a rule fixed in advance.

    Decision: success = the first correct image is ranked 1; failure = no correct image
    in the top failure_after (a miss for Recall@failure_after). Ties are broken by text id,
    ascending, and the first num_cases of each kind are kept.
      Why: the cases for the report must not be hand-picked after looking at the results
      (PDF section 6); text id order has nothing to do with how good a case looks.
      Proposed by Claude Code, 2026-11-10, at thiennp3's request; not yet decided by the team.
    """
    success = []
    failure = []
    for position in np.argsort(text_ids, kind="stable").tolist():
        rank = first_ranks[position]
        if rank == 1 and len(success) < num_cases:
            success.append(position)
        if (rank is None or rank > failure_after) and len(failure) < num_cases:
            failure.append(position)
    return success, failure


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
    pick: str = "random",
) -> list[QueryResult]:
    """Run retrieval (and re-ranking) for the queries and explain each top result."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if pick not in PICKS:
        raise ValueError(f"pick must be one of {PICKS}, got {pick!r}")
    device = resolve_device(config.device)
    graph = load_bipartite(config, train_only=False, device=device)
    gallery_rows = np.flatnonzero(graph.image_split == split_name)

    text_ids = None
    source_rows: np.ndarray | None = None
    if typed_texts:
        query_x = encode_typed_queries(config, typed_texts)
        relevant: list[set[int] | None] = [None] * len(typed_texts)
    else:
        text_x, text_image_rows, all_text_ids = load_texts(config, graph)
        split_rows = np.flatnonzero(graph.image_split[text_image_rows] == split_name)
        if pick == "random":
            rng = np.random.default_rng(sample_seed)
            query_rows = np.sort(rng.choice(split_rows, size=num_queries, replace=False))
        else:
            query_rows = split_rows  # every query: the cases are chosen after ranking
        query_x = text_x[query_rows]
        text_ids = all_text_ids[query_rows]
        source_rows = text_image_rows[query_rows]
        relevant = list(
            relevant_rows(
                text_x, text_image_rows, query_rows, split_rows, config.same_text_threshold
            )
        )

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
    ranked_graph = np.take_along_axis(graph_score, order, axis=1)

    first_ranks: list[int | None] = []
    clip_ranks: list[int | None] = []
    for q in range(len(query_x)):
        if relevant[q] is None:
            first_ranks.append(None)
            clip_ranks.append(None)
        else:
            first_ranks.append(rank_of_first(ranked_rows[q], relevant[q]))
            # FAISS returns the candidates in CLIP order, so their order is CLIP's ranking.
            clip_ranks.append(rank_of_first(candidates[q], relevant[q]))

    if pick == "fixed" and text_ids is not None:
        success, failure = pick_fixed_cases(
            text_ids, first_ranks, num_queries, failure_after=max(config.recall_ks)
        )
        chosen = success + failure
        cases: list[str | None] = ["success"] * len(success) + ["failure"] * len(failure)
    else:
        chosen = list(range(len(query_x)))
        cases = [None] * len(chosen)

    sentences = list(typed_texts)
    if text_ids is not None:
        # Looked up only for the kept queries: the regions file is large.
        sentences = sentences_of(text_ids[chosen], data_dir)

    results = []
    for q, case, sentence in zip(chosen, cases, sentences, strict=True):
        rows = ranked_rows[q]
        if model is None:
            explanation = build_explanation([], top_m, {}, {}, alpha, min_contribution)
        else:
            query = torch.from_numpy(query_x[q]).to(device)
            explanation = explain_image(
                model,
                graph,
                query,
                int(rows[0]),
                float(ranked_graph[q, 0]),
                alpha,
                top_m,
                min_contribution,
            )
        source_id = source_rank = source_explanation = None
        if source_rows is not None:
            source_row = int(source_rows[q])
            source_id = int(graph.image_ids[source_row])
            source_rank = rank_of_first(rows, {source_row})
            if model is not None and source_row != int(rows[0]):
                if source_rank is not None:
                    source_graph = float(ranked_graph[q, source_rank - 1])
                else:
                    # Not among the candidates: score it the way the candidates were scored.
                    source_graph = graph_score_of(model, graph, query, source_row)
                source_explanation = explain_image(
                    model,
                    graph,
                    query,
                    source_row,
                    source_graph,
                    alpha,
                    top_m,
                    min_contribution,
                )
        results.append(
            QueryResult(
                query=sentence,
                relevant_image_ids=None
                if relevant[q] is None
                else {int(graph.image_ids[row]) for row in relevant[q]},
                top_image_ids=[int(graph.image_ids[row]) for row in rows[:show_top]],
                top_scores=[float(value) for value in ranked_scores[q, :show_top]],
                first_relevant_rank=first_ranks[q],
                explanation=explanation,
                case=case,
                clip_rank=clip_ranks[q],
                source_image_id=source_id,
                source_rank=source_rank,
                source_explanation=source_explanation,
            )
        )
    return results


@torch.no_grad()
def graph_score_of(
    model: OneLayerSage, graph: BipartiteGraph, query: torch.Tensor, image_row: int
) -> float:
    """cos(q, h_i) through embed_images, the path score_candidates uses for ranking."""
    rows = torch.tensor([image_row], device=graph.image_x.device)
    return float(graph_scores(query, embed_images(model, graph, rows)[0]))


def rank_of_first(ranked_rows: np.ndarray, wanted: set[int]) -> int | None:
    """1-based position of the first row in `wanted`, or None if none is in the list."""
    for position, row in enumerate(ranked_rows.tolist(), start=1):
        if row in wanted:
            return position
    return None


def format_result(result: QueryResult, number: int, mode: str) -> str:
    """One query as plain text: top images, correct rank, explanations of top and source."""
    case = "" if result.case is None else f", {result.case}"
    lines = [f"=== Query {number} ({mode}{case}): {result.query}"]
    for rank, (image_id, score) in enumerate(
        zip(result.top_image_ids, result.top_scores, strict=True), start=1
    ):
        mark = ""
        if result.relevant_image_ids is not None and image_id in result.relevant_image_ids:
            mark = "  <- correct"
        lines.append(f"  {rank}. image {image_id}  score {score:.4f}{mark}")
    if result.relevant_image_ids is not None:
        lines.append(
            f"  First correct image: rank {_rank_text(result.first_relevant_rank)} "
            f"(CLIP alone: rank {_rank_text(result.clip_rank)})"
        )
    lines.append(f"  Why image {result.top_image_ids[0]} is first:")
    lines += [f"    {line}" for line in format_explanation(result.explanation).splitlines()]
    if result.source_explanation is not None:
        lines.append(
            f"  Source image {result.source_image_id} "
            f"(rank {_rank_text(result.source_rank)}), its graph score comes from:"
        )
        lines += [
            f"    {line}" for line in format_explanation(result.source_explanation).splitlines()
        ]
    return "\n".join(lines)


def cases_table(results: list[QueryResult]) -> list[dict]:
    """One row per query for a summary table (e.g. pandas.DataFrame(cases_table(...)))."""
    rows = []
    for result in results:
        supporting = ", ".join(item.source for item in result.explanation.supporting)
        opposing = ", ".join(item.source for item in result.explanation.opposing)
        rows.append(
            {
                "case": result.case,
                "query": result.query,
                "rank": _rank_text(result.first_relevant_rank),
                "rank_clip_only": _rank_text(result.clip_rank),
                "top_image": result.top_image_ids[0],
                "pushed_up_by": supporting,
                "pulled_down_by": opposing,
            }
        )
    return rows


def cases_to_markdown(results: list[QueryResult], mode: str) -> str:
    """The cases as a Markdown section for the report: query, ranks, explanations."""
    lines = [f"## Cases ({mode})", ""]
    for number, result in enumerate(results, start=1):
        case = "" if result.case is None else f" ({result.case})"
        lines.append(f"### {number}. {result.query}{case}")
        lines.append("")
        lines.append(f"- First correct image: rank {_rank_text(result.first_relevant_rank)}")
        lines.append(f"- With CLIP alone: rank {_rank_text(result.clip_rank)}")
        lines.append(f"- Top images: {', '.join(str(i) for i in result.top_image_ids)}")
        lines.append(f"- Why image {result.top_image_ids[0]} is first:")
        lines.append("")
        lines.append("```")
        lines.append(format_explanation(result.explanation))
        lines.append("```")
        if result.source_explanation is not None:
            lines.append("")
            lines.append(
                f"- Source image {result.source_image_id} "
                f"(rank {_rank_text(result.source_rank)}): its graph score comes from"
            )
            lines.append("")
            lines.append("```")
            lines.append(format_explanation(result.source_explanation))
            lines.append("```")
        lines.append("")
    return "\n".join(lines)


def _rank_text(rank: int | None) -> str:
    return "not in the top 100" if rank is None else str(rank)
