"""Write one result image's explanation as a Markdown table for the level-1 demo (ADR-0013).

Which edges are shown: the explainer of the level-1 analysis (query_analysis), Occlusion,
num_edges per side: the edges whose removal lowers g most and raises g most.

The "part of g" column comes from score_path and adds up to g: the image's own part,
the bias, the shown edges and one row summing the edges not shown. The image's own part
is always shown: hiding it would suggest the whole score comes from the graph's edges,
which the 5 + 5 analysis showed is usually not the case.
"""

from mmxai.explain.query_path import ImagePath, QueryEngine, concept_name
from mmxai.explain.score_path import ConceptShare

HOW_TO_READ = (
    "**How to read a table.** g = cos(q, h_i) is the graph score. "
    "*Occlusion*: how much g drops when that one contains edge is removed "
    "(+ = the edge pushes the image up, - = it pulls it down); the edges shown are the "
    "strongest on each side. *Part of g*: the edge's term in g with every edge kept, "
    "q^T W_nb x_c / (|N(i)| ||h_i||); with the image's own part (W_self x_i) and the bias b "
    "this column adds up to g. The two numbers can differ, even in sign, because removing "
    "an edge also changes the mean over the edges and ||h_i||. Only contains edges are "
    "graph edges: the query is not a node, it enters g through the dot product with q."
)


def shown_edges(item: ImagePath, num_edges: int) -> tuple[list[ConceptShare], list[ConceptShare]]:
    """The edges an explanation shows: (pushing up, pulling down), strongest first.

    Chosen by Occlusion, num_edges per side, as in the level-1 analysis (query_analysis).
    """
    if num_edges < 1:
        raise ValueError(f"num_edges must be at least 1, got {num_edges}")
    drops = item.occlusion_drop
    by_drop = sorted(item.path.concept_shares, key=lambda edge: drops[edge.concept_row])
    positive = [edge for edge in reversed(by_drop) if drops[edge.concept_row] > 0][:num_edges]
    negative = [edge for edge in by_drop if drops[edge.concept_row] < 0][:num_edges]
    return positive, negative


def format_image_path(engine: QueryEngine, item: ImagePath, num_edges: int) -> str:
    """Markdown for one image: ranks, scores, and the num_edges strongest edges per side
    by Occlusion, each with its part of g."""
    if num_edges < 1:
        raise ValueError(f"num_edges must be at least 1, got {num_edges}")

    alpha = engine.alpha
    lines = [
        f"### #{item.rank}: image {item.image_id} (CLIP rank {item.clip_rank})",
        "",
        f"clip {item.clip_score:.4f} · graph g {item.graph_score:.4f} · "
        f"final = {alpha:g} × clip + {1 - alpha:g} × g = {item.final_score:.4f}",
    ]
    if alpha == 1.0:
        lines += ["", "alpha = 1: the graph plays no part in this ranking."]

    drops = item.occlusion_drop
    positive, negative = shown_edges(item, num_edges)
    shown = {edge.concept_row for edge in positive + negative}
    hidden = [edge for edge in item.path.concept_shares if edge.concept_row not in shown]

    lines += [
        "",
        "| term of g | Occlusion | part of g |",
        "|---|---:|---:|",
        f"| image itself (W_self x_i) | | {item.path.self_share:+.4f} |",
        f"| bias b | | {item.path.bias_share:+.4f} |",
    ]
    for edge in positive + negative:
        name = concept_name(engine, edge.concept_row)
        drop = drops[edge.concept_row]
        lines.append(f"| {name} → contains → image | {drop:+.4f} | {edge.share:+.4f} |")
    if hidden:
        hidden_sum = sum(edge.share for edge in hidden)
        lines.append(f"| {len(hidden)} other edges | | {hidden_sum:+.4f} |")
    lines.append(f"| **total = g** | | **{item.graph_score:+.4f}** |")
    return "\n".join(lines)
