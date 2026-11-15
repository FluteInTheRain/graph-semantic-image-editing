"""Write one result image's paths as a Markdown table for the demo (ADR-0013).

Every row is a part of the graph score g from score_path, so the column adds up to g:
the image's own part, the bias, the strongest positive and negative concept edges,
and one row summing the edges not shown. Hiding the image's own part would suggest
that the whole score comes from the graph's edges, which the 5 + 5 analysis showed
is usually not the case.
"""

from mmxai.explain.query_path import ImagePath, QueryEngine, concept_name

HOW_TO_READ = (
    "**How to read a table.** g = cos(q, h_i) is the graph score. Its parts add up to g: "
    "the image itself (W_self x_i), the bias b, and one part per contains edge "
    "(q^T W_nb x_c / (|N(i)| ||h_i||)). *Occlusion* is how much g really drops when that "
    "one edge is removed; it differs from the part because removing an edge also changes "
    "the mean and ||h_i||."
)


def format_image_path(engine: QueryEngine, item: ImagePath, num_edges: int) -> str:
    """Markdown for one image: ranks, scores and the parts of g, num_edges per side."""
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

    shares = item.path.concept_shares  # largest first
    positive = [edge for edge in shares if edge.share > 0][:num_edges]
    negative = [edge for edge in reversed(shares) if edge.share < 0][:num_edges]
    shown = {edge.concept_row for edge in positive + negative}
    hidden = [edge for edge in shares if edge.concept_row not in shown]

    lines += [
        "",
        "| path | part of g | Occlusion |",
        "|---|---:|---:|",
        f"| query → image itself | {item.path.self_share:+.4f} | |",
        f"| query → bias | {item.path.bias_share:+.4f} | |",
    ]
    for edge in positive + negative:
        name = concept_name(engine, edge.concept_row)
        drop = item.occlusion_drop[edge.concept_row]
        lines.append(f"| query → {name} → contains → image | {edge.share:+.4f} | {drop:+.4f} |")
    if hidden:
        hidden_sum = sum(edge.share for edge in hidden)
        lines.append(f"| {len(hidden)} other edges | {hidden_sum:+.4f} | |")
    lines.append(f"| **total = g** | **{item.graph_score:+.4f}** | |")
    return "\n".join(lines)
