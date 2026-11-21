"""Write G-Retriever's subgraph and each result's explanation as Markdown (ADR-0013, ADR-0051).

The explanation of an image is every concept its graph score uses: the image's concepts
that lie in the query subgraph S_q. Nothing is selected or hidden, so the table is the
whole mechanism; the drop column is exact occlusion on the trained model.
"""

from mmxai.explain.retriever_path import RetrieverAnswer, RetrieverResult

HOW_TO_READ_RETRIEVER = (
    "**How to read.** PCST builds one subgraph S_q per query on the train-counted "
    "co-occurrence graph: the prize_k concepts closest to the query (*prized*) plus the "
    "concepts needed to connect them (*bridges*). An image's graph score g only sees its "
    "own concepts that lie in S_q (*matched*). *Drop*: how much g falls when that concept "
    "is removed from S_q and the image is scored again. An image with no matched concept "
    "is scored from its own vector alone."
)


def format_subgraph(answer: RetrieverAnswer) -> str:
    """Markdown: the prized concepts, the bridges and the tree edges of S_q."""
    prized = ", ".join(answer.prized) or "none"
    bridges = ", ".join(answer.bridges) or "none"
    lines = [
        "**Query subgraph S_q**",
        "",
        f"- prized (closest to the query): {prized}",
        f"- bridges (added by PCST to connect them): {bridges}",
    ]
    if answer.tree_edges:
        edges = ", ".join(f"{a} — {b}" for a, b in answer.tree_edges)
        lines.append(f"- tree edges: {edges}")
    return "\n".join(lines)


def format_retriever_result(item: RetrieverResult, alpha: float) -> str:
    """Markdown for one image: ranks, scores and every matched concept with its drop."""
    lines = [
        f"### #{item.rank}: image {item.image_id} (CLIP rank {item.clip_rank})",
        "",
        f"clip {item.clip_score:.4f} · graph g {item.graph_score:.4f} · "
        f"final = {alpha:g} × clip + {1 - alpha:g} × g = {item.final_score:.4f}",
        "",
    ]
    if alpha == 1.0:
        lines += ["alpha = 1: the graph plays no part in this ranking.", ""]
    if not item.matched:
        lines.append("No concept of this image lies in S_q: g comes from the image vector alone.")
        return "\n".join(lines)
    lines += ["| matched concept | kind | drop of g |", "|---|---|---:|"]
    for concept in sorted(item.matched, key=lambda c: c.drop, reverse=True):
        kind = "bridge" if concept.bridge else "prized"
        lines.append(f"| {concept.name} | {kind} | {concept.drop:+.4f} |")
    return "\n".join(lines)
