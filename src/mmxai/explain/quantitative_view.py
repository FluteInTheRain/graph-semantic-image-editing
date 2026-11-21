"""Write the level-3 explanation of the demo's results as Markdown (ADR-0013, ADR-0016).

One table per image: the explanation edges with the explainer's value, then what removing
them does next to removing the same number of random edges (fidelity), and how small and
real the explanation is (sparsity, validity). A first table averages the k images.
"""

import numpy as np

from mmxai.explain.query_path import QueryEngine, concept_name
from mmxai.explain.query_quantitative import QuantifiedImage

HOW_TO_READ_QUANTITATIVE = (
    "**How to read.** The explanation is the top m concept → image edges of the main "
    "explainer chosen on val, run on the trained model. *Fid+*: how much the graph score g "
    "drops when those edges are removed; *random*: the same with as many random edges "
    "(mean of several draws). A faithful explanation has Fid+ well above random. "
    "*Rank drop*: how many places the image falls when the edges are removed. *Fid−*: how "
    "much g drops when only the explanation is kept (small = the explanation alone keeps "
    "the score). *Sparsity*: share of the image's edges left out (1 = very small). "
    "*Validity*: every explanation edge is a real edge of the scored graph that reaches "
    "the image."
)


def format_quantified_summary(items: list[QuantifiedImage]) -> str:
    """Markdown: Fid+ and rank drop of the explanations next to the random control,
    averaged over the measured images."""
    reports = [item.report for item in items if item.report is not None]
    if not reports:
        return "No image could be measured (alpha = 1, or m covers every edge of each image)."
    rows = [
        f"**Mean over {len(reports)} of {len(items)} images**",
        "",
        "| measure | explanation | random (same size) |",
        "|---|---:|---:|",
        f"| Fid+ (g lost) | {np.mean([r.fidelity_plus for r in reports]):+.4f} "
        f"| {np.mean([r.random_plus_mean for r in reports]):+.4f} |",
        f"| rank drop | {np.mean([r.rank_drop for r in reports]):.2f} "
        f"| {np.mean([r.random_rank_drop_mean for r in reports]):.2f} |",
    ]
    return "\n".join(rows)


def format_quantified(engine: QueryEngine, item: QuantifiedImage, explainer: str) -> str:
    """Markdown for one image: its explanation edges and the measures of that explanation."""
    lines = [
        f"### #{item.rank}: image {item.image_id}",
        "",
        f"graph g {item.graph_score:.4f} · explanation: top {len(item.explanation)} "
        f"of {item.num_edges} edges by {explainer}",
        "",
        f"| edge | {explainer} value |",
        "|---|---:|",
    ]
    for (concept_row, _image_row), value in zip(item.explanation, item.values, strict=True):
        lines.append(f"| {concept_name(engine, concept_row)} → contains → image | {value:+.4f} |")

    report = item.report
    if report is None:
        if engine.alpha == 1.0:
            reason = "alpha = 1: the graph plays no part in this ranking."
        else:
            reason = "The explanation keeps every edge of the image, so there is nothing to remove."
        lines += ["", f"Not measured. {reason}"]
        return "\n".join(lines)

    validity = "yes" if report.all_in_reach else "no"
    lines += [
        "",
        "| measure | explanation | random (same size) |",
        "|---|---:|---:|",
        f"| Fid+ (g lost when removed) | {report.fidelity_plus:+.4f} "
        f"| {report.random_plus_mean:+.4f} |",
        f"| rank drop | {report.rank_drop} | {report.random_rank_drop_mean:.2f} |",
        f"| Fid− (g lost when only it is kept) | {report.fidelity_minus:+.4f} | |",
        f"| sparsity | {report.sparsity:.3f} | |",
        f"| validity: real edges reaching the image | {validity} "
        f"({report.connected_share:.0%} connected) | |",
    ]
    return "\n".join(lines)
