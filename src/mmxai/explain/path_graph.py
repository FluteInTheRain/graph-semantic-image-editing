"""Draw the paths behind a query's top-k images as a graph, for the level-1 demo (ADR-0013).

Three columns: the query | concept nodes | the top-k image nodes.
    contains edges (concept -> image): the edges Occlusion picked for that image
        (path_view.shown_edges), green if removing them lowers g, red if it raises g,
        thicker for a larger effect. These are real edges of the val graph.
    query -> image itself: the image's own part of g (W_self x_i), drawn because it is
        usually most of the score; hiding it would credit the edges with everything.
    query -> concept (dotted): not a graph edge; the query enters g through the dot
        product with q (score_path), so it only marks where the path starts.
No layout or path search chooses anything: every highlighted edge is one Occlusion chose.
"""

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")  # draw to a figure object, no window: the app shows it in the browser

from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from mmxai.explain.path_view import shown_edges  # noqa: E402
from mmxai.explain.query_path import ImagePath, QueryEngine, concept_name  # noqa: E402

QUERY_NODE = "query"
SELF = "self"  # query -> image itself
CONTAINS = "contains"  # concept -> image, a real graph edge
QUERY_TERM = "query_term"  # query -> concept, not a graph edge

PUSH_COLOR = "#2a9d4b"
PULL_COLOR = "#d1495b"
SELF_COLOR = "#7a7a7a"


@dataclass(frozen=True)
class PathEdge:
    source: str  # node id
    target: str
    kind: str  # SELF, CONTAINS or QUERY_TERM
    weight: float  # SELF: part of g; CONTAINS: Occlusion drop; QUERY_TERM: 0


@dataclass(frozen=True)
class PathGraph:
    """Nodes in drawing order (top to bottom) with their labels, and the edges."""

    concepts: list[tuple[str, str]]  # (node id, label)
    images: list[tuple[str, str]]
    edges: list[PathEdge]


def image_node(item: ImagePath) -> str:
    return f"image:{item.image_id}"


def concept_node(concept_row: int) -> str:
    return f"concept:{concept_row}"


def build_path_graph(engine: QueryEngine, results: list[ImagePath], num_edges: int) -> PathGraph:
    """The nodes and edges to draw for these results (module docstring)."""
    concepts = []
    seen_concepts = set()
    images = []
    edges = []
    for item in results:
        image = image_node(item)
        images.append((image, f"#{item.rank} · image {item.image_id}\ng = {item.graph_score:.3f}"))
        edges.append(PathEdge(QUERY_NODE, image, SELF, item.path.self_share))
        positive, negative = shown_edges(item, num_edges)
        for edge in positive + negative:
            concept = concept_node(edge.concept_row)
            if concept not in seen_concepts:
                seen_concepts.add(concept)
                concepts.append((concept, concept_name(engine, edge.concept_row)))
                edges.append(PathEdge(QUERY_NODE, concept, QUERY_TERM, 0.0))
            drop = item.occlusion_drop[edge.concept_row]
            edges.append(PathEdge(concept, image, CONTAINS, drop))
    return PathGraph(concepts, images, edges)


def column_positions(nodes: list[tuple[str, str]], x: float) -> dict[str, tuple[float, float]]:
    """Spread nodes evenly from top (y = 1) to bottom (y = 0) at this x."""
    positions = {}
    for index, (node, _) in enumerate(nodes):
        y = 0.5 if len(nodes) == 1 else 1 - index / (len(nodes) - 1)
        positions[node] = (x, y)
    return positions


def draw_path_graph(graph: PathGraph, query_text: str) -> Figure:
    """Matplotlib figure of the graph; the caller closes it (plt.close)."""
    positions = {QUERY_NODE: (0.0, 0.5)}
    positions.update(column_positions(graph.concepts, 1.0))
    positions.update(column_positions(graph.images, 2.0))

    contains = [edge for edge in graph.edges if edge.kind == CONTAINS]
    largest_drop = max([abs(edge.weight) for edge in contains] + [1e-12])
    selves = [edge for edge in graph.edges if edge.kind == SELF]

    height = max(4.0, 0.45 * max(len(graph.concepts), 2 * len(graph.images)))
    figure, axes = plt.subplots(figsize=(13, height))
    for edge in graph.edges:
        start = positions[edge.source]
        end = positions[edge.target]
        if edge.kind == QUERY_TERM:
            style = {"color": "#b0b0b0", "lw": 0.8, "ls": ":"}
            curve = 0.0
        elif edge.kind == SELF:
            # Curved, so it passes around the concept column instead of through it. A fixed
            # width: on the contains scale it would be ~100x thicker; its number is written on it.
            style = {"color": SELF_COLOR, "lw": 2.5, "alpha": 0.6}
            curve = -0.2
        else:
            color = PUSH_COLOR if edge.weight > 0 else PULL_COLOR
            style = {"color": color, "lw": 1 + 5 * abs(edge.weight) / largest_drop}
            curve = 0.0
        axes.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "-|>", "connectionstyle": f"arc3,rad={curve}", **style},
        )
        if edge.kind == CONTAINS:
            # Near the concept end: at the image end the labels of its edges would overlap.
            label_at = (start[0] + 0.3 * (end[0] - start[0]), start[1] + 0.3 * (end[1] - start[1]))
            axes.text(
                *label_at,
                f"{edge.weight:+.4f}",
                fontsize=7,
                color=style["color"],
                ha="center",
                va="bottom",
            )

    for node, label in graph.concepts:
        axes.text(
            *positions[node],
            label,
            ha="center",
            va="center",
            fontsize=8,
            bbox={"boxstyle": "round", "fc": "#fff6d5", "ec": "#c9a227"},
        )
    for node, label in graph.images:
        self_share = next(edge.weight for edge in selves if edge.target == node)
        axes.text(
            *positions[node],
            f"{label}\nitself {self_share:+.3f}",
            ha="center",
            va="center",
            fontsize=8,
            bbox={"boxstyle": "round", "fc": "#e3eefc", "ec": "#3d6fb6"},
        )
    axes.text(
        *positions[QUERY_NODE],
        f'query\n"{query_text}"',
        ha="center",
        va="center",
        fontsize=9,
        bbox={"boxstyle": "round", "fc": "#eeeeee", "ec": "#555555"},
    )

    axes.plot(
        [], [], color=PUSH_COLOR, lw=3, label="contains edge, removing it lowers g (Occlusion +)"
    )
    axes.plot(
        [], [], color=PULL_COLOR, lw=3, label="contains edge, removing it raises g (Occlusion −)"
    )
    axes.plot(
        [],
        [],
        color=SELF_COLOR,
        lw=3,
        label="query → image itself (W_self x_i): its part of g is in the image box",
    )
    axes.plot(
        [],
        [],
        color="#b0b0b0",
        lw=1,
        ls=":",
        label="query → concept (dot product, not a graph edge)",
    )
    axes.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=8, frameon=False)
    axes.set_xlim(-0.35, 2.35)
    axes.set_ylim(-0.12, 1.3)
    axes.axis("off")
    figure.tight_layout()
    return figure
