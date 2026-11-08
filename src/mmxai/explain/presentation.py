"""Turn edge contributions into relations a person can read."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from mmxai.explain.occlusion import EdgeContribution


@dataclass(frozen=True)
class Relation:
    """One relation in an explanation, with names instead of ids."""

    source: str
    relation: str
    target: str
    contribution: float


@dataclass(frozen=True)
class Explanation:
    """Relations that pushed the image up, and relations that pulled it down."""

    graph_used: bool
    supporting: list[Relation]
    opposing: list[Relation]


def build_explanation(
    contributions: Sequence[EdgeContribution],
    top_m: int,
    node_names: Mapping[int, str],
    relation_names: Mapping[tuple[int, int], str],
    alpha: float,
    min_contribution: float,
) -> Explanation:
    """Keep the top_m strongest relations on each side and name them.

    Contributions are shown as their effect on the final score, which weighs the graph
    by (1 - alpha). Relations whose effect is not above min_contribution are left out.
    With alpha = 1 the graph plays no part, so no relation is shown.
    Raises ValueError if a node or relation has no name.
    """
    if top_m < 1:
        raise ValueError(f"top_m must be at least 1, got {top_m}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if min_contribution < 0:
        raise ValueError(f"min_contribution must not be negative, got {min_contribution}")
    if alpha == 1.0:
        return Explanation(graph_used=False, supporting=[], opposing=[])

    effects = [
        EdgeContribution(edge.source, edge.target, (1 - alpha) * edge.contribution)
        for edge in contributions
    ]
    positive = [edge for edge in effects if edge.contribution > min_contribution]
    negative = [edge for edge in effects if edge.contribution < -min_contribution]
    positive.sort(key=lambda edge: edge.contribution, reverse=True)
    negative.sort(key=lambda edge: edge.contribution)

    return Explanation(
        graph_used=True,
        supporting=[_name(edge, node_names, relation_names) for edge in positive[:top_m]],
        opposing=[_name(edge, node_names, relation_names) for edge in negative[:top_m]],
    )


def _name(
    edge: EdgeContribution,
    node_names: Mapping[int, str],
    relation_names: Mapping[tuple[int, int], str],
) -> Relation:
    for node in (edge.source, edge.target):
        if node not in node_names:
            raise ValueError(f"node {node} has no name")
    if (edge.source, edge.target) not in relation_names:
        raise ValueError(f"edge ({edge.source}, {edge.target}) has no relation name")

    return Relation(
        source=node_names[edge.source],
        relation=relation_names[(edge.source, edge.target)],
        target=node_names[edge.target],
        contribution=edge.contribution,
    )


def format_explanation(explanation: Explanation) -> str:
    """Write the explanation as plain text, one relation per line."""
    if not explanation.graph_used:
        return "The graph was not used (alpha = 1), so no relation explains this result."
    lines = ["Pushed the image up:"]
    lines += _format_side(explanation.supporting)
    lines.append("Pulled the image down:")
    lines += _format_side(explanation.opposing)
    return "\n".join(lines)


def _format_side(relations: list[Relation]) -> list[str]:
    if not relations:
        return ["  (none)"]
    return [
        f"  {rank}. {item.source} -[{item.relation}]-> {item.target} ({item.contribution:+.2f})"
        for rank, item in enumerate(relations, start=1)
    ]
