"""Facts about a concept from ConceptNet, and a description that fits only that concept.

A fact is something true about a concept C read from one edge: "C is a sport",
"C uses a racket" (racket -UsedFor-> C). An indirect query names one category and a
few properties of C but never C itself (ADR-0050). To be a fair query it must point
to C alone: no other concept of the same group may have the same category and
properties, otherwise images of that other concept would also be correct answers.
"""

from dataclasses import dataclass

from mmxai.data.conceptnet import ConceptEdge
from mmxai.queries.generator import PLAIN_WORDS


@dataclass(frozen=True)
class Fact:
    """kind is "isa", "uses", "location", "has" or "can"; value is the other end."""

    kind: str
    value: str
    edge: ConceptEdge


@dataclass(frozen=True)
class Description:
    concept: str
    category: Fact
    properties: tuple[Fact, ...]


def facts_from_edge(edge: ConceptEdge) -> list[tuple[str, Fact]]:
    """(concept, fact) pairs one edge gives; the direction table in the plan/ADR fixes it."""
    if edge.rel == "IsA":
        return [(edge.start, Fact("isa", edge.end, edge))]
    if edge.rel == "UsedFor":
        return [(edge.end, Fact("uses", edge.start, edge))]
    if edge.rel == "AtLocation":
        return [(edge.start, Fact("location", edge.end, edge))]
    if edge.rel == "HasA":
        return [(edge.start, Fact("has", edge.end, edge))]
    if edge.rel == "PartOf":
        return [(edge.end, Fact("has", edge.start, edge))]
    if edge.rel == "CapableOf":
        return [(edge.start, Fact("can", edge.end, edge))]
    return []


def build_fact_index(edges: list[ConceptEdge]) -> dict[str, list[Fact]]:
    """Facts per concept. The same (kind, value) from several edges keeps the heaviest edge."""
    best: dict[tuple[str, str, str], Fact] = {}
    for edge in edges:
        for concept, fact in facts_from_edge(edge):
            key = (concept, fact.kind, fact.value)
            if key not in best or fact.edge.weight > best[key].edge.weight:
                best[key] = fact
    index: dict[str, list[Fact]] = {}
    for (concept, _, _), fact in sorted(best.items()):
        index.setdefault(concept, []).append(fact)
    return index


def has_fact(index: dict[str, list[Fact]], concept: str, kind: str, value: str) -> bool:
    return any(fact.kind == kind and fact.value == value for fact in index.get(concept, []))


def concepts_with(index: dict[str, list[Fact]], fact: Fact, concepts: list[str]) -> list[str]:
    """The concepts (from the given list) that share this fact."""
    sharing = []
    for concept in concepts:
        if has_fact(index, concept, fact.kind, fact.value):
            sharing.append(concept)
    return sharing


def leaks_name(value: str, concept: str) -> bool:
    """True if the value repeats a word of the concept ("tennis ball" for tennis)."""
    return bool(set(value.split()) & set(concept.split()))


def property_order(fact: Fact, index: dict[str, list[Fact]], look_alikes: list[str]) -> tuple:
    """Sort key: fewest remaining look-alikes, then heavier edge, then alphabet."""
    return (len(concepts_with(index, fact, look_alikes)), -fact.edge.weight, fact.value)


def choose_description(
    concept: str,
    pool: list[str],
    index: dict[str, list[Fact]],
    category_words: list[str],
    num_properties: int,
) -> Description | None:
    """One category + num_properties properties that together fit `concept` only.

    pool holds every concept of the same group (not only the targets), so a property
    shared with any of them is not unique. Greedy: always add the property that the
    fewest remaining look-alikes share; ties go to the heavier edge, then the
    alphabet, so the result is deterministic. None if no unique description exists.
    """
    facts = index.get(concept, [])
    others = [other for other in pool if other != concept]

    categories = []
    for fact in facts:
        if fact.kind == "isa" and fact.value in category_words:
            categories.append(fact)
    if not categories:
        return None

    def category_order(fact: Fact) -> tuple:
        return (len(concepts_with(index, fact, others)), fact.value)

    category = min(categories, key=category_order)
    look_alikes = concepts_with(index, category, others)

    candidates = []
    for fact in facts:
        if fact.kind == "isa" or not PLAIN_WORDS.fullmatch(fact.value):
            continue
        if leaks_name(fact.value, concept):
            continue
        candidates.append(fact)

    chosen: list[Fact] = []
    while len(chosen) < num_properties and candidates:
        best = min(candidates, key=lambda fact: property_order(fact, index, look_alikes))
        chosen.append(best)
        candidates.remove(best)
        look_alikes = concepts_with(index, best, look_alikes)

    if len(chosen) < num_properties or look_alikes:
        return None
    return Description(concept, category, tuple(chosen))
