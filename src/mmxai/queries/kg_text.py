"""Sentences for KG queries: name the concept (direct) or only describe it (indirect)."""

from mmxai.queries.generator import as_sentence
from mmxai.queries.kg_facts import Description

# "can" is followed by a verb phrase ("can carry passengers"), so it takes no article.
VERB_KINDS = ("can",)


def with_article(term: str) -> str:
    if term[0] in "aeiou":
        return f"an {term}"
    return f"a {term}"


def property_phrase(description: Description, templates: dict[str, str]) -> str:
    """ "is played on a court and uses a racket"; values of one kind share the verb."""
    values_by_kind: dict[str, list[str]] = {}
    for fact in description.properties:
        values_by_kind.setdefault(fact.kind, []).append(fact.value)
    phrases = []
    for kind, values in values_by_kind.items():
        objects = values if kind in VERB_KINDS else [with_article(value) for value in values]
        phrases.append(f"{templates[kind]} {' and '.join(objects)}")
    return " and ".join(phrases)


def direct_text(group: str, concept: str) -> str:
    if group == "activity":
        return as_sentence(f"a person playing {concept}")
    return as_sentence(with_article(concept))


def indirect_text(
    group: str,
    description: Description,
    templates: dict[str, str],
    activity_play_categories: list[str],
) -> str:
    """Indirect sentence. For activities, "playing a sport" but "playing with a toy".

    Only categories in activity_play_categories can be played directly; for any other
    category (toy, ...) the pilot val text "playing a toy that ..." read badly.
    """
    category = with_article(description.category.value)
    properties = property_phrase(description, templates)
    if group == "activity":
        if description.category.value in activity_play_categories:
            return as_sentence(f"a person playing {category} that {properties}")
        return as_sentence(f"a person playing with {category} that {properties}")
    return as_sentence(f"{category} that {properties}")
