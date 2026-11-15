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


def direct_text(group: str, concept: str, verb: str | None, play_predicates: list[str]) -> str:
    """Direct sentence. verb is the activity predicate, None for the object group."""
    if group == "activity" and verb in play_predicates:
        # "playing tennis": a played activity takes no article.
        return as_sentence(f"a person {verb} {concept}")
    if group == "activity":
        return as_sentence(f"a person {verb} {with_article(concept)}")
    return as_sentence(with_article(concept))


def indirect_text(
    group: str,
    description: Description,
    templates: dict[str, str],
    activity_play_categories: list[str],
    verb: str | None,
    play_predicates: list[str],
) -> str:
    """Indirect sentence. For played activities, "playing a sport" but "playing with a toy".

    Only categories in activity_play_categories can be played directly; for any other
    category (toy, ...) the pilot val text "playing a toy that ..." read badly. Other
    activity verbs ("riding an animal that ...") always take the category with an article.
    """
    category = with_article(description.category.value)
    properties = property_phrase(description, templates)
    if group == "activity" and verb in play_predicates:
        if description.category.value in activity_play_categories:
            return as_sentence(f"a person {verb} {category} that {properties}")
        return as_sentence(f"a person {verb} with {category} that {properties}")
    if group == "activity":
        return as_sentence(f"a person {verb} {category} that {properties}")
    return as_sentence(f"{category} that {properties}")
