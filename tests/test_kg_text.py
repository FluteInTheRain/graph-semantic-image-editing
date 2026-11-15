"""Direct/indirect sentences for KG queries (ADR-0050)."""

from conceptnet_fixtures import EDGES
from mmxai.queries.kg_facts import build_fact_index, choose_description
from mmxai.queries.kg_text import direct_text, indirect_text, with_article

ACTIVITY_TEMPLATES = {"uses": "uses", "location": "is played on", "has": "has", "can": "can"}
OBJECT_TEMPLATES = {"uses": "is used with", "location": "is found at", "has": "has", "can": "can"}


def test_with_article():
    assert with_article("racket") == "a racket"
    assert with_article("apple") == "an apple"


def test_activity_sentences():
    index = build_fact_index(EDGES)
    description = choose_description("tennis", ["tennis", "badminton"], index, ["sport"], 2)

    assert direct_text("activity", "tennis") == "A person playing tennis."
    assert indirect_text("activity", description, ACTIVITY_TEMPLATES) == (
        "A person playing a sport that is played on a court and uses a racket."
    )


def test_object_sentences_keep_verb_values_without_article():
    index = build_fact_index(EDGES)
    description = choose_description("train", ["train", "bus", "car"], index, ["vehicle"], 2)

    assert direct_text("object", "train") == "A train."
    assert indirect_text("object", description, OBJECT_TEMPLATES) == (
        "A vehicle that is found at a railway and can carry passengers."
    )


def test_two_properties_of_one_kind_share_the_verb():
    index = build_fact_index(EDGES)
    description = choose_description("baseball", ["baseball", "tennis"], index, ["sport"], 2)

    assert indirect_text("activity", description, ACTIVITY_TEMPLATES) == (
        "A person playing a sport that uses a bat and a glove."
    )
