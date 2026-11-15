"""Direct/indirect sentences for KG queries (ADR-0050)."""

from conceptnet_fixtures import EDGES, edge
from mmxai.queries.kg_facts import build_fact_index, choose_description
from mmxai.queries.kg_text import direct_text, indirect_text, with_article

ACTIVITY_TEMPLATES = {"uses": "uses", "location": "is played on", "has": "has", "can": "can"}
OBJECT_TEMPLATES = {"uses": "is used with", "location": "is found at", "has": "has", "can": "can"}
PLAY_CATEGORIES = ["sport", "game"]


def test_with_article():
    assert with_article("racket") == "a racket"
    assert with_article("apple") == "an apple"


def test_activity_sentences():
    index = build_fact_index(EDGES)
    description = choose_description(
        "tennis", ["tennis", "badminton"], index, ["sport"], 2, min_fact_weight=1.0
    )

    assert direct_text("activity", "tennis") == "A person playing tennis."
    assert indirect_text("activity", description, ACTIVITY_TEMPLATES, PLAY_CATEGORIES) == (
        "A person playing a sport that is played on a court and uses a racket."
    )


def test_object_sentences_keep_verb_values_without_article():
    index = build_fact_index(EDGES)
    description = choose_description(
        "train", ["train", "bus", "car"], index, ["vehicle"], 2, min_fact_weight=1.0
    )

    assert direct_text("object", "train") == "A train."
    assert indirect_text("object", description, OBJECT_TEMPLATES, PLAY_CATEGORIES) == (
        "A vehicle that is found at a railway and can carry passengers."
    )


def test_two_properties_of_one_kind_share_the_verb():
    index = build_fact_index(EDGES)
    description = choose_description(
        "baseball", ["baseball", "tennis"], index, ["sport"], 2, min_fact_weight=1.0
    )

    assert indirect_text("activity", description, ACTIVITY_TEMPLATES, PLAY_CATEGORIES) == (
        "A person playing a sport that uses a bat and a glove."
    )


def test_activity_with_a_non_play_category_is_played_with_a_toy():
    # Pilot val: "A person playing a toy that ..." read badly; a toy is played WITH.
    index = build_fact_index(
        [
            edge("IsA", "frisbee", "toy"),
            edge("IsA", "ball", "toy"),
            edge("UsedFor", "park", "frisbee"),
        ]
    )
    description = choose_description("frisbee", ["frisbee", "ball"], index, ["toy"], 1, 1.0)

    assert indirect_text("activity", description, ACTIVITY_TEMPLATES, PLAY_CATEGORIES) == (
        "A person playing with a toy that uses a park."
    )


def test_object_group_ignores_play_categories():
    index = build_fact_index([edge("IsA", "train", "vehicle"), edge("AtLocation", "train", "rail")])
    description = choose_description("train", ["train"], index, ["vehicle"], 1, 1.0)

    assert indirect_text("object", description, OBJECT_TEMPLATES, ["vehicle"]) == (
        "A vehicle that is found at a rail."
    )
