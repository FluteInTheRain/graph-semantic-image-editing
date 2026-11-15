"""Facts per concept and the unique description (ADR-0050). Synthetic edges only."""

from conceptnet_fixtures import EDGES, edge
from mmxai.queries.kg_facts import build_fact_index, choose_description, leaks_name

CATEGORIES = ["sport", "vehicle"]
SPORTS = ["tennis", "badminton", "baseball"]
VEHICLES = ["train", "bus", "car"]


def test_facts_follow_the_edge_direction():
    index = build_fact_index(EDGES)

    kinds = sorted((fact.kind, fact.value) for fact in index["tennis"])

    assert kinds == [
        ("isa", "sport"),
        ("location", "court"),
        ("uses", "racket"),
        ("uses", "tennis ball"),
    ]
    assert ("can", "carry passengers") in {(fact.kind, fact.value) for fact in index["train"]}


def test_description_is_unique_and_prefers_rare_properties():
    index = build_fact_index(EDGES)

    description = choose_description("tennis", SPORTS, index, CATEGORIES, num_properties=2)

    assert description.category.value == "sport"
    # "court" belongs to tennis only, so it is picked first; "tennis ball" leaks the name.
    assert [(fact.kind, fact.value) for fact in description.properties] == [
        ("location", "court"),
        ("uses", "racket"),
    ]


def test_no_description_when_every_property_is_shared():
    index = build_fact_index(EDGES)

    # car: vehicle + carry passengers + street, all shared with bus.
    assert choose_description("car", VEHICLES, index, CATEGORIES, num_properties=2) is None


def test_no_description_without_a_known_category():
    index = build_fact_index(EDGES)

    assert choose_description("tennis", SPORTS, index, ["animal"], num_properties=2) is None


def test_leaks_name_checks_whole_words():
    assert leaks_name("tennis ball", "tennis")
    assert leaks_name("tennis", "tennis")
    assert not leaks_name("court", "tennis")


def test_category_that_repeats_the_name_is_skipped():
    index = build_fact_index([edge("IsA", "tennis", "tennis sport")])

    # The only category word would put "tennis" into the query.
    assert choose_description("tennis", [], index, ["tennis sport"], num_properties=0) is None
