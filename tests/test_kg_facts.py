"""Facts per concept and the unique description (ADR-0050). Synthetic edges only."""

from conceptnet_fixtures import EDGES, edge
from mmxai.queries.kg_facts import build_fact_index, choose_description, leaks_name, looks_plural

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

    description = choose_description(
        "tennis", SPORTS, index, CATEGORIES, num_properties=2, min_fact_weight=1.0
    )

    assert description.category.value == "sport"
    # "court" belongs to tennis only, so it is picked first; "tennis ball" leaks the name.
    assert [(fact.kind, fact.value) for fact in description.properties] == [
        ("location", "court"),
        ("uses", "racket"),
    ]


def test_no_description_when_every_property_is_shared():
    index = build_fact_index(EDGES)

    # car: vehicle + carry passengers + street, all shared with bus.
    assert (
        choose_description(
            "car", VEHICLES, index, CATEGORIES, num_properties=2, min_fact_weight=1.0
        )
        is None
    )


def test_no_description_without_a_known_category():
    index = build_fact_index(EDGES)

    assert (
        choose_description(
            "tennis", SPORTS, index, ["animal"], num_properties=2, min_fact_weight=1.0
        )
        is None
    )


def test_leaks_name_checks_whole_words():
    assert leaks_name("tennis ball", "tennis")
    assert leaks_name("tennis", "tennis")
    assert not leaks_name("court", "tennis")


def test_category_that_repeats_the_name_is_skipped():
    index = build_fact_index([edge("IsA", "tennis", "tennis sport")])

    # The only category word would put "tennis" into the query.
    assert (
        choose_description(
            "tennis", [], index, ["tennis sport"], num_properties=0, min_fact_weight=1.0
        )
        is None
    )


def test_looks_plural_is_a_deliberately_narrow_rule():
    assert looks_plural("legs")
    assert looks_plural("glasses")
    assert not looks_plural("glass")
    assert not looks_plural("grass")
    # Known false positive: "bus" counts as plural. It only removes a candidate
    # property, never a concept, so the cost is one lost property.
    assert looks_plural("bus")


def plural_index(extra: list) -> dict:
    return build_fact_index(
        [
            edge("IsA", "tennis", "sport"),
            edge("IsA", "badminton", "sport"),
            edge("HasA", "tennis", "legs"),
            edge("UsedFor", "racket", "tennis"),
            edge("UsedFor", "racket", "badminton"),
            *extra,
        ]
    )


def test_a_plural_property_is_never_chosen():
    # "legs" is the only fact that tells tennis from badminton, but it reads "has a legs".
    index = plural_index([])

    assert (
        choose_description(
            "tennis", ["tennis", "badminton"], index, ["sport"], 1, min_fact_weight=1.0
        )
        is None
    )


def test_a_non_plural_property_is_chosen_instead_of_the_plural_one():
    index = plural_index([edge("AtLocation", "tennis", "court")])

    description = choose_description(
        "tennis", ["tennis", "badminton"], index, ["sport"], 1, min_fact_weight=1.0
    )

    assert [fact.value for fact in description.properties] == ["court"]


def test_plural_values_are_fine_for_can_facts():
    index = build_fact_index(
        [
            edge("IsA", "train", "vehicle"),
            edge("CapableOf", "train", "carry passengers"),
        ]
    )

    description = choose_description("train", ["train"], index, ["vehicle"], 1, min_fact_weight=1.0)

    assert [fact.value for fact in description.properties] == ["carry passengers"]


def test_a_property_below_min_fact_weight_is_skipped():
    # Pilot val 2026-11-15: weight-1 ConceptNet facts were often odd ("cable found at ...").
    index = build_fact_index(
        [
            edge("IsA", "tennis", "sport"),
            edge("IsA", "badminton", "sport"),
            edge("AtLocation", "tennis", "court", weight=1.0),
        ]
    )
    pool = ["tennis", "badminton"]

    weak_allowed = choose_description("tennis", pool, index, ["sport"], 1, min_fact_weight=1.0)
    strong_only = choose_description("tennis", pool, index, ["sport"], 1, min_fact_weight=2.0)

    assert [fact.value for fact in weak_allowed.properties] == ["court"]
    assert strong_only is None


def test_a_weak_fact_of_another_concept_still_makes_a_description_ambiguous():
    # badminton's "racket" fact has weight 1.0, below the threshold, but it is still true,
    # so a query "uses a racket" would also fit badminton.
    index = build_fact_index(
        [
            edge("IsA", "tennis", "sport"),
            edge("IsA", "badminton", "sport"),
            edge("UsedFor", "racket", "tennis", weight=2.0),
            edge("UsedFor", "racket", "badminton", weight=1.0),
        ]
    )

    description = choose_description(
        "tennis", ["tennis", "badminton"], index, ["sport"], 1, min_fact_weight=2.0
    )

    assert description is None
