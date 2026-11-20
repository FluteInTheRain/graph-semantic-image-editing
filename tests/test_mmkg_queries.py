import pytest

from mmxai.data.mmkg_queries import (
    DEFAULT_HEAD,
    Entity,
    build_queries,
    indirect_text,
    matching_items,
    query_record,
)

ENTITIES = {
    "I1": Entity("I1", "Ngoc Lu drum", "Trống Ngọc Lũ", "item"),
    "I2": Entity("I2", "Hoang Ha drum", "Trống Hoàng Hạ", "item"),
    "I3": Entity("I3", "Thien Mu Pagoda", "Chùa Thiên Mụ", "item"),
    "C_drum": Entity("C_drum", "bronze drum", "trống đồng", "concept"),
    "C_bronze": Entity("C_bronze", "bronze", "đồng", "concept"),
    "C_museum": Entity("C_museum", "National Museum", "Bảo tàng Quốc gia", "concept"),
    "C_treasure": Entity("C_treasure", "National Treasure", "Bảo vật quốc gia", "concept"),
}
TRIPLES = [
    {"head_id": "I1", "relation": "instance_of", "tail_id": "C_drum"},
    {"head_id": "I1", "relation": "made_from", "tail_id": "C_bronze"},
    {"head_id": "I1", "relation": "kept_at", "tail_id": "C_museum"},
    {"head_id": "I2", "relation": "instance_of", "tail_id": "C_drum"},
    {"head_id": "I2", "relation": "made_from", "tail_id": "C_bronze"},
    {"head_id": "I3", "relation": "heritage_designation", "tail_id": "C_treasure"},
]
ITEMS = ["I1", "I2", "I3"]


def test_indirect_text_hides_the_name_and_reads_the_triples():
    pairs = {("instance_of", "C_drum"), ("made_from", "C_bronze"), ("kept_at", "C_museum")}

    english = indirect_text(pairs, ENTITIES, "en")
    vietnamese = indirect_text(pairs, ENTITIES, "vi")

    assert english == "a bronze drum, made of bronze, kept at National Museum"
    assert vietnamese == "trống đồng, làm bằng đồng, lưu giữ tại Bảo tàng Quốc gia"
    assert "Ngoc Lu" not in english and "Ngọc Lũ" not in vietnamese


def test_indirect_text_without_instance_of_uses_the_default_head():
    text = indirect_text({("heritage_designation", "C_treasure")}, ENTITIES, "en")

    assert text == f"{DEFAULT_HEAD[0]}, listed as National Treasure"


def test_indirect_text_does_not_depend_on_pair_order():
    pairs = [("made_from", "C_bronze"), ("instance_of", "C_drum"), ("instance_of", "C_museum")]

    assert indirect_text(set(pairs), ENTITIES, "en") == indirect_text(
        set(reversed(pairs)), ENTITIES, "en"
    )


def test_every_item_that_has_the_described_pairs_is_relevant():
    pairs_by_item = {
        "I1": {("instance_of", "C_drum"), ("made_from", "C_bronze"), ("kept_at", "C_museum")},
        "I2": {("instance_of", "C_drum"), ("made_from", "C_bronze")},
    }

    # I2's description also fits I1, which has the same pairs and one more.
    assert matching_items(pairs_by_item["I2"], pairs_by_item, ITEMS) == ("I1", "I2")
    # I1's description is more specific: only I1 fits.
    assert matching_items(pairs_by_item["I1"], pairs_by_item, ITEMS) == ("I1",)


def test_build_queries_writes_four_queries_per_item():
    queries = build_queries(ENTITIES, TRIPLES, ITEMS)

    assert len(queries) == 4 * len(ITEMS)
    assert {(query.kind, query.lang) for query in queries if query.item_id == "I1"} == {
        ("direct", "en"),
        ("direct", "vi"),
        ("indirect", "en"),
        ("indirect", "vi"),
    }
    assert len({query.qid for query in queries}) == len(queries)


def test_direct_query_is_the_name_and_only_that_item_is_relevant():
    queries = {query.qid: query for query in build_queries(ENTITIES, TRIPLES, ITEMS)}

    assert queries["I1_direct_vi"].text == "Trống Ngọc Lũ"
    assert queries["I1_direct_vi"].relevant_items == ("I1",)


def test_indirect_queries_of_both_languages_share_the_answer():
    queries = {query.qid: query for query in build_queries(ENTITIES, TRIPLES, ITEMS)}

    assert queries["I2_indirect_en"].relevant_items == ("I1", "I2")
    assert queries["I2_indirect_vi"].relevant_items == ("I1", "I2")


def test_unknown_relation_is_refused():
    triples = TRIPLES + [{"head_id": "I1", "relation": "sister_city", "tail_id": "I3"}]

    with pytest.raises(ValueError, match="sister_city"):
        build_queries(ENTITIES, triples, ITEMS)


def test_item_without_triples_gets_direct_queries_only():
    queries = build_queries(ENTITIES, TRIPLES[:-1], ITEMS)

    assert sorted(query.qid for query in queries if query.item_id == "I3") == [
        "I3_direct_en",
        "I3_direct_vi",
    ]


def test_query_record_lists_every_image_of_every_relevant_item():
    queries = {query.qid: query for query in build_queries(ENTITIES, TRIPLES, ITEMS)}
    images_by_item = {"I1": ["I1_a", "I1_b"], "I2": ["I2_a"], "I3": ["I3_a"]}

    record = query_record(queries["I2_indirect_en"], images_by_item)

    assert record["relevant"] == ["I1_a", "I1_b", "I2_a"]
    assert record["relevant_items"] == ["I1", "I2"]
    assert record["type"] == "indirect" and record["lang"] == "en"
    assert record["dataset"] == "mmkg_vi"
