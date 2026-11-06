"""Direct/indirect query generation from synthetic scene graphs (ADR-0044).

The person/frisbee/dog scene is a made-up fixture, not a Visual Genome example.
"""

from collections import Counter
from dataclasses import replace

import pytest
import yaml

from mmxai.data.scene_graph import Relation, SceneGraph, SceneObject, build_scene_graphs
from mmxai.queries.generator import (
    REJECT_AMBIGUOUS_NAME,
    REJECT_GENERIC_NAME,
    REJECT_TRIVIAL_PREDICATE,
    enumerate_chains,
    generate_pairs,
    indirect_text,
    load_query_benchmark_config,
    reasoning_type,
    usable_relations,
    verb_phrase,
)
from scene_graph_fixtures import CONFIG, TEST_IDS, synthetic_annotations


def fixture_graphs(image_ids: list[int]) -> dict[int, SceneGraph]:
    graphs, _ = build_scene_graphs(*synthetic_annotations(), image_ids=set(image_ids))
    return graphs


def make_graph(relations: list[tuple[int, int, str, int]], names: dict[int, str]) -> SceneGraph:
    """Build a graph by hand: relations are (relationship_id, subject_id, predicate, object_id)."""
    graph = SceneGraph(1)
    for object_id, name in names.items():
        graph.objects[object_id] = SceneObject(object_id, name, name)
    for relationship_id, subject_id, predicate, object_id in relations:
        graph.relations.append(
            Relation(
                relationship_id,
                subject_id,
                names[subject_id],
                predicate,
                object_id,
                names[object_id],
                predicate,
                names[subject_id],
                names[object_id],
            )
        )
    return graph


def only_pair(records: list[dict]) -> tuple[dict, dict]:
    assert len(records) == 2
    direct = next(record for record in records if record["type"] == "direct")
    indirect = next(record for record in records if record["type"] == "indirect")
    return direct, indirect


def test_fixture_gives_the_expected_direct_and_indirect_text():
    records, _ = generate_pairs(fixture_graphs([100, 300]), CONFIG)
    direct, indirect = only_pair(records)

    assert direct["text"] == "A person is holding a frisbee, and a dog looks at the frisbee."
    assert indirect["text"] == "A person is holding the thing that a dog looks at."
    assert indirect["hidden_names"] == ["frisbee"]
    assert indirect["reasoning_type"] == "shared_object"


def test_pair_shares_id_evidence_and_answers():
    records, _ = generate_pairs(fixture_graphs([100, 300]), CONFIG)
    direct, indirect = only_pair(records)

    assert direct["pair_id"] == indirect["pair_id"]
    assert direct["relevant"] == indirect["relevant"]
    assert [item["relationship_id"] for item in indirect["relation_evidence"]] == [1, 2]
    assert indirect["chain_object_ids"] == [1, 2, 3]
    assert indirect["verification_status"] == "candidate"


def test_answers_are_every_test_image_with_the_chain_not_only_the_source():
    records, _ = generate_pairs(fixture_graphs(TEST_IDS), CONFIG)
    direct, _ = only_pair(records)

    # 100 and 200 hold the same chain; 300 has a dog looking at a ball instead.
    assert direct["relevant"] == ["100", "200"]


def test_answers_never_come_from_images_outside_the_given_split():
    # Val image 400 and train image 500 hold the same chain but are not loaded.
    records, _ = generate_pairs(fixture_graphs(TEST_IDS), CONFIG)

    for record in records:
        assert set(record["relevant"]) <= {str(image_id) for image_id in TEST_IDS}


def test_ambiguous_names_generic_names_and_trivial_predicates_are_rejected():
    names = {1: "tree", 2: "tree", 3: "dog", 4: "it", 5: "leg"}
    graph = make_graph([(1, 1, "near", 3), (2, 4, "on", 3), (3, 5, "of", 3)], names)
    rejections = Counter()

    usable = usable_relations(graph, ["it"], ["of"], rejections)

    assert usable == []
    assert rejections == Counter(
        {REJECT_AMBIGUOUS_NAME: 1, REJECT_GENERIC_NAME: 1, REJECT_TRIVIAL_PREDICATE: 1}
    )


def test_indirect_text_never_names_the_hidden_object():
    records, _ = generate_pairs(fixture_graphs(TEST_IDS), CONFIG)

    for record in records:
        if record["type"] == "indirect":
            for name in record["hidden_names"]:
                assert name not in record["text"].lower()


def test_enumerate_chains_respects_min_and_max_hops():
    names = {1: "man", 2: "horse", 3: "beach", 4: "sea"}
    graph = make_graph([(1, 1, "riding", 2), (2, 2, "on", 3), (3, 3, "near", 4)], names)

    two_hops = enumerate_chains(1, graph.relations, 2, 2)
    up_to_three = enumerate_chains(1, graph.relations, 2, 3)

    assert sorted(chain.hop_count for chain in two_hops) == [2, 2]
    assert sorted(chain.hop_count for chain in up_to_three) == [2, 2, 3]


def test_path_and_shared_subject_chains_read_correctly():
    path_graph = make_graph(
        [(1, 1, "riding", 2), (2, 2, "on", 3)], {1: "man", 2: "horse", 3: "beach"}
    )
    path_chain = enumerate_chains(1, path_graph.relations, 2, 2)[0]
    subject_graph = make_graph(
        [(1, 1, "holding", 2), (2, 1, "wearing", 3)], {1: "man", 2: "umbrella", 3: "hat"}
    )
    subject_chain = enumerate_chains(1, subject_graph.relations, 2, 2)[0]

    assert reasoning_type(path_chain) == "path"
    assert indirect_text(path_chain, path_graph) == "A man is riding the thing that is on a beach."
    assert reasoning_type(subject_chain) == "shared_subject"
    assert indirect_text(subject_chain, subject_graph) == (
        "The thing that is wearing a hat is holding an umbrella."
    )


def test_verb_phrase_adds_is_only_where_needed():
    assert verb_phrase("holding") == "is holding"
    assert verb_phrase("on") == "is on"
    assert verb_phrase("parked on") == "is parked on"
    assert verb_phrase("has") == "has"
    assert verb_phrase("looks at") == "looks at"


def test_generation_is_deterministic_for_a_seed():
    first, _ = generate_pairs(fixture_graphs(TEST_IDS), CONFIG)
    second, _ = generate_pairs(fixture_graphs(TEST_IDS), CONFIG)

    assert first == second


def test_limit_pairs_and_duplicate_texts_are_respected():
    records, _ = generate_pairs(fixture_graphs([100, 200]), replace(CONFIG, limit_pairs=1))

    assert len(records) == 2
    # With a higher limit, the second image repeats the first one's sentence and is skipped.
    records, rejections = generate_pairs(fixture_graphs(TEST_IDS), CONFIG)
    assert len(records) == 2
    assert rejections["duplicate_query_text"] == 1


def test_too_generic_chain_is_rejected():
    config = replace(CONFIG, max_candidate_images=1)

    records, rejections = generate_pairs(fixture_graphs(TEST_IDS), config)

    assert records == []
    assert rejections["too_many_candidate_images"] >= 1


def test_config_rejects_fewer_than_two_hops(tmp_path):
    values = {field: getattr(CONFIG, field) for field in CONFIG.__dataclass_fields__}
    values["min_hops"] = 1
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")

    with pytest.raises(ValueError, match="min_hops"):
        load_query_benchmark_config(path)
