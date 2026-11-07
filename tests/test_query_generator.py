"""Direct/indirect query generation from synthetic scene graphs (ADR-0044).

The person/frisbee/dog scene is a made-up fixture, not a Visual Genome example.
"""

from collections import Counter
from dataclasses import replace

import pytest
import yaml

from mmxai.data.scene_graph import Relation, SceneGraph, SceneObject
from mmxai.queries.generator import (
    REJECT_AMBIGUOUS_NAME,
    REJECT_GENERIC_NAME,
    REJECT_NO_ACTION,
    REJECT_NON_WORD_NAME,
    REJECT_PREDICATE_NOT_ALLOWED,
    REJECT_REPEATED_KEY,
    REJECT_SHAPE,
    canonical_predicate,
    canonicalize_graphs,
    chain_problem,
    direct_text,
    enumerate_chains,
    first_mention,
    generate_pairs,
    indirect_text,
    is_plural,
    load_query_benchmark_config,
    reasoning_type,
    relation_evidence,
    usable_relations,
)
from scene_graph_fixtures import CONFIG, TEST_IDS, canonical_fixture_graphs


def fixture_graphs(image_ids: list[int]) -> dict[int, SceneGraph]:
    return canonical_fixture_graphs(image_ids)


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

    assert direct["text"] == "A person is holding a frisbee, and a dog is looking at the frisbee."
    assert indirect["text"] == "A person is holding the thing that a dog is looking at."
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


def test_canonical_predicate_strips_copula_article_and_applies_aliases():
    aliases = {"wears": "wearing"}

    assert canonical_predicate("on a", aliases) == "on"
    assert canonical_predicate("are on", aliases) == "on"
    assert canonical_predicate("is sitting on the", aliases) == "sitting on"
    assert canonical_predicate("wears", aliases) == "wearing"
    # Only whole trailing words are articles: "near" must not lose its "a".
    assert canonical_predicate("near", aliases) == "near"


def test_canonicalize_graphs_keeps_the_raw_predicate():
    graph = make_graph([(1, 1, "wears", 2)], {1: "man", 2: "hat"})

    canonical = canonicalize_graphs({1: graph}, {"wears": "wearing"})

    assert canonical[1].relations[0].predicate == "wearing"
    assert canonical[1].relations[0].raw_predicate == "wears"
    assert graph.relations[0].predicate == "wears"  # input left unchanged


def test_ambiguous_generic_and_non_whitelisted_relations_are_rejected():
    names = {1: "tree", 2: "tree", 3: "dog", 4: "it", 5: "leg"}
    graph = make_graph([(1, 1, "near", 3), (2, 4, "on", 3), (3, 5, "of", 3)], names)
    rejections = Counter()

    usable = usable_relations(graph, ["it"], {"near": "is near", "on": "is on"}, rejections)

    assert usable == []
    assert rejections == Counter(
        {REJECT_AMBIGUOUS_NAME: 1, REJECT_GENERIC_NAME: 1, REJECT_PREDICATE_NOT_ALLOWED: 1}
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


def test_path_chain_reads_correctly():
    graph = make_graph([(1, 1, "riding", 2), (2, 2, "on", 3)], {1: "man", 2: "horse", 3: "beach"})
    chain = enumerate_chains(1, graph.relations, 2, 2)[0]
    templates = CONFIG.predicate_templates

    assert direct_text(chain, graph, templates) == (
        "A man is riding a horse, and the horse is on a beach."
    )
    assert indirect_text(chain, graph, templates) == "A man is riding something that is on a beach."


def test_path_chain_reads_the_same_when_found_backwards():
    # The walk starts at the lowest id (beach), so node order is beach, horse, man.
    graph = make_graph([(1, 3, "on", 1), (2, 2, "riding", 3)], {1: "beach", 2: "man", 3: "horse"})
    chain = enumerate_chains(1, graph.relations, 2, 2)[0]

    assert indirect_text(chain, graph, CONFIG.predicate_templates) == (
        "A man is riding something that is on a beach."
    )


def test_plural_names_get_some_and_are():
    shoes = SceneObject(1, "shoes", "shoes", "shoe.n.01")
    glass = SceneObject(2, "glass", "glass", "glass.n.02")
    man = SceneObject(3, "man", "man", "")
    apple = SceneObject(4, "apple", "apple", "apple.n.01")

    assert is_plural(shoes)
    assert not is_plural(glass)
    assert not is_plural(man)
    assert first_mention(shoes) == "some shoes"
    assert first_mention(apple) == "an apple"
    assert first_mention(man) == "a man"


def test_plural_subject_takes_are():
    graph = make_graph([(1, 1, "on", 2), (2, 3, "wearing", 1)], {1: "shoes", 2: "floor", 3: "man"})
    graph.objects[1] = SceneObject(1, "shoes", "shoes", "shoe.n.01")
    chain = enumerate_chains(1, graph.relations, 2, 2)[0]

    assert direct_text(chain, graph, CONFIG.predicate_templates) == (
        "A man is wearing some shoes, and the shoes are on a floor."
    )


def test_shared_subject_chains_are_rejected():
    graph = make_graph(
        [(1, 1, "holding", 2), (2, 1, "wearing", 3)], {1: "man", 2: "umbrella", 3: "hat"}
    )
    chain = enumerate_chains(1, graph.relations, 2, 2)[0]

    assert reasoning_type(chain) == "shared_subject"
    assert chain_problem(chain, graph, set(), CONFIG) == REJECT_SHAPE


def test_chain_without_an_action_predicate_is_rejected():
    graph = make_graph([(1, 1, "on", 2), (2, 3, "near", 2)], {1: "cup", 2: "table", 3: "plate"})
    chain = enumerate_chains(1, graph.relations, 2, 2)[0]

    assert reasoning_type(chain) == "shared_object"
    assert chain_problem(chain, graph, set(), CONFIG) == REJECT_NO_ACTION


def test_chain_whose_ends_share_a_synset_is_rejected():
    names = {1: "man", 2: "frisbee", 3: "guy"}
    graph = make_graph([(1, 1, "holding", 2), (2, 3, "looking at", 2)], names)
    graph.objects[1] = SceneObject(1, "man", "man", "man.n.01")
    graph.objects[3] = SceneObject(3, "guy", "guy", "man.n.01")
    chain = enumerate_chains(1, graph.relations, 2, 2)[0]

    assert chain_problem(chain, graph, set(), CONFIG) == REJECT_REPEATED_KEY


def test_path_chain_with_an_action_is_accepted():
    graph = make_graph([(1, 1, "riding", 2), (2, 2, "on", 3)], {1: "man", 2: "horse", 3: "beach"})
    chain = enumerate_chains(1, graph.relations, 2, 2)[0]

    assert reasoning_type(chain) == "path"
    assert chain_problem(chain, graph, set(), CONFIG) is None


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

    with pytest.raises(ValueError, match="2 hops"):
        load_query_benchmark_config(path)


def test_config_rejects_action_predicates_without_template(tmp_path):
    values = {field: getattr(CONFIG, field) for field in CONFIG.__dataclass_fields__}
    values["action_predicates"] = ["holding", "flying"]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")

    with pytest.raises(ValueError, match="flying"):
        load_query_benchmark_config(path)


def test_config_rejects_more_than_two_hops(tmp_path):
    values = {field: getattr(CONFIG, field) for field in CONFIG.__dataclass_fields__}
    values["max_hops"] = 3
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")

    with pytest.raises(ValueError, match="2 hops"):
        load_query_benchmark_config(path)


def test_evidence_carries_the_synsets_of_both_objects():
    graph = make_graph([(1, 1, "riding", 2), (2, 2, "on", 3)], {1: "man", 2: "horse", 3: "beach"})
    graph.objects[1] = SceneObject(1, "man", "man", "man.n.01")
    chain = enumerate_chains(1, graph.relations, 2, 2)[0]

    first = relation_evidence(chain, graph)[0]

    assert first["subject_synset"] == "man.n.01"
    assert first["object_synset"] == ""


def test_names_with_digits_or_symbols_are_rejected():
    names = {1: "52", 2: "green/white uniform", 3: "baseball player", 4: "hat"}
    graph = make_graph([(1, 1, "on", 2), (2, 3, "wearing", 4)], names)
    rejections = Counter()

    usable = usable_relations(graph, ["it"], CONFIG.predicate_templates, rejections)

    assert [relation.relationship_id for relation in usable] == [2]
    assert rejections == Counter({REJECT_NON_WORD_NAME: 1})
