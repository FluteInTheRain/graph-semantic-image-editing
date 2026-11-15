"""KG query generation end to end on synthetic data (ADR-0050)."""

from dataclasses import replace

from conceptnet_fixtures import EDGES
from mmxai.data.scene_graph import Relation, SceneGraph, SceneObject
from mmxai.queries.generator import canonicalize_graphs
from mmxai.queries.kg_facts import build_fact_index
from mmxai.queries.kg_generator import KgQueryConfig, generate_kg_pairs

CONFIG = KgQueryConfig(
    dataset="visual_genome",
    data_dir="unused",
    objects_file="objects.json",
    relationships_file="relationships.json",
    split_file="unused",
    split="test",
    conceptnet_edges="unused",
    predicate_aliases={"plays": "playing"},
    activity_predicates=["playing"],
    person_synsets=["man.n.01"],
    category_words=["sport", "vehicle"],
    activity_templates={"uses": "uses", "location": "is played on", "has": "has", "can": "can"},
    object_templates={
        "uses": "is used with",
        "location": "is found at",
        "has": "has",
        "can": "can",
    },
    num_properties=2,
    min_train_images=1,
    min_answer_images=1,
    max_answer_images=10,
    limit_activity_pairs=5,
    limit_object_pairs=5,
    author="test",
    candidates_path="unused",
    manifest_path="unused",
    benchmark_path="unused",
    min_verified_pairs=1,
)


def scene(image_id: int, thing: str, predicate: str = "playing") -> SceneGraph:
    man = SceneObject(1, "man", "man", "man.n.01")
    target = SceneObject(2, thing.split(".")[0], thing.split(".")[0], thing)
    graph = SceneGraph(image_id, {1: man, 2: target})
    graph.relations.append(
        Relation(image_id, 1, "man", predicate, 2, target.name, predicate, "man", target.name)
    )
    return graph


def fixture_graphs() -> tuple[dict, dict]:
    """Canonical graphs, as run_kg_generation passes them ("plays" becomes "playing")."""
    train = {
        1: scene(1, "tennis.n.01"),
        2: scene(2, "badminton.n.01"),
        3: scene(3, "train.n.01", "near"),
        4: scene(4, "car.n.01", "near"),
        5: scene(5, "bus.n.01", "near"),  # makes car ambiguous (shared facts)
    }
    test = {
        100: scene(100, "tennis.n.01"),
        101: scene(101, "tennis.n.01", "plays"),
        102: scene(102, "train.n.01", "near"),
        103: scene(103, "car.n.01", "near"),  # car has answers, so it fails on the description
    }
    aliases = CONFIG.predicate_aliases
    return canonicalize_graphs(train, aliases), canonicalize_graphs(test, aliases)


def test_pairs_hide_the_concept_and_share_answers():
    train, test = fixture_graphs()

    records, rejections = generate_kg_pairs(train, test, build_fact_index(EDGES), CONFIG)

    tennis = [record for record in records if record.get("hidden_concept") == "tennis.n.01"]
    direct, indirect = sorted(tennis, key=lambda record: record["type"])
    assert direct["text"] == "A person playing tennis."
    assert "tennis" not in indirect["text"].lower()
    # "plays" is canonicalised to "playing", so image 101 is an answer too.
    assert direct["relevant"] == indirect["relevant"] == ["100", "101"]
    assert indirect["query_group"] == "activity"
    assert {item["start"] for item in indirect["kg_evidence"]} >= {"racket"}


def test_concepts_without_unique_description_or_answers_are_rejected():
    train, test = fixture_graphs()

    records, rejections = generate_kg_pairs(train, test, build_fact_index(EDGES), CONFIG)

    concepts = {record["hidden_concept"] for record in records}
    assert "car.n.01" not in concepts  # every fact shared with bus/car look-alikes
    assert "badminton.n.01" not in concepts  # no test image
    assert rejections["no_unique_description"] >= 1
    assert rejections["answer_count_out_of_range"] >= 1


def test_answers_never_leave_the_given_split():
    train, test = fixture_graphs()

    records, _ = generate_kg_pairs(train, test, build_fact_index(EDGES), CONFIG)

    for record in records:
        assert set(record["relevant"]) <= {str(image_id) for image_id in test}


def test_limits_per_group():
    train, test = fixture_graphs()
    config = replace(CONFIG, limit_activity_pairs=0)

    records, _ = generate_kg_pairs(train, test, build_fact_index(EDGES), config)

    assert {record["query_group"] for record in records} == {"object"}


def test_term_with_several_synsets_in_a_group_is_rejected():
    # bat.n.01 (animal) and bat.n.05 (club) would give two pairs with the same text.
    train = {1: scene(1, "bat.n.01", "near"), 2: scene(2, "bat.n.05", "near")}
    test = {10: scene(10, "bat.n.01", "near"), 11: scene(11, "bat.n.05", "near")}

    records, rejections = generate_kg_pairs(train, test, build_fact_index(EDGES), CONFIG)

    assert not [r for r in records if r["hidden_concept"].startswith("bat.")]
    assert rejections["term_has_several_synsets"] == 2
