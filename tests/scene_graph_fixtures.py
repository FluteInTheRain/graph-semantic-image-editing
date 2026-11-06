"""Tiny SYNTHETIC Visual Genome annotations for the query benchmark tests.

Nothing here comes from Visual Genome. The person/frisbee/dog scene is made up
to have a known answer; do not cite it as a real VG example.

Images and their (made-up) split:
    100 test  person -holding-> frisbee, dog -looks at-> frisbee, two trees (ambiguous)
    200 test  the same person/frisbee/dog chain, other object ids
    300 test  person -holding-> frisbee, but the dog looks at a ball
    400 val   the same chain (must never be an answer of a test query)
    500 train the same chain
"""

import json
from pathlib import Path

from mmxai.data.scene_graph import SceneGraph, build_scene_graphs
from mmxai.queries.generator import QueryBenchmarkConfig, canonicalize_graphs

TEST_IDS = [100, 200, 300]
VAL_IDS = [400]
TRAIN_IDS = [500]

# Generator settings for the tests; paths are filled in by tests that write files.
CONFIG = QueryBenchmarkConfig(
    dataset="visual_genome",
    data_dir="unused",
    objects_file="objects.json",
    relationships_file="relationships.json",
    split_file="unused",
    split="test",
    seed=0,
    limit_pairs=10,
    min_hops=2,
    max_hops=2,
    max_pairs_per_image=1,
    max_candidate_images=25,
    generic_names=["it", "thing"],
    predicate_aliases={"looks at": "looking at", "rides": "riding"},
    predicate_templates={
        "holding": "is holding",
        "looking at": "is looking at",
        "riding": "is riding",
        "wearing": "is wearing",
        "sitting on": "is sitting on",
        "on": "is on",
        "near": "is near",
    },
    action_predicates=["holding", "looking at", "riding", "wearing", "sitting on"],
    author="test",
    candidates_path="unused",
    manifest_path="unused",
    benchmark_path="unused",
    min_verified_pairs=1,
)


def obj(object_id: int, name: str) -> dict:
    return {"object_id": object_id, "names": [name], "x": 0, "y": 0, "w": 1, "h": 1}


def rel(relationship_id: int, subject: dict, predicate: str, target: dict) -> dict:
    return {
        "relationship_id": relationship_id,
        "predicate": predicate,
        # VG writes the subject's name as "name" and the object's as "names"; mimic both.
        "subject": {"object_id": subject["object_id"], "name": subject["names"][0]},
        "object": target,
    }


def frisbee_scene(image_id: int, base: int) -> tuple[dict, dict]:
    """objects.json and relationships.json entries of one person/frisbee/dog image."""
    person = obj(base + 1, "Person")
    frisbee = obj(base + 2, "frisbee")
    dog = obj(base + 3, "dog")
    objects = {"image_id": image_id, "objects": [person, frisbee, dog]}
    relationships = {
        "image_id": image_id,
        "relationships": [
            rel(base * 10 + 1, person, "HOLDING", frisbee),
            rel(base * 10 + 2, dog, "looks_at", frisbee),
        ],
    }
    return objects, relationships


def synthetic_annotations() -> tuple[list[dict], list[dict]]:
    """(objects entries, relationships entries) for images 100..500."""
    objects_entries = []
    relationships_entries = []
    for image_id, base in [(100, 0), (200, 20), (400, 40), (500, 50)]:
        objects, relationships = frisbee_scene(image_id, base)
        objects_entries.append(objects)
        relationships_entries.append(relationships)

    # Image 100 also has two trees: "the tree" would be ambiguous.
    tree_a = obj(4, "tree")
    tree_b = obj(5, "tree")
    objects_entries[0]["objects"].extend([tree_a, tree_b])
    relationships_entries[0]["relationships"].append(rel(3, tree_a, "near", obj(3, "dog")))

    person = obj(31, "person")
    frisbee = obj(32, "frisbee")
    dog = obj(33, "dog")
    ball = obj(34, "ball")
    objects_entries.append({"image_id": 300, "objects": [person, frisbee, dog, ball]})
    relationships_entries.append(
        {
            "image_id": 300,
            "relationships": [
                rel(301, person, "holding", frisbee),
                rel(302, dog, "looks at", ball),
            ],
        }
    )
    return objects_entries, relationships_entries


def canonical_fixture_graphs(image_ids: list[int]) -> dict[int, SceneGraph]:
    """Fixture graphs of the given images, with predicates canonicalised as in run_generation."""
    graphs, _ = build_scene_graphs(*synthetic_annotations(), image_ids=set(image_ids))
    return canonicalize_graphs(graphs, CONFIG.predicate_aliases)


def write_annotations(data_dir: Path) -> None:
    objects_entries, relationships_entries = synthetic_annotations()
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "objects.json").write_text(json.dumps(objects_entries), encoding="utf-8")
    (data_dir / "relationships.json").write_text(
        json.dumps(relationships_entries), encoding="utf-8"
    )


def write_split(path: Path) -> None:
    split = {
        "dataset": "visual_genome",
        "seed": 0,
        "train": [str(image_id) for image_id in TRAIN_IDS],
        "val": [str(image_id) for image_id in VAL_IDS],
        "test": [str(image_id) for image_id in TEST_IDS],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split), encoding="utf-8")
