"""Visual Genome objects -> concepts per image, on tiny synthetic annotations (ADR-0045).

Made-up annotations, not Visual Genome data. Images 1-3 are train, 4-5 test:
    1 train  dog (dog.n.01), puppy (dog.n.01), couch (sofa.n.01)
    2 train  couch (sofa.n.01), couch (couch.n.03), grass (grass.n.01)
    3 train  couch (sofa.n.01), top (no synset)
    4 test   couch (no synset), zebra (no synset), cat (cat.n.01)
    5 test   zebra (zebra.n.01)          zebra has a synset only in test
"""

import numpy as np

from mmxai.data.scene_graph import build_scene_graphs
from mmxai.graph.visual_genome import (
    build_name_to_synset,
    concept_text,
    image_concepts_from_scene_graphs,
)


def obj(object_id: int, name: str, synsets: list[str]) -> dict:
    return {"object_id": object_id, "names": [name], "synsets": synsets}


OBJECTS = [
    {
        "image_id": 1,
        "objects": [
            obj(11, "dog", ["dog.n.01"]),
            obj(12, "puppy", ["dog.n.01"]),
            obj(13, "couch", ["sofa.n.01"]),
        ],
    },
    {
        "image_id": 2,
        "objects": [
            obj(21, "couch", ["sofa.n.01"]),
            obj(22, "couch", ["couch.n.03"]),
            obj(23, "grass", ["grass.n.01"]),
        ],
    },
    {"image_id": 3, "objects": [obj(31, "couch", ["sofa.n.01"]), obj(32, "top", [])]},
    {
        "image_id": 4,
        "objects": [obj(41, "couch", []), obj(42, "zebra", []), obj(43, "cat", ["cat.n.01"])],
    },
    {"image_id": 5, "objects": [obj(51, "zebra", ["zebra.n.01"])]},
    {"image_id": 6, "objects": []},
]
TRAIN_IDS = np.array([1, 2, 3])


def scene_graphs():
    graphs, _ = build_scene_graphs(OBJECTS, [], image_ids=None)
    return graphs


def test_name_dictionary_picks_the_most_common_train_synset():
    name_to_synset = build_name_to_synset(scene_graphs(), TRAIN_IDS)

    # couch: sofa.n.01 three times, couch.n.03 once.
    assert name_to_synset["couch"] == "sofa.n.01"


def test_name_dictionary_is_learned_on_train_images_only():
    # zebra has a synset only in test image 5, so the dictionary must not know it.
    name_to_synset = build_name_to_synset(scene_graphs(), TRAIN_IDS)

    assert "zebra" not in name_to_synset
    assert "cat" not in name_to_synset


def test_test_images_cannot_change_the_synset_of_a_train_name():
    # Many test "couch" objects say couch.n.03; the train majority (sofa.n.01) must still win.
    flooded = list(OBJECTS)
    for number in range(10):
        flooded.append(
            {"image_id": 100 + number, "objects": [obj(1000 + number, "couch", ["couch.n.03"])]}
        )
    graphs, _ = build_scene_graphs(flooded, [], image_ids=None)

    assert build_name_to_synset(graphs, TRAIN_IDS) == build_name_to_synset(
        scene_graphs(), TRAIN_IDS
    )
    assert build_name_to_synset(graphs, TRAIN_IDS)["couch"] == "sofa.n.01"


def test_name_dictionary_breaks_ties_by_synset_order():
    tie = [
        {"image_id": 1, "objects": [obj(1, "bank", ["bank.n.09"]), obj(2, "bank", ["bank.n.01"])]}
    ]
    graphs, _ = build_scene_graphs(tie, [], image_ids=None)

    assert build_name_to_synset(graphs, np.array([1])) == {"bank": "bank.n.01"}


def test_objects_with_the_same_synset_become_one_concept():
    concepts, _ = image_concepts_from_scene_graphs(scene_graphs(), {})

    assert concepts[1] == {"dog.n.01", "sofa.n.01"}


def test_object_without_a_synset_gets_one_from_its_name():
    name_to_synset = build_name_to_synset(scene_graphs(), TRAIN_IDS)

    concepts, _ = image_concepts_from_scene_graphs(scene_graphs(), name_to_synset)

    # Test image 4: couch is looked up, zebra is unknown, cat keeps its own synset.
    assert concepts[4] == {"sofa.n.01", "cat.n.01"}


def test_synset_sources_are_counted():
    name_to_synset = build_name_to_synset(scene_graphs(), TRAIN_IDS)

    _, counts = image_concepts_from_scene_graphs(scene_graphs(), name_to_synset)

    # Annotated: 11 12 13 21 22 23 31 43 51. Looked up: 41 (couch). Unknown: 32 (top), 42 (zebra).
    assert counts == {"annotated": 9, "from_name": 1, "unmapped": 2}


def test_image_without_objects_has_no_concepts():
    concepts, _ = image_concepts_from_scene_graphs(scene_graphs(), {})

    assert concepts[6] == set()


def test_concept_text_is_the_synset_lemma_in_plain_words():
    assert concept_text("fire_hydrant.n.01") == "fire hydrant"
    assert concept_text("dog.n.01") == "dog"
