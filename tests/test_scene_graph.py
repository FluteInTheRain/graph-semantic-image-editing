"""Visual Genome scene graph parser on tiny synthetic annotations (ADR-0044)."""

import pytest

from mmxai.data.scene_graph import (
    ADDED_FROM_RELATIONSHIP,
    DROP_DUPLICATE_TRIPLE,
    DROP_EMPTY_NAME,
    DROP_EMPTY_PREDICATE,
    DROP_MISSING_IMAGE_ID,
    DROP_MISSING_RELATIONSHIP_ID,
    DROP_SELF_LOOP,
    build_scene_graphs,
    load_scene_graphs,
    normalize_text,
)
from scene_graph_fixtures import synthetic_annotations, write_annotations


def test_normalize_text_lowercases_collapses_spaces_and_trims_punctuation():
    assert normalize_text("  Looks_At.  ") == "looks at"
    assert normalize_text("next   to") == "next to"


def test_relation_keeps_normalised_and_raw_strings():
    graphs, _ = build_scene_graphs(*synthetic_annotations(), image_ids={100})
    relation = graphs[100].relation_by_id(1)

    assert (relation.subject_name, relation.predicate, relation.object_name) == (
        "person",
        "holding",
        "frisbee",
    )
    assert (relation.raw_subject_name, relation.raw_predicate) == ("Person", "HOLDING")


def test_objects_with_the_same_name_stay_separate_nodes():
    graphs, _ = build_scene_graphs(*synthetic_annotations(), image_ids={100})

    assert 4 in graphs[100].objects and 5 in graphs[100].objects
    assert graphs[100].name_counts()["tree"] == 2


def test_image_ids_restrict_the_graphs_to_one_split():
    graphs, _ = build_scene_graphs(*synthetic_annotations(), image_ids={100, 200})

    assert sorted(graphs) == [100, 200]


def test_endpoint_missing_from_objects_json_is_added_from_the_relation():
    objects = [{"image_id": 1, "objects": [{"object_id": 1, "names": ["man"]}]}]
    relationships = [
        {
            "image_id": 1,
            "relationships": [
                {
                    "relationship_id": 9,
                    "predicate": "riding",
                    "subject": {"object_id": 1, "name": "man"},
                    "object": {"object_id": 2, "names": ["horse"]},
                }
            ],
        }
    ]
    graphs, report = build_scene_graphs(objects, relationships, image_ids=None)

    assert graphs[1].objects[2].name == "horse"
    assert report[ADDED_FROM_RELATIONSHIP] == 1


def test_malformed_records_are_dropped_and_counted_by_reason():
    man = {"object_id": 1, "names": ["man"]}
    hat = {"object_id": 2, "names": ["hat"]}
    relations = [
        {"predicate": "wearing", "subject": man, "object": hat},  # no relationship_id
        {"relationship_id": 1, "predicate": "  ", "subject": man, "object": hat},
        {"relationship_id": 2, "predicate": "near", "subject": man, "object": man},
        {"relationship_id": 3, "predicate": "wearing", "subject": man, "object": hat},
        {"relationship_id": 4, "predicate": "Wearing", "subject": man, "object": hat},
        {"relationship_id": 5, "predicate": "on", "subject": man, "object": {"object_id": 3}},
    ]
    objects = [{"image_id": 1, "objects": [man, hat]}, {"objects": []}]
    graphs, report = build_scene_graphs(
        objects, [{"image_id": 1, "relationships": relations}], image_ids=None
    )

    assert [relation.relationship_id for relation in graphs[1].relations] == [3]
    assert report[DROP_MISSING_RELATIONSHIP_ID] == 1
    assert report[DROP_EMPTY_PREDICATE] == 1
    assert report[DROP_SELF_LOOP] == 1
    assert report[DROP_DUPLICATE_TRIPLE] == 1
    assert report[DROP_EMPTY_NAME] == 1
    assert report[DROP_MISSING_IMAGE_ID] == 1


def test_load_scene_graphs_reads_both_files(tmp_path):
    write_annotations(tmp_path)

    graphs, _ = load_scene_graphs(tmp_path, image_ids={300})

    assert len(graphs[300].relations) == 2


def test_load_scene_graphs_explains_missing_annotations(tmp_path):
    with pytest.raises(FileNotFoundError, match="docs/query_benchmark.md"):
        load_scene_graphs(tmp_path, image_ids=None)
