"""Query validation, candidate relevance and the human-review workflow (ADR-0044).

All data is synthetic (tests/scene_graph_fixtures.py), not Visual Genome.
"""

import copy
import csv
from pathlib import Path

import pytest
import yaml

from mmxai.data.scene_graph import Relation, SceneGraph, SceneObject
from mmxai.queries.generator import generate_pairs, load_query_benchmark_config, run_generation
from mmxai.queries.schema import load_jsonl
from mmxai.queries.validator import (
    MANIFEST_COLUMNS,
    build_verified_benchmark,
    check_kg_evidence,
    images_matching_pattern,
    manifest_rows,
    pattern_from_record,
    read_review_manifest,
    run_build_benchmark,
    validate_queries,
    write_review_manifest,
)
from scene_graph_fixtures import (
    CONFIG,
    TEST_IDS,
    VAL_IDS,
    canonical_fixture_graphs,
    write_annotations,
    write_split,
)

TEST_ID_STRINGS = {str(image_id) for image_id in TEST_IDS}


def graphs_and_candidates() -> tuple[dict, list[dict]]:
    graphs = canonical_fixture_graphs(TEST_IDS)
    records, _ = generate_pairs(graphs, CONFIG)
    return graphs, records


def verified_row(row: dict, reviewer: str = "an", checker: str = "binh") -> dict:
    row = dict(row)
    row["verification_status"] = "verified"
    row["reviewer"] = reviewer
    row["checker"] = checker
    row["verified_relevant"] = "100 200"
    return row


def test_pattern_from_record_finds_the_same_answers_as_the_generator():
    graphs, records = graphs_and_candidates()
    names, steps = pattern_from_record(records[0])

    assert names == ["person", "frisbee", "dog"]
    assert images_matching_pattern(names, steps, graphs) == [100, 200]


def test_generated_candidates_are_valid_against_the_annotations():
    graphs, records = graphs_and_candidates()

    assert validate_queries(records, TEST_ID_STRINGS, graphs) == []


def test_duplicate_qid_is_reported():
    _, records = graphs_and_candidates()
    records.append(copy.deepcopy(records[0]))

    assert any("duplicate qid" in problem for problem in validate_queries(records, TEST_ID_STRINGS))


def test_relevant_image_outside_test_is_reported():
    _, records = graphs_and_candidates()
    records[0]["relevant"] = records[0]["relevant"] + [str(VAL_IDS[0])]

    problems = validate_queries(records, TEST_ID_STRINGS)

    assert any("is not in test" in problem for problem in problems)


def test_broken_pair_is_reported():
    _, records = graphs_and_candidates()

    problems = validate_queries(records[:1], TEST_ID_STRINGS)

    assert any("expected one direct + one indirect" in problem for problem in problems)


def test_evidence_that_is_not_annotated_is_reported():
    graphs, records = graphs_and_candidates()
    records[1]["relation_evidence"][0]["predicate"] = "throwing"

    problems = validate_queries(records, TEST_ID_STRINGS, graphs)

    assert any("differs" in problem for problem in problems)


def test_indirect_text_that_names_the_hidden_object_is_reported():
    _, records = graphs_and_candidates()
    indirect = next(record for record in records if record["type"] == "indirect")
    indirect["text"] = "A person is holding the frisbee that a dog looks at."

    problems = validate_queries(records, TEST_ID_STRINGS)

    assert any("names hidden objects" in problem for problem in problems)


def test_candidates_fail_when_verification_is_required():
    _, records = graphs_and_candidates()

    problems = validate_queries(records, TEST_ID_STRINGS, require_verified=True)

    assert any("not verified" in problem for problem in problems)


def test_manifest_has_one_candidate_row_per_pair(tmp_path):
    _, records = graphs_and_candidates()
    path = tmp_path / "manifest.csv"

    write_review_manifest(records, path)
    rows = read_review_manifest(path)

    assert len(rows) == len(records) // 2
    assert list(rows[0]) == MANIFEST_COLUMNS
    assert rows[0]["verification_status"] == "candidate"
    assert rows[0]["candidate_relevant"] == "100 200"


def test_unreviewed_rows_never_enter_the_benchmark():
    _, records = graphs_and_candidates()
    rows = manifest_rows(records)
    rows[0]["verification_status"] = "reviewed"
    rows[0]["reviewer"] = "an"
    rows[0]["verified_relevant"] = "100"

    assert build_verified_benchmark(records, rows, TEST_ID_STRINGS) == []


def test_verified_pair_uses_the_human_answers():
    _, records = graphs_and_candidates()
    row = verified_row(manifest_rows(records)[0])
    row["verified_relevant"] = "200"

    final = build_verified_benchmark(records, [row], TEST_ID_STRINGS)

    assert len(final) == 2
    assert {record["verification_status"] for record in final} == {"verified"}
    assert all(record["relevant"] == ["200"] for record in final)
    assert final[0]["checked_by"] == "binh"


def test_reviewer_cannot_check_their_own_review():
    _, records = graphs_and_candidates()
    row = verified_row(manifest_rows(records)[0], reviewer="an", checker="An")

    with pytest.raises(ValueError, match="different member"):
        build_verified_benchmark(records, [row], TEST_ID_STRINGS)


def test_single_reviewer_mode_accepts_an_empty_checker():
    _, records = graphs_and_candidates()
    row = verified_row(manifest_rows(records)[0], checker="")

    final = build_verified_benchmark(records, [row], TEST_ID_STRINGS, required_reviewers=1)

    assert len(final) == 2


def test_single_reviewer_mode_accepts_checker_equal_to_reviewer():
    _, records = graphs_and_candidates()
    row = verified_row(manifest_rows(records)[0], reviewer="an", checker="An")

    final = build_verified_benchmark(records, [row], TEST_ID_STRINGS, required_reviewers=1)

    assert len(final) == 2


def test_single_reviewer_mode_still_needs_a_reviewer():
    _, records = graphs_and_candidates()
    row = verified_row(manifest_rows(records)[0], reviewer="", checker="binh")

    with pytest.raises(ValueError, match="needs a reviewer"):
        build_verified_benchmark(records, [row], TEST_ID_STRINGS, required_reviewers=1)


def test_two_reviewer_mode_rejects_an_empty_checker():
    _, records = graphs_and_candidates()
    row = verified_row(manifest_rows(records)[0], checker="")

    with pytest.raises(ValueError, match="both reviewer and checker"):
        build_verified_benchmark(records, [row], TEST_ID_STRINGS)


def test_required_reviewers_must_be_one_or_two():
    _, records = graphs_and_candidates()
    row = verified_row(manifest_rows(records)[0])

    with pytest.raises(ValueError, match="required_reviewers"):
        build_verified_benchmark(records, [row], TEST_ID_STRINGS, required_reviewers=3)


def test_verified_answer_outside_test_is_refused():
    _, records = graphs_and_candidates()
    row = verified_row(manifest_rows(records)[0])
    row["verified_relevant"] = "100 400"

    with pytest.raises(ValueError, match="not in the test split"):
        build_verified_benchmark(records, [row], TEST_ID_STRINGS)


def test_edited_indirect_text_may_not_reveal_the_hidden_object():
    _, records = graphs_and_candidates()
    row = verified_row(manifest_rows(records)[0])
    row["edited_indirect_text"] = "A person holds a frisbee that a dog looks at."

    with pytest.raises(ValueError, match="names hidden objects"):
        build_verified_benchmark(records, [row], TEST_ID_STRINGS)


def write_config(tmp_path: Path, split: str) -> Path:
    data_dir = tmp_path / "data"
    write_annotations(data_dir)
    split_file = tmp_path / "split.json"
    write_split(split_file)
    values = {field: getattr(CONFIG, field) for field in CONFIG.__dataclass_fields__}
    values.update(
        {
            "data_dir": str(data_dir),
            "split_file": str(split_file),
            "split": split,
            "candidates_path": str(tmp_path / "out" / "candidates.jsonl"),
            "manifest_path": str(tmp_path / "out" / "manifest.csv"),
            "benchmark_path": str(tmp_path / "out" / "benchmark.jsonl"),
        }
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    return path


def test_run_generation_writes_candidates_and_manifest_of_one_split(tmp_path):
    config = load_query_benchmark_config(write_config(tmp_path, "test"))

    summary = run_generation(config)
    records = load_jsonl(config.candidates_path)

    assert summary["num_pairs"] == 1
    assert {record["split"] for record in records} == {"test"}
    for record in records:
        assert set(record["relevant"]) <= TEST_ID_STRINGS
    assert Path(config.manifest_path).exists()


def test_val_pilot_only_uses_val_images(tmp_path):
    config = load_query_benchmark_config(write_config(tmp_path, "val"))

    run_generation(config)
    records = load_jsonl(config.candidates_path)

    assert {record["source_image_id"] for record in records} == {str(VAL_IDS[0])}
    assert {tuple(record["relevant"]) for record in records} == {(str(VAL_IDS[0]),)}


def test_run_generation_refuses_to_overwrite(tmp_path):
    config = load_query_benchmark_config(write_config(tmp_path, "test"))
    run_generation(config)

    with pytest.raises(FileExistsError):
        run_generation(config)


def test_build_refuses_too_few_verified_pairs(tmp_path):
    config = load_query_benchmark_config(write_config(tmp_path, "test"))
    run_generation(config)

    # Nobody reviewed anything: zero verified pairs, so no benchmark file is written.
    with pytest.raises(ValueError, match="only 0 verified pairs"):
        run_build_benchmark(
            config.candidates_path,
            config.manifest_path,
            config.split_file,
            config.benchmark_path,
            1,
        )
    assert not Path(config.benchmark_path).exists()


def test_build_writes_the_verified_benchmark(tmp_path):
    config = load_query_benchmark_config(write_config(tmp_path, "test"))
    run_generation(config)
    rows = read_review_manifest(config.manifest_path)
    with open(config.manifest_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerow(verified_row(rows[0]))

    num_pairs = run_build_benchmark(
        config.candidates_path, config.manifest_path, config.split_file, config.benchmark_path, 1
    )

    assert num_pairs == 1
    final = load_jsonl(config.benchmark_path)
    assert validate_queries(final, TEST_ID_STRINGS, require_verified=True) == []


def synset_graph(image_id: int, rider: SceneObject) -> SceneGraph:
    horse = SceneObject(2, "horse", "horse", "horse.n.01")
    graph = SceneGraph(image_id, {1: rider, 2: horse})
    graph.relations.append(
        Relation(9, 1, rider.name, "riding", 2, "horse", "riding", rider.name, "horse")
    )
    return graph


def test_answers_match_by_synset_not_only_by_name():
    graphs = {
        1: synset_graph(1, SceneObject(1, "man", "man", "man.n.01")),
        2: synset_graph(2, SceneObject(1, "guy", "guy", "man.n.01")),
        3: synset_graph(3, SceneObject(1, "man", "man", "")),  # no synset: key "man"
    }

    assert images_matching_pattern(["man.n.01", "horse.n.01"], [("riding", True)], graphs) == [
        1,
        2,
    ]


def test_pattern_from_record_reads_synset_keys_from_evidence():
    record = {
        "qid": "q",
        "chain_object_ids": [1, 2],
        "relation_evidence": [
            {
                "relationship_id": 9,
                "subject_id": 1,
                "subject_name": "guy",
                "subject_synset": "man.n.01",
                "predicate": "riding",
                "object_id": 2,
                "object_name": "horse",
                "object_synset": "",
            }
        ],
    }

    assert pattern_from_record(record) == (["man.n.01", "horse"], [("riding", True)])


def kg_pair() -> list[dict]:
    shared = {
        "dataset": "visual_genome",
        "relevant": ["100"],
        "path_hint": "racket - UsedFor - tennis",
        "author": "test",
        "pair_id": "vgkg-test-a001",
        "split": "test",
        "verification_status": "candidate",
        "hidden_concept": "tennis.n.01",
        "query_group": "activity",
        "kg_evidence": [
            {"uri": "/a/x", "rel": "UsedFor", "start": "racket", "end": "tennis", "weight": 2.0}
        ],
    }
    direct = {"qid": "vgkg-test-a001-d", "text": "A person playing tennis.", "type": "direct"}
    direct.update(shared, reasoning_type="explicit", hidden_names=[])
    indirect = {
        "qid": "vgkg-test-a001-i",
        "text": "A person playing a sport that uses a racket.",
        "type": "indirect",
    }
    indirect.update(shared, reasoning_type="kg_description", hidden_names=["tennis"])
    return [direct, indirect]


def test_manifest_rows_accept_kg_records():
    rows = manifest_rows(kg_pair())

    assert rows[0]["query_group"] == "activity"
    assert rows[0]["hidden_concept"] == "tennis.n.01"
    assert rows[0]["kg_edges"] == "/a/x"
    assert rows[0]["relationship_ids"] == ""
    assert list(rows[0]) == MANIFEST_COLUMNS


def test_kg_pair_passes_the_generic_checks():
    assert validate_queries(kg_pair(), {"100"}) == []


def test_check_kg_evidence_rejects_unknown_edges():
    assert check_kg_evidence(kg_pair(), {"/a/x"}) == []
    problems = check_kg_evidence(kg_pair(), {"/a/other"})
    assert len(problems) == 2
    assert "/a/x" in problems[0]
