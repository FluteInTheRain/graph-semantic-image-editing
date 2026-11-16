import csv
import json
from pathlib import Path

import pytest

from mmxai.data.mmkg_build import (
    MmkgBuildConfig,
    Review,
    build_mmkg,
    load_mmkg_build_config,
    read_review,
    write_built_mmkg,
)
from mmxai.data.mmkg_csv import write_rows

ENTITY_HEADER = ["entity_id", "name_vi", "name_en", "in_domain", "commons_category", "p18_files"]
TRIPLE_HEADER = ["head_id", "property", "tail_id"]
CANDIDATE_HEADER = [
    "file_name",
    "entity_id",
    "origin",
    "mime",
    "width",
    "height",
    "license",
    "sha1",
]
CREDIT_HEADER = ["file_name", "artist", "uploader"]
REVIEW_HEADER = ["kind", "key", "review_status", "reason", "fix_name_vi", "reviewer", "date"]

RELATIONS = {"P131": "located_in", "P276": "kept_at", "P195": "kept_at"}


def make_config(min_items_per_concept: int = 3) -> MmkgBuildConfig:
    return MmkgBuildConfig(
        snapshot_dir="snap",
        review_file="review.csv",
        output_dir="out",
        allowed_mime_types=["image/jpeg"],
        allowed_license_names=["CC0"],
        allowed_license_prefixes=[],
        attribution_license_prefixes=[],
        min_short_side_px=224,
        min_images_per_item=3,
        max_images_per_item=5,
        min_items_per_concept=min_items_per_concept,
        relations=RELATIONS,
        max_file_bytes=100000,
    )


def entity(entity_id: str, name_vi: str = "", in_domain: str = "true") -> dict[str, str]:
    return {
        "entity_id": entity_id,
        "name_vi": name_vi or f"vi {entity_id}",
        "name_en": f"en {entity_id}",
        "in_domain": in_domain,
        "commons_category": "",
        "p18_files": "",
    }


def images_of(entity_id: str, count: int = 3) -> list[dict[str, str]]:
    return [
        {
            "file_name": f"{entity_id}_{i}.jpg",
            "entity_id": entity_id,
            "origin": "category",
            "mime": "image/jpeg",
            "width": "800",
            "height": "600",
            "license": "CC0",
            "sha1": f"sha{entity_id}{i}".ljust(12, "0"),
        }
        for i in range(count)
    ]


def write_snapshot(
    directory: Path,
    entities: list[dict[str, str]],
    triples: list[tuple[str, str, str]],
    image_counts: dict[str, int],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, str]] = []
    for entity_id, count in image_counts.items():
        candidates.extend(images_of(entity_id, count))
    credits = [{"file_name": c["file_name"], "artist": "A", "uploader": ""} for c in candidates]
    triple_rows = [{"head_id": h, "property": p, "tail_id": t} for h, p, t in triples]
    write_rows(directory, "entities_raw", ENTITY_HEADER, entities, 100000)
    write_rows(directory, "triples_raw", TRIPLE_HEADER, triple_rows, 100000)
    write_rows(directory, "image_candidates", CANDIDATE_HEADER, candidates, 100000)
    write_rows(directory, "image_credits", CREDIT_HEADER, credits, 100000)


def write_review_file(path: Path, rows: list[dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=REVIEW_HEADER)
        writer.writeheader()
        writer.writerows(rows)


def review_row(kind: str, key: str, status: str, fix: str = "") -> dict[str, str]:
    return {
        "kind": kind,
        "key": key,
        "review_status": status,
        "reason": "",
        "fix_name_vi": fix,
        "reviewer": "",
        "date": "",
    }


def three_items_snapshot(directory: Path, shared_by: int = 3) -> None:
    """Items Q1..Q3 with images; Q9 (no images) is the tail of `shared_by` of them."""
    items = ["Q1", "Q2", "Q3"]
    entities = [entity(i) for i in items] + [entity("Q9", in_domain="false")]
    triples = [(i, "P131", "Q9") for i in items[:shared_by]]
    write_snapshot(directory, entities, triples, {i: 3 for i in items})


def build(directory: Path, review=None, config=None):
    return build_mmkg(directory, review or Review({}, {}), config or make_config())


def ids(built) -> dict[str, str]:
    return {row["entity_id"]: row["type"] for row in built.entities}


def test_concept_needs_three_items(tmp_path):
    three_items_snapshot(tmp_path / "a", shared_by=3)
    three_items_snapshot(tmp_path / "b", shared_by=2)

    kept = build(tmp_path / "a")
    dropped = build(tmp_path / "b")

    assert ids(kept)["Q9"] == "concept"
    assert len(kept.triples) == 3
    assert "Q9" not in ids(dropped)
    assert dropped.triples == []
    assert dropped.report["triples_dropped_end_not_kept"] == 2


def test_location_and_collection_merge_to_kept_at(tmp_path):
    entities = [entity("Q1"), entity("M", in_domain="false")]
    write_snapshot(tmp_path, entities, [("Q1", "P276", "M"), ("Q1", "P195", "M")], {"Q1": 3})

    built = build(tmp_path, config=make_config(min_items_per_concept=1))

    assert len(built.triples) == 1
    assert built.triples[0]["relation"] == "kept_at"
    assert built.triples[0]["wikidata_property"] == "P195|P276"
    assert built.triples[0]["source"] == "https://www.wikidata.org/wiki/Q1"
    pending_keys = {row["key"] for row in built.pending if row["kind"] == "triple"}
    assert pending_keys == {"Q1|P195|M", "Q1|P276|M"}


def test_domain_entity_with_few_images_can_be_concept(tmp_path):
    items = ["Q1", "Q2", "Q3"]
    entities = [entity(i) for i in items] + [entity("MUSEUM")]
    triples = [(i, "P276", "MUSEUM") for i in items]
    counts = {"Q1": 3, "Q2": 3, "Q3": 3, "MUSEUM": 1}
    write_snapshot(tmp_path, entities, triples, counts)

    built = build(tmp_path)

    assert ids(built)["MUSEUM"] == "concept"
    assert all(row["entity_id"] != "MUSEUM" for row in built.images)


def test_concept_to_concept_triple_kept_when_both_kept(tmp_path):
    items = ["Q1", "Q2", "Q3"]
    entities = [entity(i) for i in items] + [entity("C1", in_domain="false")]
    entities.append(entity("C2", in_domain="false"))
    triples = [(i, "P131", "C1") for i in items] + [(i, "P276", "C2") for i in items]
    triples.append(("C1", "P131", "C2"))
    write_snapshot(tmp_path, entities, triples, {i: 3 for i in items})

    built = build(tmp_path)

    assert ("C1", "located_in", "C2") in {
        (t["head_id"], t["relation"], t["tail_id"]) for t in built.triples
    }


def test_fix_name_vi_overrides_label(tmp_path):
    three_items_snapshot(tmp_path)
    review = Review({("entity", "Q1"): "ok"}, {"Q1": "Tên đã sửa"})

    built = build(tmp_path, review)

    names = {row["entity_id"]: row["name_vi"] for row in built.entities}
    assert names["Q1"] == "Tên đã sửa"
    assert names["Q2"] == "vi Q2"


def test_missing_name_vi_is_counted(tmp_path):
    entities = [entity("Q1"), entity("Q2"), entity("Q3")]
    entities[0]["name_vi"] = ""
    # entity() fills an empty name, so blank it after construction.
    write_snapshot(tmp_path, entities, [], {"Q1": 3, "Q2": 3, "Q3": 3})

    built = build(tmp_path)

    assert built.report["entities_missing_name_vi"] == 1


def test_entity_drop_removes_its_triples(tmp_path):
    three_items_snapshot(tmp_path)
    review = Review({("entity", "Q9"): "drop"}, {})

    built = build(tmp_path, review)

    assert "Q9" not in ids(built)
    assert built.triples == []
    assert built.report["entities_dropped_review"] == 1


def test_entity_drop_removes_item_and_its_images(tmp_path):
    three_items_snapshot(tmp_path)
    review = Review({("entity", "Q1"): "drop"}, {})

    built = build(tmp_path, review)

    assert "Q1" not in ids(built)
    assert all(row["entity_id"] != "Q1" for row in built.images)
    assert built.report["images_dropped_entity_review_drop"] == 3
    assert all(row["head_id"] != "Q1" for row in built.triples)


def test_triple_review_drop_and_unknown_relation_are_counted(tmp_path):
    entities = [entity("Q1"), entity("M", in_domain="false")]
    triples = [("Q1", "P276", "M"), ("Q1", "P999", "M")]
    write_snapshot(tmp_path, entities, triples, {"Q1": 3})
    review = Review({("triple", "Q1|P276|M"): "drop"}, {})

    built = build(tmp_path, review, make_config(min_items_per_concept=1))

    assert built.triples == []
    assert built.report["triples_dropped_review"] == 1
    assert built.report["triples_dropped_unknown_relation"] == 1


def test_every_reference_exists(tmp_path):
    three_items_snapshot(tmp_path)

    built = build(tmp_path)

    entity_ids = {row["entity_id"] for row in built.entities}
    for triple in built.triples:
        assert triple["head_id"] in entity_ids
        assert triple["tail_id"] in entity_ids
        assert triple["head_id"] != triple["tail_id"]
    for image in built.images:
        assert image["entity_id"] in entity_ids
    for item_id, kind in ids(built).items():
        if kind == "item":
            count = sum(1 for image in built.images if image["entity_id"] == item_id)
            assert 3 <= count <= 5


def test_tables_are_sorted(tmp_path):
    three_items_snapshot(tmp_path)

    built = build(tmp_path)

    assert [r["entity_id"] for r in built.entities] == sorted(
        r["entity_id"] for r in built.entities
    )
    triple_keys = [(t["head_id"], t["relation"], t["tail_id"]) for t in built.triples]
    assert triple_keys == sorted(triple_keys)
    image_entities = [r["entity_id"] for r in built.images]
    assert image_entities == sorted(image_entities)


def test_build_is_deterministic(tmp_path):
    three_items_snapshot(tmp_path)

    first = build(tmp_path)
    second = build(tmp_path)

    assert first == second


def test_pending_rows_are_appended_not_rewritten(tmp_path):
    snapshot = tmp_path / "snap"
    three_items_snapshot(snapshot)
    review_file = tmp_path / "review.csv"
    ok_row = review_row("entity", "Q1", "ok")
    write_review_file(review_file, [ok_row])
    before = review_file.read_bytes()
    review = read_review(review_file)

    # Q9 is the tail of 3 items, so with a threshold of 4 it is not official.
    built = build(snapshot, review, make_config(min_items_per_concept=4))
    write_built_mmkg(built, tmp_path / "out", review_file, 100000)

    after = review_file.read_bytes()
    assert after.startswith(before)
    appended = list(csv.DictReader(after[len(before) :].decode().splitlines(), REVIEW_HEADER))
    assert appended
    assert all(row["review_status"] == "pending" for row in appended)
    assert ("entity", "Q1") not in {(r["kind"], r["key"]) for r in appended}
    assert {r["kind"] for r in appended} == {"image", "entity"}
    assert "Q9" not in {r["key"] for r in appended}
    # A second build with the new review has nothing more to append.
    again = build(snapshot, read_review(review_file), make_config(min_items_per_concept=4))
    assert again.pending == []
    assert again.report["pending_entity"] == 2


def test_review_file_is_created_with_header(tmp_path):
    three_items_snapshot(tmp_path / "snap")
    review_file = tmp_path / "review.csv"

    built = build(tmp_path / "snap")
    write_built_mmkg(built, tmp_path / "out", review_file, 100000)

    lines = review_file.read_text(encoding="utf-8").splitlines()
    assert lines[0] == ",".join(REVIEW_HEADER)
    assert len(lines) == 1 + len(built.pending)


def test_write_outputs_tables_and_report(tmp_path):
    three_items_snapshot(tmp_path / "snap")
    built = build(tmp_path / "snap")
    out = tmp_path / "out"

    write_built_mmkg(built, out, tmp_path / "review.csv", 100000)

    assert (out / "entities.csv").exists()
    assert (out / "triples.csv").exists()
    assert (out / "images.csv").exists()
    report = json.loads((out / "build_report.json").read_text(encoding="utf-8"))
    assert report == built.report


def test_report_counts_raw_and_official(tmp_path):
    three_items_snapshot(tmp_path)

    report = build(tmp_path).report

    assert report["raw_entities"] == 4
    assert report["raw_triples"] == 3
    assert report["raw_image_candidates"] == 9
    assert report["entities_item"] == 3
    assert report["entities_concept"] == 1
    assert report["triples"] == 3
    assert report["images"] == 9
    assert report["images_dropped_mime"] == 0
    assert report["triples_dropped_review"] == 0
    assert report["pending_image"] == 9
    assert report["pending_triple"] == 3
    assert report["pending_entity"] == 4


def test_placeholder_snapshot_dir_is_rejected(tmp_path):
    config_file = tmp_path / "build.yaml"
    config_file.write_text(
        "snapshot_dir: mmkg/raw/YYYYMMDD\n"
        "review_file: mmkg/review.csv\n"
        "output_dir: mmkg\n"
        "allowed_mime_types: [image/jpeg]\n"
        "allowed_license_names: [CC0]\n"
        "allowed_license_prefixes: ['CC BY ']\n"
        "attribution_license_prefixes: ['CC BY ']\n"
        "min_short_side_px: 224\n"
        "min_images_per_item: 3\n"
        "max_images_per_item: 5\n"
        "min_items_per_concept: 3\n"
        "max_file_bytes: 1048576\n"
        "relations: {P131: located_in}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="placeholder"):
        load_mmkg_build_config(config_file)


def test_candidate_entity_missing_from_raw_entities_fails(tmp_path):
    write_snapshot(tmp_path, [entity("Q1")], [], {"Q1": 3, "Q7": 3})

    with pytest.raises(ValueError, match="Q7"):
        build(tmp_path)


def test_triple_end_missing_from_raw_entities_fails(tmp_path):
    write_snapshot(tmp_path, [entity("Q1")], [("Q1", "P131", "Q8")], {"Q1": 3})

    with pytest.raises(ValueError, match="Q8"):
        build(tmp_path)


def test_append_after_file_without_final_newline_keeps_human_row(tmp_path):
    snapshot = tmp_path / "snap"
    three_items_snapshot(snapshot)
    review_file = tmp_path / "review.csv"
    human = ",".join(REVIEW_HEADER) + "\r\nentity,Q1,ok,,,,2026-11-16"
    review_file.write_bytes(human.encode("utf-8"))

    built = build(snapshot, read_review(review_file))
    write_built_mmkg(built, tmp_path / "out", review_file, 100000)

    data = review_file.read_bytes()
    assert data.startswith(human.encode("utf-8") + b"\r\n")
    with open(review_file, encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    human_row = [r for r in rows if r["kind"] == "entity" and r["key"] == "Q1"]
    assert len(human_row) == 1
    assert human_row[0]["date"] == "2026-11-16"
    assert human_row[0]["review_status"] == "ok"
    assert len(rows) == 1 + len(built.pending)
    assert read_review(review_file).status[("entity", "Q1")] == "ok"
