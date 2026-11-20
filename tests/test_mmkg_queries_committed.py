"""The committed MMKG query file is exactly what the generator writes from mmkg/ (ADR-0053)."""

import json
from collections import Counter
from pathlib import Path

import pytest

from mmxai.data.mmkg_queries import build_queries, load_mmkg, query_record

REPO_ROOT = Path(__file__).resolve().parents[1]
QUERY_FILE = REPO_ROOT / "queries" / "mmkg_vi" / "queries.jsonl"

pytestmark = pytest.mark.skipif(not QUERY_FILE.exists(), reason="no MMKG query file yet")


def committed_records() -> list[dict]:
    with open(QUERY_FILE, encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def test_query_file_matches_the_generator():
    entities, triples, images_by_item = load_mmkg(REPO_ROOT / "mmkg")
    queries = build_queries(entities, triples, sorted(images_by_item))

    assert committed_records() == [query_record(query, images_by_item) for query in queries]


def test_no_two_items_share_a_name():
    entities, _, images_by_item = load_mmkg(REPO_ROOT / "mmkg")
    for lang in ("en", "vi"):
        names = Counter(entities[item].name(lang) for item in images_by_item)
        assert [name for name, count in names.items() if count > 1] == []


def test_every_query_has_an_answer_and_a_unique_id():
    records = committed_records()

    assert all(record["relevant"] for record in records)
    assert len({record["qid"] for record in records}) == len(records)
