"""Every committed query file in queries/ is valid and points to test images only (ADR-0009).

Official benchmark files (*_test.jsonl) must hold verified queries only (ADR-0044);
other files (candidates) only need a valid schema and test-split answers.
"""

import json
from pathlib import Path

import pytest

from mmxai.queries.schema import load_jsonl
from mmxai.queries.validator import validate_queries

REPO_ROOT = Path(__file__).resolve().parents[1]
QUERY_FILES = sorted((REPO_ROOT / "queries").glob("*.jsonl"))
SPLIT_FILE = REPO_ROOT / "splits" / "visual_genome" / "split_seed0.json"


def load_test_ids() -> set[str]:
    with open(SPLIT_FILE, encoding="utf-8") as file:
        return set(json.load(file)["test"])


@pytest.mark.parametrize("path", QUERY_FILES, ids=[path.name for path in QUERY_FILES])
def test_committed_query_file_is_valid(path):
    records = load_jsonl(path)
    require_verified = path.name.endswith("_test.jsonl")

    problems = validate_queries(records, load_test_ids(), require_verified=require_verified)

    assert problems == []
