"""Checks on the committed MMKG tables in mmkg/ (skipped before the first build)."""

from pathlib import Path

import pytest

from mmxai.data.mmkg_build import license_is_allowed, load_mmkg_build_config
from mmxai.data.mmkg_csv import read_rows

REPO_ROOT = Path(__file__).resolve().parents[1]
MMKG_DIR = REPO_ROOT / "mmkg"
BUILD_CONFIG = REPO_ROOT / "configs" / "mmkg_vi_build.yaml"
# The repo refuses committed files of 1 MiB or more.
MAX_FILE_BYTES = 1048576

pytestmark = pytest.mark.skipif(
    not (MMKG_DIR / "entities.csv").exists(), reason="the MMKG has not been built yet"
)


def test_every_committed_file_is_below_one_mib():
    too_big = []
    for path in MMKG_DIR.rglob("*"):
        # Downloaded images are gitignored, so their size does not matter here.
        if not path.is_file() or (MMKG_DIR / "images") in path.parents:
            continue
        if path.stat().st_size >= MAX_FILE_BYTES:
            too_big.append(str(path.relative_to(REPO_ROOT)))
    assert too_big == []


def test_triples_and_images_point_at_known_entities():
    entity_ids = {row["entity_id"] for row in read_rows(MMKG_DIR, "entities")}
    for triple in read_rows(MMKG_DIR, "triples"):
        assert triple["head_id"] in entity_ids, triple
        assert triple["tail_id"] in entity_ids, triple
    for image in read_rows(MMKG_DIR, "images"):
        assert image["entity_id"] in entity_ids, image


def test_every_item_has_three_to_five_images():
    config = load_mmkg_build_config(BUILD_CONFIG)
    image_count = {}
    for image in read_rows(MMKG_DIR, "images"):
        image_count[image["entity_id"]] = image_count.get(image["entity_id"], 0) + 1
    for entity in read_rows(MMKG_DIR, "entities"):
        if entity["type"] != "item":
            continue
        count = image_count.get(entity["entity_id"], 0)
        assert config.min_images_per_item <= count <= config.max_images_per_item, entity


def test_every_image_has_an_allowed_license_and_credit():
    config = load_mmkg_build_config(BUILD_CONFIG)
    for image in read_rows(MMKG_DIR, "images"):
        assert license_is_allowed(image["license"], config), image
        assert image["author"].strip() != "", image
        assert image["source_url"].strip() != "", image
