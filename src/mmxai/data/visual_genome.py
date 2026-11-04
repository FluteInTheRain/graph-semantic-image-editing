"""Read Visual Genome v1.2 images and region descriptions, and sample regions per image."""

import json
from pathlib import Path

import numpy as np

from mmxai.data.records import ImageRecord, TextRecord

IMAGE_FILE = "image_data.json"
REGION_FILE = "region_descriptions.json"


def load_image_records(data_dir: Path, images_dir: Path | None = None) -> list[ImageRecord]:
    """List all images, sorted by id. The url tells which folder holds each image."""
    if images_dir is None:
        images_dir = Path(data_dir) / "images"

    with open(Path(data_dir) / IMAGE_FILE, encoding="utf-8") as file:
        entries = json.load(file)

    records = []
    for entry in entries:
        url_parts = entry["url"].split("/")
        folder = url_parts[-2]
        filename = url_parts[-1]
        records.append(ImageRecord(entry["image_id"], Path(images_dir) / folder / filename))
    return sorted(records, key=lambda record: record.image_id)


def load_region_texts(data_dir: Path) -> list[TextRecord]:
    """List all regions, sorted by image then region id.

    Blank phrases are dropped: the embedding of an empty string is not a query.
    """
    with open(Path(data_dir) / REGION_FILE, encoding="utf-8") as file:
        entries = json.load(file)

    regions = []
    for entry in entries:
        for region in entry["regions"]:
            phrase = region["phrase"].strip()
            if phrase:
                regions.append(TextRecord(region["image_id"], region["region_id"], phrase))
    return sorted(regions, key=lambda region: (region.image_id, region.text_id))


def sample_regions(
    regions: list[TextRecord], regions_per_image: int, seed: int
) -> list[TextRecord]:
    """Pick up to regions_per_image regions per image at random, reproducibly.

    The choice ignores content and happens before any retrieval result exists, so
    it cannot favour a model. Fixed ordering plus one generator makes it repeatable.
    """
    regions_by_image: dict[int, list[TextRecord]] = {}
    for region in sorted(regions, key=lambda region: (region.image_id, region.text_id)):
        regions_by_image.setdefault(region.image_id, []).append(region)

    rng = np.random.default_rng(seed)
    sampled = []
    for image_id in sorted(regions_by_image):
        image_regions = regions_by_image[image_id]
        if len(image_regions) <= regions_per_image:
            sampled.extend(image_regions)
            continue
        chosen = rng.choice(len(image_regions), size=regions_per_image, replace=False)
        # Sorting keeps rows in the documented order: by image, then by region id.
        for position in sorted(chosen.tolist()):
            sampled.append(image_regions[position])
    return sampled
