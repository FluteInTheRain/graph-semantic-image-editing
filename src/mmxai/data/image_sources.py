"""Where the demo shows a Visual Genome image from: the local file, else the original url.

A gallery can show a url, so the 15 GB of images are optional for the demo; only
image_data.json (with the urls) is needed. Kept apart from visual_genome.py, whose code
is copied into the Kaggle feature notebook (ADR-0041) and does not need this.
"""

import json
from pathlib import Path

from mmxai.data.visual_genome import IMAGE_FILE, load_image_records


def image_sources(data_dir: Path, images_dir: Path | None = None) -> dict[int, str]:
    """image id -> local file path if it exists, else the url from image_data.json."""
    with open(Path(data_dir) / IMAGE_FILE, encoding="utf-8") as file:
        entries = json.load(file)
    url_of_id = {}
    for entry in entries:
        url_of_id[entry["image_id"]] = entry["url"]

    sources = {}
    for record in load_image_records(data_dir, images_dir):
        if record.path.exists():
            sources[record.image_id] = str(record.path)
        else:
            sources[record.image_id] = url_of_id[record.image_id]
    return sources
