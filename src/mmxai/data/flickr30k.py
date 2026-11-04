"""Read Flickr30k images and captions from Karpathy's dataset_flickr30k.json."""

import json
from pathlib import Path

from mmxai.data.records import ImageRecord, TextRecord

CAPTION_FILE = "dataset_flickr30k.json"


def read_image_entries(data_dir: Path) -> list[dict]:
    """Return the caption file's image entries, sorted by image id."""
    with open(Path(data_dir) / CAPTION_FILE, encoding="utf-8") as file:
        entries = json.load(file)["images"]
    return sorted(entries, key=lambda entry: image_id_from_filename(entry["filename"]))


def image_id_from_filename(filename: str) -> int:
    """Use the number in the filename: unlike imgid, it also matches other Flickr30k sources."""
    stem = Path(filename).stem
    if not stem.isdigit():
        raise ValueError(f"expected a numeric Flickr30k filename, got {filename}")
    return int(stem)


def load_image_records(data_dir: Path, images_dir: Path | None = None) -> list[ImageRecord]:
    """List images named in the caption file. Other files in the folder are ignored."""
    if images_dir is None:
        images_dir = Path(data_dir) / "images"

    records = []
    for entry in read_image_entries(data_dir):
        image_id = image_id_from_filename(entry["filename"])
        records.append(ImageRecord(image_id, Path(images_dir) / entry["filename"]))
    return records


def load_captions(data_dir: Path) -> list[TextRecord]:
    """List every caption, sorted by image then caption position.

    The file's "split" field is deliberately not read: official splits live in splits/.
    """
    captions = []
    for entry in read_image_entries(data_dir):
        image_id = image_id_from_filename(entry["filename"])
        for caption_index, sentence in enumerate(entry["sentences"]):
            captions.append(TextRecord(image_id, caption_index, sentence["raw"]))
    return captions
