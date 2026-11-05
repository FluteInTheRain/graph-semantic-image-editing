"""Tests for the Flickr30k loader, using a tiny fake caption file."""

import json
from pathlib import Path

import pytest

from mmxai.data.flickr30k import load_captions, load_image_records


def write_caption_file(data_dir: Path, filenames: list[str]) -> None:
    images = []
    for imgid, filename in enumerate(filenames):
        sentences = [{"raw": f"{filename} caption {index}"} for index in range(5)]
        images.append(
            {"filename": filename, "imgid": imgid, "split": "train", "sentences": sentences}
        )
    (data_dir / "dataset_flickr30k.json").write_text(json.dumps({"images": images}))


def test_image_id_comes_from_filename(tmp_path):
    write_caption_file(tmp_path, ["1000092795.jpg"])
    records = load_image_records(tmp_path)
    assert records[0].image_id == 1000092795
    assert records[0].path == tmp_path / "images" / "1000092795.jpg"


def test_records_sorted_by_image_id(tmp_path):
    write_caption_file(tmp_path, ["30.jpg", "4.jpg", "200.jpg"])
    records = load_image_records(tmp_path)
    assert [record.image_id for record in records] == [4, 30, 200]


def test_extra_images_in_folder_are_ignored(tmp_path):
    write_caption_file(tmp_path, ["1.jpg"])
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "1.jpg").write_bytes(b"x")
    (tmp_path / "images" / "2.jpg").write_bytes(b"x")
    assert [record.image_id for record in load_image_records(tmp_path)] == [1]


def test_caption_index_runs_zero_to_four(tmp_path):
    write_caption_file(tmp_path, ["1.jpg"])
    captions = load_captions(tmp_path)
    assert [caption.text_id for caption in captions] == [0, 1, 2, 3, 4]
    assert captions[2].text == "1.jpg caption 2"
    assert all(caption.image_id == 1 for caption in captions)


def test_captions_sorted_by_image_id(tmp_path):
    write_caption_file(tmp_path, ["9.jpg", "2.jpg"])
    captions = load_captions(tmp_path)
    assert [caption.image_id for caption in captions] == [2] * 5 + [9] * 5


def test_non_numeric_filename_raises(tmp_path):
    write_caption_file(tmp_path, ["photo.jpg"])
    with pytest.raises(ValueError, match="photo.jpg"):
        load_image_records(tmp_path)


def test_images_dir_override(tmp_path):
    write_caption_file(tmp_path, ["1.jpg"])
    records = load_image_records(tmp_path, images_dir=Path("/kaggle/input/flickr"))
    assert records[0].path == Path("/kaggle/input/flickr/1.jpg")
