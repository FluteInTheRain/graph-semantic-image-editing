"""Tests for the Visual Genome loader and region sampling, using tiny fake files."""

import json
from pathlib import Path

from mmxai.data.records import TextRecord
from mmxai.data.visual_genome import load_image_records, load_region_texts, sample_regions


def write_image_data(data_dir: Path) -> None:
    images = [
        {"image_id": 7, "url": "https://example.org/rak248/VG_100K/7.jpg"},
        {"image_id": 1, "url": "https://example.org/rak248/VG_100K_2/1.jpg"},
    ]
    (data_dir / "image_data.json").write_text(json.dumps(images))


def write_regions(data_dir: Path) -> None:
    regions = [
        {
            "id": 7,
            "regions": [
                {"region_id": 71, "image_id": 7, "phrase": "a red bus"},
                {"region_id": 70, "image_id": 7, "phrase": "   "},
            ],
        },
        {"id": 1, "regions": [{"region_id": 11, "image_id": 1, "phrase": "a green clock"}]},
    ]
    (data_dir / "region_descriptions.json").write_text(json.dumps(regions))


def make_regions(num_images: int, regions_per_image: int) -> list[TextRecord]:
    regions = []
    for image_id in range(num_images):
        for offset in range(regions_per_image):
            region_id = image_id * 1000 + offset
            regions.append(TextRecord(image_id, region_id, f"region {region_id}"))
    return regions


def test_image_path_uses_folder_from_url(tmp_path):
    write_image_data(tmp_path)
    records = load_image_records(tmp_path)
    assert records[0].image_id == 1
    assert records[0].path == tmp_path / "images" / "VG_100K_2" / "1.jpg"
    assert records[1].path == tmp_path / "images" / "VG_100K" / "7.jpg"


def test_images_dir_override(tmp_path):
    write_image_data(tmp_path)
    records = load_image_records(tmp_path, images_dir=Path("/mnt/vg"))
    assert records[0].path == Path("/mnt/vg/VG_100K_2/1.jpg")


def test_empty_phrases_are_dropped(tmp_path):
    write_regions(tmp_path)
    regions = load_region_texts(tmp_path)
    assert [region.text_id for region in regions] == [11, 71]
    assert regions[1].text == "a red bus"


def test_sampling_is_deterministic():
    regions = make_regions(num_images=20, regions_per_image=30)
    assert sample_regions(regions, 5, seed=0) == sample_regions(regions, 5, seed=0)


def test_sampling_changes_with_seed():
    regions = make_regions(num_images=20, regions_per_image=30)
    assert sample_regions(regions, 5, seed=0) != sample_regions(regions, 5, seed=1)


def test_sampling_takes_at_most_k_without_repeats():
    regions = make_regions(num_images=3, regions_per_image=30)
    sampled = sample_regions(regions, 5, seed=0)
    assert len(sampled) == 15
    assert len({region.text_id for region in sampled}) == 15


def test_image_with_fewer_regions_keeps_all():
    regions = make_regions(num_images=1, regions_per_image=3)
    assert sample_regions(regions, 5, seed=0) == regions


def test_sampling_output_sorted_by_image_then_region():
    regions = make_regions(num_images=3, regions_per_image=30)
    shuffled = list(reversed(regions))
    sampled = sample_regions(shuffled, 5, seed=0)
    keys = [(region.image_id, region.text_id) for region in sampled]
    assert keys == sorted(keys)
    assert sampled == sample_regions(regions, 5, seed=0)
