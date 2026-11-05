"""Tests for the extraction pipeline. A fake encoder module replaces torch and CLIP."""

import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from mmxai.data.records import ImageRecord, TextRecord
from mmxai.features.cache import load_image_features, load_meta, load_text_features
from mmxai.features.pipeline import (
    apply_limits,
    build_meta,
    check_failed_fraction,
    filter_texts_for_images,
    load_dataset_records,
    run_extraction,
    with_path_overrides,
)
from mmxai.features.verify import FeatureVerificationError
from mmxai.utils.config import FeatureConfig

DIM = 16


def make_config(tmp_path: Path, dataset: str = "flickr30k") -> FeatureConfig:
    sampling = {}
    if dataset == "visual_genome":
        sampling = {"regions_per_image": 2, "sampling_seed": 0}
    return FeatureConfig(
        dataset=dataset,
        dataset_version="1.2" if dataset == "visual_genome" else "karpathy",
        data_dir=str(tmp_path / "data"),
        output_root=str(tmp_path / "out"),
        model="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        device="cpu",
        image_batch_size=2,
        text_batch_size=2,
        num_workers=0,
        max_failed_image_fraction=0.5,
        **sampling,
    )


def write_flickr_files(data_dir: Path, num_images: int = 6) -> None:
    data_dir.mkdir(parents=True)
    images = []
    for image_id in range(1, num_images + 1):
        sentences = [{"raw": f"{image_id}"} for _ in range(5)]
        images.append({"filename": f"{image_id}.jpg", "imgid": 0, "sentences": sentences})
    (data_dir / "dataset_flickr30k.json").write_text(json.dumps({"images": images}))


def write_visual_genome_files(data_dir: Path) -> None:
    data_dir.mkdir(parents=True)
    images = []
    regions = []
    for image_id in [1, 2]:
        images.append({"image_id": image_id, "url": f"https://x.org/VG_100K/{image_id}.jpg"})
        image_regions = [
            {"region_id": image_id * 10 + offset, "image_id": image_id, "phrase": f"{image_id}"}
            for offset in range(4)
        ]
        regions.append({"id": image_id, "regions": image_regions})
    (data_dir / "image_data.json").write_text(json.dumps(images))
    (data_dir / "region_descriptions.json").write_text(json.dumps(regions))


def vector_for(image_id: int) -> np.ndarray:
    """A unit vector that depends on the image, so texts and images line up."""
    vector = np.random.default_rng(image_id).normal(size=DIM)
    return (vector / np.linalg.norm(vector)).astype(np.float32)


@pytest.fixture
def fake_encoder(monkeypatch):
    """Stand in for clip_encoder. Image id 2 fails if listed in `unreadable`."""
    module = types.ModuleType("mmxai.features.clip_encoder")
    module.unreadable = set()
    module.calls = []

    def encode_images(bundle, records, batch_size, num_workers):
        module.calls.append("images")
        kept = [record.image_id for record in records if record.image_id not in module.unreadable]
        failed = [record.image_id for record in records if record.image_id in module.unreadable]
        embeddings = np.stack([vector_for(image_id) for image_id in kept])
        return embeddings, np.array(kept, dtype=np.int64), failed

    def encode_texts(bundle, texts, batch_size):
        module.calls.append("texts")
        return np.stack([vector_for(int(text)) for text in texts])

    module.load_frozen_clip = lambda model_name, pretrained, device: types.SimpleNamespace(
        device="cpu"
    )
    module.library_versions = lambda: {"open_clip": "3.3.0", "torch": "fake"}
    module.encode_images = encode_images
    module.encode_texts = encode_texts
    monkeypatch.setitem(sys.modules, "mmxai.features.clip_encoder", module)
    return module


def test_failed_fraction_at_threshold_passes():
    check_failed_fraction(1, 1000, 0.001)


def test_failed_fraction_above_threshold_raises():
    with pytest.raises(ValueError, match="2 of 1000"):
        check_failed_fraction(2, 1000, 0.001)


def test_filter_texts_drops_failed_images():
    texts = [TextRecord(1, 0, "a"), TextRecord(2, 0, "b")]
    assert filter_texts_for_images(texts, {2}) == [TextRecord(2, 0, "b")]


def test_load_dataset_records_samples_visual_genome(tmp_path):
    config = make_config(tmp_path, "visual_genome")
    write_visual_genome_files(Path(config.data_dir))
    images, texts = load_dataset_records(config)
    assert len(images) == 2
    assert len(texts) == 4


def test_apply_limits_keeps_texts_of_kept_images():
    images = [ImageRecord(image_id, Path(f"{image_id}.jpg")) for image_id in [1, 2, 3]]
    texts = [TextRecord(image_id, 0, "x") for image_id in [1, 2, 3]]
    kept_images, kept_texts = apply_limits(images, texts, limit_images=2, limit_texts=None)
    assert [image.image_id for image in kept_images] == [1, 2]
    assert [text.image_id for text in kept_texts] == [1, 2]


def test_apply_limits_caps_texts():
    images = [ImageRecord(1, Path("1.jpg"))]
    texts = [TextRecord(1, index, "x") for index in range(5)]
    _, kept_texts = apply_limits(images, texts, limit_images=None, limit_texts=3)
    assert len(kept_texts) == 3


def test_meta_matches_spec_fields(tmp_path):
    config = make_config(tmp_path, "visual_genome")
    meta = build_meta(config, 2, 4, 512, [], False, {"open_clip": "3.3.0", "torch": "t"}, "abc")
    assert meta["text_source"] == "region_descriptions"
    assert meta["regions_per_image"] == 2
    assert meta["embedding_dim"] == 512
    assert meta["normalized"] is True
    assert meta["open_clip_version"] == "3.3.0"
    assert meta["code_hash"] == "abc"
    assert "created_at" in meta


def test_meta_marks_partial_run(tmp_path):
    config = make_config(tmp_path)
    meta = build_meta(config, 2, 4, 512, [], True, {"open_clip": "x", "torch": "t"}, "abc")
    assert meta["is_partial"] is True
    assert meta["text_source"] == "karpathy_captions"


def test_with_path_overrides_changes_only_given_paths(tmp_path):
    config = make_config(tmp_path)
    changed = with_path_overrides(config, None, "/images", "/out")
    assert changed.data_dir == config.data_dir
    assert changed.images_dir == "/images"
    assert changed.output_root == "/out"


def test_importing_pipeline_does_not_import_torch():
    code = "import sys, mmxai.features.pipeline; sys.exit('torch' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_run_extraction_writes_verified_flickr_output(tmp_path, fake_encoder):
    config = make_config(tmp_path)
    write_flickr_files(Path(config.data_dir))
    output_dir = run_extraction(config)

    embeddings, image_ids = load_image_features(output_dir)
    texts = load_text_features(output_dir)
    assert output_dir.parent.name == "flickr30k"
    assert embeddings.shape == (6, DIM)
    assert image_ids.tolist() == [1, 2, 3, 4, 5, 6]
    assert texts["embeddings"].shape == (30, DIM)
    assert texts["caption_index"].tolist()[:5] == [0, 1, 2, 3, 4]
    assert "text_ids" not in texts
    assert len(load_meta(output_dir)["code_hash"]) == 64


def test_run_extraction_uses_region_ids_for_visual_genome(tmp_path, fake_encoder):
    config = make_config(tmp_path, "visual_genome")
    write_visual_genome_files(Path(config.data_dir))
    output_dir = run_extraction(config)
    texts = load_text_features(output_dir)
    assert len(texts["text_ids"]) == 4
    assert set(texts["text_ids"].tolist()) <= {10, 11, 12, 13, 20, 21, 22, 23}


def test_run_extraction_drops_texts_of_failed_images(tmp_path, fake_encoder):
    config = make_config(tmp_path)
    write_flickr_files(Path(config.data_dir))
    fake_encoder.unreadable.add(2)
    output_dir = run_extraction(config)

    texts = load_text_features(output_dir)
    assert 2 not in texts["image_ids"].tolist()
    assert load_meta(output_dir)["failed_image_ids"] == [2]


def test_run_extraction_stops_when_too_many_images_fail(tmp_path, fake_encoder):
    config = make_config(tmp_path)
    write_flickr_files(Path(config.data_dir))
    fake_encoder.unreadable.update({1, 2, 3, 4})
    with pytest.raises(ValueError, match="4 of 6"):
        run_extraction(config)
    assert not Path(config.output_root).exists()


def test_partial_run_is_marked_and_kept_apart(tmp_path, fake_encoder):
    config = make_config(tmp_path)
    write_flickr_files(Path(config.data_dir))
    output_dir = run_extraction(config, limit_images=3, limit_texts=7)

    assert output_dir.parent.name == "flickr30k_partial"
    meta = load_meta(output_dir)
    assert meta["is_partial"] is True
    assert meta["num_images"] == 3
    assert meta["num_texts"] == 7


def test_rerun_fails_before_encoding(tmp_path, fake_encoder):
    config = make_config(tmp_path)
    write_flickr_files(Path(config.data_dir))
    run_extraction(config)
    fake_encoder.calls.clear()
    with pytest.raises(FileExistsError):
        run_extraction(config)
    assert fake_encoder.calls == []


def test_failed_verification_leaves_no_output_folder(tmp_path, fake_encoder):
    config = make_config(tmp_path)
    write_flickr_files(Path(config.data_dir))
    fake_encoder.encode_texts = lambda bundle, texts, batch_size: np.full((len(texts), DIM), np.nan)
    with pytest.raises(FeatureVerificationError):
        run_extraction(config)
    # Nothing may remain that could be published as a finished result.
    assert list(Path(config.output_root).rglob("meta.json")) == []


def test_run_extraction_prints_the_device(tmp_path, fake_encoder, capsys):
    config = make_config(tmp_path)
    write_flickr_files(Path(config.data_dir))
    fake_encoder.load_frozen_clip = lambda model, pretrained, device: types.SimpleNamespace(
        device="cuda"
    )
    run_extraction(config)
    assert "device: cuda" in capsys.readouterr().out
