"""Tests for the CLIP encoder, using a small fake model instead of downloaded weights."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")

from mmxai.data.records import ImageRecord  # noqa: E402 - only importable when torch exists
from mmxai.features.clip_encoder import (  # noqa: E402 - only importable when torch exists
    ClipBundle,
    encode_images,
    encode_texts,
    freeze_model,
)


class FakeClip(torch.nn.Module):
    """Returns vectors that depend on the input, like CLIP but with no weights."""

    def encode_image(self, images):
        return images.mean(dim=(1, 2))

    def encode_text(self, tokens):
        return tokens


def fake_preprocess(image):
    # CLIP expects three channels, so a grayscale image must arrive as RGB.
    array = np.asarray(image, dtype=np.float32) / 255 + 0.01
    assert array.shape[-1] == 3
    return torch.from_numpy(array)


def fake_tokenizer(texts):
    return torch.tensor([[float(len(text)), 1.0, 2.0] for text in texts])


def make_bundle():
    return ClipBundle(FakeClip(), fake_preprocess, fake_tokenizer, "cpu")


def write_image(path, color, mode="RGB"):
    Image.new(mode, (4, 4), color).save(path)
    return path


def test_freeze_model_disables_gradients_and_sets_eval():
    model = torch.nn.Linear(2, 2)
    frozen = freeze_model(model)
    assert frozen.training is False
    assert all(not parameter.requires_grad for parameter in frozen.parameters())


def test_encode_images_returns_unit_float32(tmp_path):
    records = [ImageRecord(1, write_image(tmp_path / "1.png", (200, 10, 10)))]
    embeddings, image_ids, failed = encode_images(make_bundle(), records, 2, 0)
    assert embeddings.dtype == np.float32
    assert np.isclose(np.linalg.norm(embeddings[0]), 1.0)
    assert image_ids.tolist() == [1]
    assert failed == []


@pytest.mark.parametrize("num_workers", [0, 2])
def test_encode_images_keeps_input_order(tmp_path, num_workers):
    records = []
    for image_id in [5, 3, 9, 1, 7]:
        path = write_image(tmp_path / f"{image_id}.png", (image_id * 20, 5, 5))
        records.append(ImageRecord(image_id, path))
    embeddings, image_ids, _ = encode_images(make_bundle(), records, 2, num_workers)
    assert image_ids.tolist() == [5, 3, 9, 1, 7]
    assert embeddings.shape == (5, 3)


def test_encode_images_reports_missing_and_corrupt(tmp_path):
    good = write_image(tmp_path / "good.png", (1, 2, 3))
    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_bytes(b"not an image")
    records = [
        ImageRecord(1, good),
        ImageRecord(2, tmp_path / "missing.jpg"),
        ImageRecord(3, corrupt),
    ]
    embeddings, image_ids, failed = encode_images(make_bundle(), records, 2, 0)
    assert image_ids.tolist() == [1]
    assert len(embeddings) == 1
    assert failed == [2, 3]


def test_encode_images_handles_grayscale(tmp_path):
    records = [ImageRecord(1, write_image(tmp_path / "gray.png", 100, mode="L"))]
    _, image_ids, failed = encode_images(make_bundle(), records, 1, 0)
    assert image_ids.tolist() == [1]
    assert failed == []


def test_encode_texts_preserves_row_count_across_batches():
    embeddings = encode_texts(make_bundle(), ["a", "bb", "ccc", "dddd", "e"], 2)
    assert embeddings.shape == (5, 3)
    assert embeddings.dtype == np.float32
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)
