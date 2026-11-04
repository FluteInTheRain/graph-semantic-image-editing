"""Encode images and texts with a frozen CLIP model. CLIP is never trained (PDF section 4.1)."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from mmxai.data.records import ImageRecord

PROGRESS_EVERY_BATCHES = 100


@dataclass
class ClipBundle:
    """The frozen model together with the exact preprocessing it was trained with."""

    model: torch.nn.Module
    preprocess: Callable
    tokenizer: Callable
    device: str


def freeze_model(model: torch.nn.Module) -> torch.nn.Module:
    """Switch to inference mode and make sure no weight can change."""
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def resolve_device(device: str) -> str:
    """Turn "auto" into a real device name."""
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_frozen_clip(model_name: str, pretrained: str, device: str) -> ClipBundle:
    """Load CLIP weights and preprocessing from open_clip."""
    # Imported here so tests with a fake model do not need open_clip.
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    real_device = resolve_device(device)
    model = freeze_model(model).to(real_device)
    return ClipBundle(model, preprocess, tokenizer, real_device)


def library_versions() -> dict[str, str]:
    """Record the versions that produced the embeddings."""
    import open_clip

    return {"open_clip": open_clip.__version__, "torch": torch.__version__}


class ImageRecordDataset(Dataset):
    """Reads images in DataLoader workers, because decoding is the slow part."""

    def __init__(self, records: list[ImageRecord], preprocess: Callable) -> None:
        self.records = records
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[int, torch.Tensor | None]:
        record = self.records[index]
        # One bad file must not end a run of 100,000 images. The caller counts the
        # failures and stops if there are too many.
        try:
            with Image.open(record.path) as image:
                return record.image_id, self.preprocess(image.convert("RGB"))
        except (OSError, ValueError):
            return record.image_id, None


def collate_images(
    batch: list[tuple[int, torch.Tensor | None]],
) -> tuple[torch.Tensor | None, list[int], list[int]]:
    """Separate readable images from failed ones. Returns (images, image_ids, failed_ids)."""
    tensors = []
    image_ids = []
    failed_ids = []
    for image_id, tensor in batch:
        if tensor is None:
            failed_ids.append(image_id)
        else:
            tensors.append(tensor)
            image_ids.append(image_id)

    if not tensors:
        return None, image_ids, failed_ids
    return torch.stack(tensors), image_ids, failed_ids


def normalize(features: torch.Tensor) -> np.ndarray:
    """Scale to length 1 so a dot product equals cosine similarity."""
    features = features.float()
    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().astype(np.float32)


def join_batches(batches: list[np.ndarray]) -> np.ndarray:
    """Join batch results. np.concatenate cannot handle an empty list."""
    if not batches:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(batches, axis=0)


def encode_images(
    bundle: ClipBundle, records: list[ImageRecord], batch_size: int, num_workers: int
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Return (embeddings, image_ids that were encoded, failed image_ids), in input order."""
    dataset = ImageRecordDataset(records, bundle.preprocess)
    # shuffle=False keeps rows aligned with the ids we record.
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_images,
    )

    batches = []
    all_image_ids = []
    all_failed_ids = []
    with torch.no_grad():
        for batch_number, (images, image_ids, failed_ids) in enumerate(loader, start=1):
            all_failed_ids.extend(failed_ids)
            if images is not None:
                features = bundle.model.encode_image(images.to(bundle.device))
                batches.append(normalize(features))
                all_image_ids.extend(image_ids)
            if batch_number % PROGRESS_EVERY_BATCHES == 0:
                print(f"images: {len(all_image_ids)} encoded, {len(all_failed_ids)} failed")

    return join_batches(batches), np.array(all_image_ids, dtype=np.int64), all_failed_ids


def encode_texts(bundle: ClipBundle, texts: list[str], batch_size: int) -> np.ndarray:
    """Return one embedding per text, in input order."""
    batches = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            tokens = bundle.tokenizer(texts[start : start + batch_size]).to(bundle.device)
            batches.append(normalize(bundle.model.encode_text(tokens)))
            batch_number = len(batches)
            if batch_number % PROGRESS_EVERY_BATCHES == 0:
                print(f"texts: {start + batch_size} encoded")
    return join_batches(batches)
