"""Train the ADR-0046 model to re-rank CLIP's top-k (contrastive loss with hard negatives).

For one training text t (a region description of a TRAIN image):
    candidates = its own image (correct) + n_hard wrong images from CLIP's top-k of t
    loss       = cross-entropy of cos(t, h)/temperature over the candidates (InfoNCE)

Everything here sees train images and train texts only (PDF section 6, ADR-0046).
"""

import copy
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional

from mmxai.index.image_index import build_image_index, search_images
from mmxai.models.sage import ConceptToImageSage


@dataclass(frozen=True)
class TrainingData:
    """Train images in graph row order, their concept means, and the train texts."""

    image_x: torch.Tensor  # (num_images, dim) CLIP vectors of train images
    neighbor_mean: torch.Tensor  # (num_images, dim) mean CLIP vector of each image's concepts
    text_x: torch.Tensor  # (num_texts, dim) CLIP vectors of train texts
    text_image_row: torch.Tensor  # (num_texts,) row of the image each text describes
    candidate_rows: torch.Tensor  # (num_texts, top_k) CLIP's top-k image rows per text
    is_negative: torch.Tensor  # (num_texts, top_k) False for the text's own image and duplicates


def same_text_mask(
    text_x: np.ndarray,
    text_image_row: np.ndarray,
    candidate_rows: np.ndarray,
    threshold: float,
    chunk_size: int = 128,
) -> np.ndarray:
    """Mark candidates that own a text equal to the query text (cosine >= threshold).

    Such an image is a correct answer under ADR-0043, so it must not be trained as a
    wrong one. Only each text's own candidates are checked, not all 500k texts.
    """
    num_images = int(max(text_image_row.max(), candidate_rows.max())) + 1
    # texts_of_image[r] lists the text rows of image r, padded with -1.
    counts = np.bincount(text_image_row, minlength=num_images)
    texts_of_image = np.full((num_images, max(int(counts.max()), 1)), -1, dtype=np.int64)
    filled = np.zeros(num_images, dtype=np.int64)
    for text_row, image_row in enumerate(text_image_row.tolist()):
        texts_of_image[image_row, filled[image_row]] = text_row
        filled[image_row] += 1

    is_same = np.zeros(candidate_rows.shape, dtype=bool)
    for start in range(0, len(text_x), chunk_size):
        stop = start + chunk_size
        other_texts = texts_of_image[candidate_rows[start:stop]]  # (chunk, top_k, max_texts)
        other_x = text_x[np.maximum(other_texts, 0)]  # padding reads row 0, masked below
        similarity = np.einsum("ckmd,cd->ckm", other_x, text_x[start:stop])
        similarity[other_texts < 0] = -1.0
        is_same[start:stop] = (similarity >= threshold).any(axis=2)
    return is_same


def hard_negative_pool(
    image_x: np.ndarray,
    text_x: np.ndarray,
    text_image_row: np.ndarray,
    top_k: int,
    same_text_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (candidate_rows, is_negative) for every text, searched among TRAIN images only.

    Decision (ADR-0046): hard negatives are wrong images from CLIP's top-k of the text.
      Why: these are exactly the mistakes that re-ranking has to fix.
      Decided by: qphulong, 2026-11-10 (training target); n_hard is a Claude Code proposal.
    """
    image_rows = np.arange(len(image_x), dtype=np.int64)
    index = build_image_index(image_x, image_rows)
    _, candidate_rows = search_images(index, text_x, top_k)
    is_own_image = candidate_rows == text_image_row[:, None]
    is_duplicate = same_text_mask(text_x, text_image_row, candidate_rows, same_text_threshold)
    return candidate_rows, ~(is_own_image | is_duplicate)


def sample_negatives(
    is_negative: torch.Tensor, num_negatives: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick num_negatives allowed columns per row at random.

    Returns (columns, valid). valid is False where a row had fewer allowed columns
    than asked; those slots are masked out of the loss.
    """
    if num_negatives > is_negative.shape[1]:
        raise ValueError(
            f"cannot sample {num_negatives} negatives from {is_negative.shape[1]} candidates"
        )
    keys = torch.rand(is_negative.shape, generator=generator)
    keys[~is_negative] = 2.0  # larger than any random key, so picked last
    keys, columns = torch.topk(keys, num_negatives, dim=1, largest=False)
    return columns, keys <= 1.0


def batch_loss(
    model: ConceptToImageSage,
    data: TrainingData,
    text_rows: torch.Tensor,
    num_negatives: int,
    temperature: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """InfoNCE loss of one batch of texts; the correct image is always column 0."""
    columns, valid = sample_negatives(data.is_negative[text_rows], num_negatives, generator)
    negative_rows = torch.gather(data.candidate_rows[text_rows], 1, columns)
    image_rows = torch.cat([data.text_image_row[text_rows, None], negative_rows], dim=1)

    h = model.forward_from_neighbor_mean(data.image_x[image_rows], data.neighbor_mean[image_rows])
    query = functional.normalize(data.text_x[text_rows], dim=-1)
    cosine = (functional.normalize(h, dim=-1) * query[:, None, :]).sum(dim=-1)

    logits = cosine / temperature
    always_valid = torch.ones(len(text_rows), 1, dtype=torch.bool)
    logits = logits.masked_fill(~torch.cat([always_valid, valid], dim=1), float("-inf"))
    targets = torch.zeros(len(text_rows), dtype=torch.long)
    return functional.cross_entropy(logits, targets)


def train_one_epoch(
    model: ConceptToImageSage,
    data: TrainingData,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    num_negatives: int,
    temperature: float,
    generator: torch.Generator,
) -> float:
    """One pass over every training text in random order; returns the mean loss."""
    model.train()
    order = torch.randperm(len(data.text_x), generator=generator)
    total = 0.0
    for start in range(0, len(order), batch_size):
        text_rows = order[start : start + batch_size]
        loss = batch_loss(model, data, text_rows, num_negatives, temperature, generator)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(text_rows)
    return total / len(order)


@dataclass
class FitResult:
    best_state: dict
    best_epoch: int  # 0 means the untrained model (plain CLIP) was best
    history: list[dict]


def fit(
    model: ConceptToImageSage,
    data: TrainingData,
    validate: Callable[[ConceptToImageSage], float],
    learning_rate: float,
    batch_size: int,
    num_negatives: int,
    temperature: float,
    max_epochs: int,
    patience: int,
    generator: torch.Generator,
    log: Callable[[str], None] = print,
) -> FitResult:
    """Train with early stopping: keep the weights of the epoch with the best val score.

    validate(model) returns the val score to maximise. It is called before training
    too, so if training never beats the untrained model, the untrained model is kept.
    """
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=learning_rate)

    best_score = validate(model)
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    history = [{"epoch": 0, "train_loss": None, "val_score": best_score}]
    log(f"epoch 0 (untrained): val {best_score:.4f}")

    for epoch in range(1, max_epochs + 1):
        loss = train_one_epoch(
            model, data, optimizer, batch_size, num_negatives, temperature, generator
        )
        score = validate(model)
        history.append({"epoch": epoch, "train_loss": loss, "val_score": score})
        log(f"epoch {epoch}: train loss {loss:.4f}, val {score:.4f}")
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            log(f"no val improvement for {patience} epochs, stopping")
            break
    return FitResult(best_state=best_state, best_epoch=best_epoch, history=history)
