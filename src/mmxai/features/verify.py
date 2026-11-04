"""Check an extracted feature set before anyone relies on it. Needs numpy only."""

from pathlib import Path

import numpy as np

from mmxai.features.cache import load_image_features, load_meta, load_text_features

NORM_TOLERANCE = 1e-3
CAPTIONS_PER_IMAGE = 5


class FeatureVerificationError(ValueError):
    """Raised with every problem found in a feature set."""


def check_embeddings(name: str, embeddings: np.ndarray) -> list[str]:
    """Cosine similarity is only correct for finite vectors of length 1."""
    if not np.isfinite(embeddings).all():
        return [f"{name} embeddings contain NaN or Inf"]
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.isclose(norms, 1.0, atol=NORM_TOLERANCE).all():
        return [f"{name} embeddings are not unit length"]
    return []


def check_row_counts(
    image_embeddings: np.ndarray, image_ids: np.ndarray, texts: dict, meta: dict
) -> list[str]:
    """Every array in a file must have one row per item, matching meta.json."""
    errors = []
    if len(image_embeddings) != len(image_ids):
        errors.append("image rows differ between embeddings and image_ids")
    if len(image_ids) != meta["num_images"]:
        errors.append(f"num_images in meta is {meta['num_images']}, found {len(image_ids)}")

    num_texts = len(texts["embeddings"])
    for name, values in texts.items():
        if len(values) != num_texts:
            errors.append(f"text rows differ between embeddings and {name}")
    if num_texts != meta["num_texts"]:
        errors.append(f"num_texts in meta is {meta['num_texts']}, found {num_texts}")
    return errors


def check_ids(image_ids: np.ndarray, text_image_ids: np.ndarray) -> list[str]:
    """Each text's image id is the retrieval ground truth, so it must point to a real image."""
    errors = []
    if len(np.unique(image_ids)) != len(image_ids):
        errors.append("image_ids contain duplicate values")
    if not np.isin(text_image_ids, image_ids).all():
        errors.append("some texts point to an unknown image")
    return errors


def check_texts_per_image(text_image_ids: np.ndarray, meta: dict) -> list[str]:
    """Visual Genome has at most k sampled regions; a complete Flickr30k has exactly 5 captions."""
    if len(text_image_ids) == 0:
        return []
    _, counts = np.unique(text_image_ids, return_counts=True)
    if meta["dataset"] == "visual_genome" and counts.max() > meta["regions_per_image"]:
        return [f"some images have more than {meta['regions_per_image']} regions"]
    # A trial run may stop in the middle of an image's captions.
    is_complete_flickr = meta["dataset"] == "flickr30k" and not meta["is_partial"]
    if is_complete_flickr and (counts != CAPTIONS_PER_IMAGE).any():
        return [f"some images do not have exactly {CAPTIONS_PER_IMAGE} captions"]
    return []


def check_alignment(
    image_embeddings: np.ndarray,
    image_ids: np.ndarray,
    text_embeddings: np.ndarray,
    text_image_ids: np.ndarray,
    seed: int,
    num_pairs: int,
) -> list[str]:
    """Correct text-image pairs should score above shuffled pairs.

    This only detects rows that lost their alignment. The result never feeds
    into choosing a setting, so it is not tuning on any split.
    """
    if len(image_ids) < 2 or len(text_image_ids) == 0:
        return []

    row_of_image = {int(image_id): row for row, image_id in enumerate(image_ids)}
    rng = np.random.default_rng(seed)
    sample_size = min(num_pairs, len(text_image_ids))
    text_rows = rng.choice(len(text_image_ids), size=sample_size, replace=False)
    matched_rows = np.array([row_of_image[int(text_image_ids[row])] for row in text_rows])
    shuffled_rows = rng.permutation(matched_rows)

    samples = text_embeddings[text_rows]
    matched_score = np.sum(samples * image_embeddings[matched_rows], axis=1).mean()
    shuffled_score = np.sum(samples * image_embeddings[shuffled_rows], axis=1).mean()
    if matched_score <= shuffled_score:
        return [
            "text and image rows look misaligned: "
            f"matched cosine {matched_score:.4f}, shuffled cosine {shuffled_score:.4f}"
        ]
    return []


def verify_features(output_dir: Path, seed: int = 0, num_pairs: int = 1000) -> None:
    """Raise FeatureVerificationError listing every problem found in output_dir."""
    meta = load_meta(output_dir)
    image_embeddings, image_ids = load_image_features(output_dir)
    texts = load_text_features(output_dir)
    text_embeddings = texts["embeddings"]
    text_image_ids = texts["image_ids"]

    errors = check_row_counts(image_embeddings, image_ids, texts, meta)
    errors += check_embeddings("image", image_embeddings)
    errors += check_embeddings("text", text_embeddings)
    errors += check_ids(image_ids, text_image_ids)
    errors += check_texts_per_image(text_image_ids, meta)

    # The alignment check looks rows up by id, so it needs the checks above to have passed.
    if not errors:
        errors += check_alignment(
            image_embeddings, image_ids, text_embeddings, text_image_ids, seed, num_pairs
        )

    if errors:
        raise FeatureVerificationError("; ".join(errors))
