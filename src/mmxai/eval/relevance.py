"""Decide which images are correct answers for each text query (ADR-0043).

Short region descriptions such as "a man" appear in many images. Counting only
the source image as correct would punish CLIP for returning another image that
has exactly the same sentence, so every such image counts as relevant.
"""

import numpy as np


def same_text_relevant_images(
    text_embeddings: np.ndarray,
    text_image_ids: np.ndarray,
    threshold: float,
    chunk_size: int = 1024,
) -> list[set[int]]:
    """For each text, return the ids of images that own a text with cosine >= threshold.

    The .npz files hold vectors, not strings, so equal sentences are found by
    near-equal vectors. The threshold only absorbs float noise from batched GPU
    encoding; it must never be tuned on results.
    """
    relevant: list[set[int]] = []
    # Chunks keep memory at chunk_size x num_texts instead of num_texts x num_texts.
    for start in range(0, len(text_embeddings), chunk_size):
        chunk = text_embeddings[start : start + chunk_size]
        similarities = chunk @ text_embeddings.T
        rows, columns = np.nonzero(similarities >= threshold)

        chunk_sets: list[set[int]] = []
        for row in range(len(chunk)):
            # The source image is always relevant, even if rounding puts self-similarity
            # a hair below the threshold.
            chunk_sets.append({int(text_image_ids[start + row])})
        for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
            chunk_sets[row].add(int(text_image_ids[column]))
        relevant.extend(chunk_sets)
    return relevant
