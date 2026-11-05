"""Decide which images are correct answers for each text query (ADR-0043).

Short region descriptions such as "a man" appear in many images. Counting only
the source image as correct would punish CLIP for returning another image that
has exactly the same sentence, so every such image counts as relevant.

Decision (ADR-0043): every gallery image that owns the same sentence is relevant,
because 26% of text vectors are repeated and "source image only" would mark
correct answers as wrong. Rejected: dropping repeated queries (biases the query
set towards long sentences) and matching strings (needs the multi-GB JSON).
Decision (ADR-0043): only the 5 sampled regions of each image are compared, so
absolute Recall is a lower bound; every model uses the same rule, so comparisons
stay fair.
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
    near-equal vectors.

    Decision (ADR-0043): threshold 0.9999 (config `same_text_threshold`), because
    batched GPU encoding moves equal sentences apart by about 1e-6 while different
    sentences sit far lower. It was fixed before any run and must never be tuned on results.
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
