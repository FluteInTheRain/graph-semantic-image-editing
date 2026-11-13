"""Image -> text retrieval with the same pair score as text -> image (spec 5.2, ADR-0043).

An image is the query and the split's texts are the gallery. A text t is scored for
the query image i with the same pair score the other direction uses:

    final = alpha * cos(t, x_i) + (1 - alpha) * cos(t, h_i)

x_i is the CLIP vector and h_i the graph vector of image i, so alpha means the same
thing in both directions.

Relevance (spec 5.2): text row r is correct for image i when i owns a text with the
same sentence as r (the rule of ADR-0043). The image's own texts are always correct.
"""

from dataclasses import dataclass

import numpy as np

from mmxai.eval.relevance import same_text_relevant_images
from mmxai.index.image_index import build_image_index, search_images


@dataclass(frozen=True)
class TextCandidates:
    """CLIP's top-k texts for every query image of one split, best first."""

    query_image_ids: np.ndarray  # (num_images,) id of each query image
    text_embeddings: np.ndarray  # (num_texts, dim) the split's texts; a text row indexes here
    candidate_rows: np.ndarray  # (num_images, top_k) text rows
    clip_scores: np.ndarray  # (num_images, top_k) cos(t, x_i), decreasing along a row
    relevant: list[set[int]]  # per image, the text rows that count as correct


def retrieve_text_candidates(
    image_embeddings: np.ndarray,
    image_ids: np.ndarray,
    text_embeddings: np.ndarray,
    text_image_ids: np.ndarray,
    eval_image_ids: np.ndarray,
    top_k: int,
    same_text_threshold: float,
) -> TextCandidates:
    """Search the split's texts with the split's images.

    The masks are the same as in retrieve_candidates, so both directions judge the
    same images and the same texts.
    """
    missing = np.setdiff1d(eval_image_ids, image_ids)
    if len(missing) > 0:
        raise ValueError(f"{len(missing)} split images have no embedding, e.g. {missing[0]}")
    missing_texts = np.setdiff1d(eval_image_ids, text_image_ids)
    if len(missing_texts) > 0:
        raise ValueError(f"{len(missing_texts)} split images have no text, e.g. {missing_texts[0]}")

    image_rows = np.isin(image_ids, eval_image_ids)
    text_rows = np.isin(text_image_ids, eval_image_ids)
    query_embeddings = image_embeddings[image_rows]
    query_image_ids = image_ids[image_rows]
    gallery_embeddings = text_embeddings[text_rows]
    gallery_image_ids = text_image_ids[text_rows]

    # The index "ids" are the text rows, so a search result is directly a text row.
    text_row_numbers = np.arange(len(gallery_embeddings))
    index = build_image_index(gallery_embeddings, text_row_numbers)
    clip_scores, candidate_rows = search_images(index, query_embeddings, top_k)

    # same_text_relevant_images answers "which images are right for this text";
    # invert it to "which texts are right for this image".
    images_of_text = same_text_relevant_images(
        gallery_embeddings, gallery_image_ids, same_text_threshold
    )
    texts_of_image: dict[int, set[int]] = {}
    for image_id in query_image_ids.tolist():
        texts_of_image[image_id] = set()
    for text_row, images in enumerate(images_of_text):
        for image_id in images:
            texts_of_image[image_id].add(text_row)
    relevant = [texts_of_image[image_id] for image_id in query_image_ids.tolist()]

    return TextCandidates(
        query_image_ids, gallery_embeddings, candidate_rows, clip_scores, relevant
    )


def text_graph_scores(
    candidates: TextCandidates,
    image_h: np.ndarray,
    h_image_ids: np.ndarray,
    chunk_size: int = 512,
) -> np.ndarray:
    """Return cos(t, h_i) for every candidate text: shape (num_images, top_k).

    image_h row r is the graph vector of image h_image_ids[r].
    """
    row_of_id = {}
    for row, image_id in enumerate(h_image_ids.tolist()):
        row_of_id[image_id] = row
    missing = set(candidates.query_image_ids.tolist()) - set(row_of_id)
    if missing:
        raise ValueError(f"{len(missing)} query images have no graph vector, e.g. {min(missing)}")

    unit_h = image_h / np.linalg.norm(image_h, axis=1, keepdims=True)
    texts = candidates.text_embeddings
    unit_texts = texts / np.linalg.norm(texts, axis=1, keepdims=True)
    query_h_rows = np.array([row_of_id[i] for i in candidates.query_image_ids.tolist()])
    query_h = unit_h[query_h_rows]  # (num_images, dim)

    scores = np.empty(candidates.candidate_rows.shape, dtype=np.float32)
    # Chunks: all images x 100 texts x 512 numbers at once would take gigabytes.
    for start in range(0, len(scores), chunk_size):
        stop = start + chunk_size
        chunk_texts = unit_texts[candidates.candidate_rows[start:stop]]  # (chunk, top_k, dim)
        scores[start:stop] = np.einsum("qkd,qd->qk", chunk_texts, query_h[start:stop])
    return scores


def ranked_text_rows(candidates: TextCandidates, final_scores: np.ndarray) -> list[list[int]]:
    """Return each image's candidate text rows sorted by final score, best first.

    The sort is stable, so equal scores keep CLIP's order (alpha = 1 is plain CLIP).
    """
    if final_scores.shape != candidates.candidate_rows.shape:
        raise ValueError("final_scores shape must match candidate_rows")
    rankings = []
    for image in range(len(final_scores)):
        order = np.argsort(-final_scores[image], kind="stable")
        rankings.append(candidates.candidate_rows[image, order].tolist())
    return rankings
