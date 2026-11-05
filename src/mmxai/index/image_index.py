"""Exact nearest-neighbour search over image embeddings with FAISS (PDF section 4.1)."""

from dataclasses import dataclass

import faiss
import numpy as np


@dataclass
class ImageIndex:
    """A FAISS index plus the image id of each of its rows.

    FAISS only knows row numbers, so the ids are kept next to it to turn
    results back into image ids.
    """

    index: faiss.IndexFlatIP
    image_ids: np.ndarray


def build_image_index(embeddings: np.ndarray, image_ids: np.ndarray) -> ImageIndex:
    """Index L2-normalised embeddings so that inner product equals cosine similarity.

    IndexFlatIP compares the query with every image: exact and deterministic,
    and fast enough for a few thousand images.
    """
    if len(embeddings) != len(image_ids):
        raise ValueError(f"{len(embeddings)} embeddings but {len(image_ids)} image ids")
    vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return ImageIndex(index=index, image_ids=np.asarray(image_ids, dtype=np.int64))


def search_images(
    image_index: ImageIndex, query_embeddings: np.ndarray, top_k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (scores, image_ids), both of shape (num_queries, top_k), best first."""
    num_images = image_index.index.ntotal
    # FAISS fills missing results with row -1, which would silently map to the last image id.
    if top_k > num_images:
        raise ValueError(f"top_k={top_k} but the index holds only {num_images} images")
    queries = np.ascontiguousarray(query_embeddings, dtype=np.float32)
    scores, rows = image_index.index.search(queries, top_k)
    return scores, image_index.image_ids[rows]
