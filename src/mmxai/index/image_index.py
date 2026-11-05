"""Exact nearest-neighbour search over image embeddings with FAISS (PDF section 4.1)."""

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

INDEX_FILE = "images.faiss"
IDS_FILE = "image_ids.npy"
META_FILE = "meta.json"


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

    Decision (ADR-0043): use IndexFlatIP, an exact search over every image, not an
    approximate index (IVF, HNSW).
      Why: deterministic (the same list on every run) and searching 5,000 images takes
      seconds. Inner product equals cosine because the features are L2-normalised
      (ADR-0041).
      Decided by: FluteInTheRain, 2026-11-05 (proposed by Claude Code in ADR-0043;
      matches "FAISS Flat" assumed in ADR-0011).
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


def save_image_index(image_index: ImageIndex, folder: str | Path, meta: dict) -> None:
    """Write the index, its row-to-id map and a description. Refuses to overwrite.

    The id file matters as much as the index: without it a search result is only
    a row number. meta.json says where the vectors came from, for anyone who
    downloads the folder without the repo.
    """
    folder = Path(folder)
    if folder.exists():
        raise FileExistsError(f"{folder} already exists; delete it to build again")
    folder.mkdir(parents=True)
    faiss.write_index(image_index.index, str(folder / INDEX_FILE))
    np.save(folder / IDS_FILE, image_index.image_ids, allow_pickle=False)
    with open(folder / META_FILE, "w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2)


def load_image_index(folder: str | Path) -> ImageIndex:
    """Read a folder written by save_image_index."""
    folder = Path(folder)
    index = faiss.read_index(str(folder / INDEX_FILE))
    image_ids = np.load(folder / IDS_FILE, allow_pickle=False)
    if index.ntotal != len(image_ids):
        raise ValueError(f"{folder}: index has {index.ntotal} rows but {len(image_ids)} ids")
    return ImageIndex(index=index, image_ids=image_ids)
