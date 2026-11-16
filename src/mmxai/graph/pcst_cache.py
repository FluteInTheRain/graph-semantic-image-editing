"""Packed on-disk cache of per-text PCST subgraphs (ADR-0051, stage 2).

PCST takes milliseconds per text, but stage 2 needs the subgraph S_q of every train text
at every epoch. So the subgraphs are computed once and cached.

Layout: `values` holds the concept rows of all subgraphs back to back, and `offsets[t]` is
where text t starts (text t ends at `offsets[t + 1]`). A `<split>.meta.json` next to the
arrays records how they were made; loading with different settings must fail, so a stale
cache can never be used silently.
"""

import json
from pathlib import Path

import numpy as np

PAD = -1  # marks unused slots in a padded matrix; real concept rows are >= 0


def pack_subgraphs(rows: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Pack one concept-row array per text into (offsets, values)."""
    offsets = np.zeros(len(rows) + 1, dtype=np.int64)
    for text, row in enumerate(rows):
        offsets[text + 1] = offsets[text] + len(row)
    if len(rows) == 0:
        values = np.zeros(0, dtype=np.int64)
    else:
        values = np.concatenate(rows).astype(np.int64)
    return offsets, values


def unpack_row(offsets: np.ndarray, values: np.ndarray, text: int) -> np.ndarray:
    """Concept rows of one text."""
    return values[offsets[text] : offsets[text + 1]]


def padded_rows(offsets: np.ndarray, values: np.ndarray) -> np.ndarray:
    """All texts as a (num_texts, width) matrix padded with PAD; width is at least 1."""
    num_texts = len(offsets) - 1
    lengths = np.diff(offsets)
    width = max(int(lengths.max()) if num_texts > 0 else 0, 1)
    matrix = np.full((num_texts, width), PAD, dtype=np.int64)
    for text in range(num_texts):
        row = unpack_row(offsets, values, text)
        matrix[text, : len(row)] = row
    return matrix


def save_cache(
    cache_dir: str | Path, split: str, offsets: np.ndarray, values: np.ndarray, meta: dict
) -> None:
    """Write `<split>.npz` and `<split>.meta.json`; never overwrite an existing cache."""
    cache_dir = Path(cache_dir)
    arrays_path = cache_dir / f"{split}.npz"
    if arrays_path.exists():
        raise FileExistsError(f"{arrays_path} already exists; delete it to rebuild")
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(arrays_path, offsets=offsets, values=values)
    meta_text = json.dumps(meta, indent=2, sort_keys=True)
    (cache_dir / f"{split}.meta.json").write_text(meta_text, encoding="utf-8")


def load_cache(
    cache_dir: str | Path, split: str, expected_meta: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Load (offsets, values); fail if the cache was made with other settings."""
    cache_dir = Path(cache_dir)
    meta_text = (cache_dir / f"{split}.meta.json").read_text(encoding="utf-8")
    stored_meta = json.loads(meta_text)
    for key, expected in expected_meta.items():
        if stored_meta.get(key) != expected:
            raise ValueError(
                f"cache {split}: {key} is {stored_meta.get(key)!r}, expected {expected!r}"
            )
    with np.load(cache_dir / f"{split}.npz") as arrays:
        offsets = arrays["offsets"]
        values = arrays["values"]
    if "num_texts" in expected_meta and len(offsets) - 1 != expected_meta["num_texts"]:
        raise ValueError(
            f"cache {split}: num_texts is {len(offsets) - 1}, expected {expected_meta['num_texts']}"
        )
    return offsets, values
