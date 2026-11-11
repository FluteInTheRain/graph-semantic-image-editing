"""Contains edges predicted by frozen CLIP instead of read from annotations (ADR-0046).

Each image is linked to the k concepts whose CLIP text vector is closest to the
image's CLIP vector. Needs numpy only.

Decision (ADR-0046): use these edges for every split.
  Why: an image's edges depend only on its own vector, never on labels or on a
  statistic over other images. Test images therefore do not get edges from the
  annotations that also define the query answers (PR #6 review, point A2), and the
  same kind of edge is used in training and in testing.
  Decided by: qphulong, 2026-11-10.
"""

import numpy as np


def clip_topk_concepts(
    image_ids: np.ndarray,
    image_features: np.ndarray,
    concepts: list[str],
    concept_features: np.ndarray,
    k: int,
    chunk_size: int = 4096,
) -> dict[int, set[str]]:
    """Return {image_id: the k concepts with the highest cosine to that image}.

    Row i of image_features belongs to image_ids[i]; row j of concept_features
    belongs to concepts[j]. The result has the same form as the annotation-based
    mapping, so build_contains_graph turns it into edges unchanged.
    """
    if not 1 <= k <= len(concepts):
        raise ValueError(f"k must be between 1 and {len(concepts)} concepts, got {k}")

    # Normalise so the dot product is the cosine, whatever the stored vector lengths.
    image_unit = image_features / np.linalg.norm(image_features, axis=1, keepdims=True)
    concept_unit = concept_features / np.linalg.norm(concept_features, axis=1, keepdims=True)

    image_concepts = {}
    # Chunks keep memory small: all 108k images x 3.9k concepts at once is ~1.7 GB.
    for start in range(0, len(image_ids), chunk_size):
        similarity = image_unit[start : start + chunk_size] @ concept_unit.T
        # A stable sort on the negated similarity puts the highest first and breaks
        # ties by concept order, so the same input always gives the same edges.
        top_rows = np.argsort(-similarity, axis=1, kind="stable")[:, :k]
        chunk_ids = image_ids[start : start + chunk_size].tolist()
        for image_id, rows in zip(chunk_ids, top_rows, strict=True):
            image_concepts[image_id] = {concepts[row] for row in rows.tolist()}
    return image_concepts
