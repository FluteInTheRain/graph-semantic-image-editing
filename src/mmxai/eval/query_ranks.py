"""Per-query ranks under plain CLIP and CLIP + Graph, and the 5 success / 5 failure pick.

A query's rank is the 1-based position of its first relevant image (ADR-0043), or
None when no relevant image is among CLIP's top-k candidates. CLIP + Graph only
re-orders those candidates, so a query has a CLIP + Graph rank exactly when it has
a CLIP rank.

Selection rule (PDF section 7 asks for 5 success and 5 failure queries). Rules 1-5 were
fixed before looking at any result; rule 6 was added after the first val run showed
repeated sentences (commit fc26470, results/records/20261113_cap-do-1/README.md):
    1. Keep only queries whose relevant image is in CLIP's top-k (CLIP rank not None).
    2. change = CLIP rank - CLIP+Graph rank (positive = the graph moved it up).
    3. Success: change > 0, largest change first. Failure: change < 0, most negative first.
    4. Ties: the smaller query index (the query's row in the split) first.
    5. Take `count` of each; fewer if not enough exist. Unchanged queries are never picked.
    6. Queries with the same key (the same sentence) count once: only the first in the
       order above is kept. Visual Genome repeats region descriptions, and equal sentences
       get equal ranks, so without this rule one sentence could fill several slots.
"""

import numpy as np

from mmxai.eval.metrics import first_relevant_rank
from mmxai.rerank.fusion import Candidates, fuse_scores, rerank_ranks


def clip_ranks(candidates: Candidates) -> list[int | None]:
    """Rank of the first relevant image in CLIP's own order (the plain CLIP baseline)."""
    ranks = []
    for query in range(len(candidates.candidate_ids)):
        retrieved = candidates.candidate_ids[query].tolist()
        ranks.append(first_relevant_rank(retrieved, candidates.relevant[query]))
    return ranks


def fused_ranks(candidates: Candidates, graph_scores: np.ndarray, alpha: float) -> list[int | None]:
    """Rank of the first relevant image under final = alpha * clip + (1 - alpha) * graph.

    Calls the same fuse_scores and rerank_ranks as training and the test run, so the
    ranks are exactly the ones the pipeline reports.
    """
    final = fuse_scores(candidates.clip_scores, graph_scores, alpha)
    return rerank_ranks(candidates, final)


def ranked_columns(final_scores: np.ndarray) -> np.ndarray:
    """Candidate columns of one query, best first (the same stable sort as rerank_ranks)."""
    return np.argsort(-final_scores, kind="stable")


def rank_of_column(final_scores: np.ndarray, column: int) -> int:
    """1-based rank of one candidate column of one query under these final scores."""
    order = ranked_columns(final_scores)
    return int(np.flatnonzero(order == column)[0]) + 1


def select_queries(
    clip: list[int | None],
    fused: list[int | None],
    count: int,
    query_keys: list[str] | None = None,
) -> tuple[list[int], list[int]]:
    """Return (success, failure) query indices by the rule in the module docstring.

    query_keys: one key per query for rule 6 (e.g. the normalised sentence); None means
    every query is distinct. Each list has at most `count` indices; the caller warns
    when one has fewer.
    """
    if len(clip) != len(fused):
        raise ValueError(f"{len(clip)} CLIP ranks but {len(fused)} CLIP+Graph ranks")
    if query_keys is not None and len(query_keys) != len(clip):
        raise ValueError(f"{len(query_keys)} query keys but {len(clip)} queries")
    if count < 1:
        raise ValueError(f"count must be at least 1, got {count}")

    improved = []  # (change, query index)
    worsened = []
    for query in range(len(clip)):
        if clip[query] is None:
            continue  # rule 1: the relevant image is not among the candidates
        if fused[query] is None:
            raise ValueError(f"query {query} has a CLIP rank but no CLIP+Graph rank")
        change = clip[query] - fused[query]
        if change > 0:
            improved.append((change, query))
        elif change < 0:
            worsened.append((change, query))

    # Largest gain first, then smaller query index.
    improved.sort(key=lambda item: (-item[0], item[1]))
    # Largest loss (most negative change) first, then smaller query index.
    worsened.sort(key=lambda item: (item[0], item[1]))

    success = first_distinct(improved, count, query_keys)
    failure = first_distinct(worsened, count, query_keys)
    return success, failure


def first_distinct(
    ordered: list[tuple[int, int]], count: int, query_keys: list[str] | None
) -> list[int]:
    """The first `count` query indices of `ordered`, skipping a key already taken (rule 6)."""
    picked = []
    taken_keys = set()
    for _, query in ordered:
        if len(picked) == count:
            break
        key = str(query) if query_keys is None else query_keys[query]
        if key in taken_keys:
            continue
        taken_keys.add(key)
        picked.append(query)
    return picked
