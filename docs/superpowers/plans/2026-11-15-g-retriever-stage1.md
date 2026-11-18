# G-Retriever Stage 1 (PCST subgraph rerank) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-rank CLIP's top-100 with a query-dependent graph score: for each query, PCST picks a connected subgraph `S_q` of the concept co-occurrence graph, and an image scores `|N(i) ∩ S_q| / |S_q|` (ADR-0051, stage 1, no training). Hyperparameters `k`, `C`, `alpha` are chosen on val; test runs once. Includes ablations "no tree" and "shuffled PMI edges".

**Architecture:** `src/mmxai/graph/pcst.py` (prizes, PCST call, undirected pairs) → `src/mmxai/models/subgraph_scorer.py` (image concept sets, overlap score, explanation) → `src/mmxai/eval/pcst_rerank.py` (config, select on val, evaluate test). Inputs, candidates, fusion, alpha choice, metrics and run files reuse `mmxai.eval.graph_rerank` and `mmxai.rerank.fusion` unchanged.

**Tech Stack:** Python 3.11, numpy, networkx (already in group `graph`), `pcst-fast` (new, GPL, ADR-0051), pytest, ruff.

**Spec:** `docs/decisions/0051-g-retriever-chon-do-thi-con-pcst-de-xep-hang-lai.md`. Reference code: `retrieval_via_pcst` in https://github.com/XiaoxinHe/G-Retriever/blob/main/src/dataset/utils/retrieval.py (node prizes `topk..1`, constant edge cost, `pcst_fast(edges, prizes, costs, -1, 1, "gw", 0)`, nodes = selected vertices ∪ endpoints of selected edges).

**Repo rules:** branch `feat/g-retriever-pcst-rerank` (checked out; ADR-0051 committed). Commands as `python -m uv run ...`. English code/comments, Vietnamese docs, simple viva-explainable code, no hard-coded paths/hyperparameters. Commit after each task, never push. Do **not** run anything on the test split in this plan; real runs need artifacts that are not on this machine (see "Out of scope").

**Known pre-existing failures (ignore):** 2 tests in `tests/test_notebooks_match_src.py` (Windows symlink).

---

## File map

| File | Responsibility |
|---|---|
| `pyproject.toml`, `uv.lock` | Add `pcst-fast` to group `graph` |
| `src/mmxai/graph/pcst.py` (new) | Undirected concept pairs, rank prizes, PCST subgraph, "no tree" subgraph |
| `src/mmxai/graph/shuffle.py` (modify) | `shuffle_concept_pairs` (degree-preserving, ablation 2) |
| `src/mmxai/models/subgraph_scorer.py` (new) | Concepts per image, overlap scores for candidates, explanation record |
| `src/mmxai/eval/pcst_rerank.py` (new) | `PcstRerankConfig`, `select_on_val`, `evaluate_test` |
| `scripts/select_pcst_rerank.py`, `scripts/evaluate_pcst_rerank.py` (new) | Thin CLIs |
| `configs/clip_pcst_vg.yaml`, `configs/clip_pcst_vg_no_tree.yaml`, `configs/clip_pcst_vg_shuffled.yaml` (new) | Main run + 2 ablations |
| `tests/test_pcst.py`, `tests/test_subgraph_scorer.py`, `tests/test_pcst_rerank.py` (new), `tests/test_graph_shuffle*.py` (extend) | Synthetic tests |

---

### Task 1: Add the `pcst-fast` dependency

**Files:** `pyproject.toml`, `uv.lock`

- [ ] **Step 1:** Run `python -m uv add --group graph "pcst-fast>=1.0.10"`.
- [ ] **Step 2:** Check it imports and solves a 3-node toy problem:

```bash
python -m uv run python -c "import numpy as np; from pcst_fast import pcst_fast; v, e = pcst_fast(np.array([[0,1],[1,2]]), np.array([2.0,0.0,2.0]), np.array([0.5,0.5]), -1, 1, 'gw', 0); print(sorted(v.tolist()), sorted(e.tolist()))"
```

Expected: `[0, 1, 2] [0, 1]` (the zero-prize middle node is kept as a bridge). If install fails on Windows (no wheel), STOP and report BLOCKED with the error; do not vendor or reimplement.
- [ ] **Step 3:** In `pyproject.toml`, put a comment above the new line: `# PCST solver for the G-Retriever reranker (ADR-0051). License: GNU GPL.` Run `python -m uv run pytest -q tests/test_scene_graph.py` (env still works).
- [ ] **Step 4: Commit** `git add pyproject.toml uv.lock && git commit -m "build(deps): add pcst-fast for the G-Retriever reranker (ADR-0051)" -m "Co-Authored-By: <trailer from your system reminder>"`

---

### Task 2: PCST subgraph of a query

**Files:** Create `src/mmxai/graph/pcst.py`, Test `tests/test_pcst.py`

- [ ] **Step 1: Failing tests** `tests/test_pcst.py`

```python
"""PCST subgraphs of a query on a tiny synthetic concept graph (ADR-0051)."""

import numpy as np
import pytest

pytest.importorskip("pcst_fast")

from mmxai.graph.pcst import no_tree_subgraph, query_subgraph, rank_prizes, undirected_pairs  # noqa: E402

# Concepts 0..3. Only 1 links 0 and 2; 3 hangs off 2.
PAIRS = np.array([[0, 1], [1, 2], [2, 3]])
# Unit vectors in 2-D: the query points between concept 0 and concept 2.
CONCEPTS = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
QUERY = np.array([0.6, 0.8])


def test_rank_prizes_give_k_to_the_best_and_zero_outside_top_k():
    similarity = np.array([0.1, 0.9, 0.5, 0.7])

    assert rank_prizes(similarity, 2).tolist() == [0.0, 2.0, 0.0, 1.0]


def test_rank_prizes_break_ties_by_concept_row():
    assert rank_prizes(np.array([0.5, 0.5, 0.1]), 1).tolist() == [1.0, 0.0, 0.0]


def test_undirected_pairs_drop_reverse_duplicates_and_self_loops():
    source = np.array([0, 1, 2, 2])
    target = np.array([1, 0, 2, 3])

    assert undirected_pairs(source, target).tolist() == [[0, 1], [2, 3]]


def test_cheap_edges_keep_the_zero_prize_bridge():
    subgraph = query_subgraph(QUERY, CONCEPTS, PAIRS, k=2, edge_cost=0.1)

    # Concepts 2 (best) and 0 get prizes 2 and 1; concept 1 is the only way to join them.
    assert subgraph.concepts.tolist() == [0, 1, 2]
    assert subgraph.prizes[1] == 0.0


def test_expensive_edges_keep_only_the_best_concept():
    subgraph = query_subgraph(QUERY, CONCEPTS, PAIRS, k=2, edge_cost=10.0)

    assert subgraph.concepts.tolist() == [2]


def test_no_tree_subgraph_is_the_prized_concepts_only():
    subgraph = no_tree_subgraph(QUERY, CONCEPTS, k=2)

    assert subgraph.concepts.tolist() == [0, 2]
```

(Check by hand: cos(q, c0)=0.6, c1=−0.6, c2=0.8, c3=−0.8 → top-2 = c2 (prize 2), c0 (prize 1).)

- [ ] **Step 2: Run, expect FAIL** — `python -m uv run pytest -q tests/test_pcst.py`

- [ ] **Step 3: Implement** `src/mmxai/graph/pcst.py`

```python
"""Query-dependent concept subgraphs with PCST, as G-Retriever retrieves them (ADR-0051).

For a query q, the k concepts most similar to q (CLIP cosine) get prizes k, k-1, ..., 1;
every edge of the concept graph costs the same constant C. A Prize-Collecting Steiner
Tree keeps a connected set of concepts that collects much prize for little edge cost,
so a zero-prize concept is kept when it is the cheap way to join two prized ones (a
"bridge"). This mirrors retrieval_via_pcst of the G-Retriever code; our edges have no
text, so edges get no prizes (the one difference, written in ADR-0051).
"""

from dataclasses import dataclass

import numpy as np
from pcst_fast import pcst_fast

# pcst_fast arguments used by the G-Retriever code: unrooted, one tree, GW pruning.
UNROOTED = -1
NUM_TREES = 1
PRUNING = "gw"
QUIET = 0


@dataclass(frozen=True)
class QuerySubgraph:
    """concepts: sorted concept rows of S_q; prizes: the prize of every concept row."""

    concepts: np.ndarray  # int64
    tree_pairs: np.ndarray  # int64, shape (num_tree_edges, 2), edges of the tree
    prizes: np.ndarray  # float64, shape (num_concepts,)


def undirected_pairs(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """(a, b) with a < b, one per unordered pair, sorted. PCST works on undirected edges."""
    pairs = set()
    for a, b in zip(source.tolist(), target.tolist(), strict=True):
        if a != b:
            pairs.add((min(a, b), max(a, b)))
    return np.array(sorted(pairs), dtype=np.int64).reshape(-1, 2)


def unit_rows(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.linalg.norm(matrix, axis=-1, keepdims=True)


def rank_prizes(similarity: np.ndarray, k: int) -> np.ndarray:
    """Prize k for the most similar concept, k-1 for the next, ..., 0 outside the top k.

    Ties are broken by concept row (stable sort) so the result is deterministic.
    """
    order = np.argsort(-similarity, kind="stable")
    prizes = np.zeros(len(similarity), dtype=np.float64)
    top = order[: min(k, len(similarity))]
    prizes[top] = np.arange(len(top), 0, -1, dtype=np.float64)
    return prizes


def query_prizes(query: np.ndarray, concept_vectors: np.ndarray, k: int) -> np.ndarray:
    similarity = unit_rows(concept_vectors) @ (query / np.linalg.norm(query))
    return rank_prizes(similarity, k)


def query_subgraph(
    query: np.ndarray, concept_vectors: np.ndarray, pairs: np.ndarray, k: int, edge_cost: float
) -> QuerySubgraph:
    """S_q: the PCST tree on the concept graph for this query."""
    prizes = query_prizes(query, concept_vectors, k)
    costs = np.full(len(pairs), edge_cost, dtype=np.float64)
    vertices, edge_rows = pcst_fast(pairs, prizes, costs, UNROOTED, NUM_TREES, PRUNING, QUIET)
    tree_pairs = pairs[edge_rows]
    # As in the G-Retriever code: the subgraph is the chosen vertices plus both ends of
    # every chosen edge.
    concepts = np.unique(np.concatenate([vertices, tree_pairs.ravel()])).astype(np.int64)
    return QuerySubgraph(concepts, tree_pairs.astype(np.int64), prizes)


def no_tree_subgraph(query: np.ndarray, concept_vectors: np.ndarray, k: int) -> QuerySubgraph:
    """Ablation 1 (ADR-0051): only the prized concepts, no bridges, no tree."""
    prizes = query_prizes(query, concept_vectors, k)
    concepts = np.flatnonzero(prizes > 0).astype(np.int64)
    return QuerySubgraph(concepts, np.zeros((0, 2), dtype=np.int64), prizes)
```

If `pcst_fast` rejects an empty `pairs` array, make `query_subgraph` return `no_tree_subgraph(...)` restricted to the single best concept when `len(pairs) == 0`, and add a test for that case.

- [ ] **Step 4: Run, expect PASS** → 6 passed. If `test_expensive_edges_keep_only_the_best_concept` fails because GW pruning returns a different single vertex or several, print `query_subgraph(...)` and report the actual library behaviour before changing the test (do not guess).
- [ ] **Step 5:** ruff check + format; **Commit** `feat(graph): PCST concept subgraph of a query (G-Retriever)`.

---

### Task 3: Degree-preserving shuffle of concept pairs (ablation 2)

**Files:** Modify `src/mmxai/graph/shuffle.py`; Test: the existing shuffle test file (find it with `ls tests | grep shuffle`), or `tests/test_pcst.py` if none.

- [ ] **Step 1: Failing test**

```python
def test_shuffle_concept_pairs_keeps_degrees_and_changes_edges():
    from mmxai.graph.shuffle import shuffle_concept_pairs

    pairs = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [4, 6]])

    shuffled = shuffle_concept_pairs(pairs, seed=0)

    def degrees(edge_pairs):
        return np.bincount(edge_pairs.ravel(), minlength=8).tolist()

    assert degrees(shuffled) == degrees(pairs)
    assert {tuple(p) for p in shuffled.tolist()} != {tuple(p) for p in pairs.tolist()}
    assert shuffle_concept_pairs(pairs, seed=0).tolist() == shuffled.tolist()  # deterministic
```

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** in `shuffle.py`:

```python
def shuffle_concept_pairs(pairs: np.ndarray, seed: int) -> np.ndarray:
    """Ablation 2 of ADR-0051: rewire concept-concept edges, keeping every concept's degree.

    Double edge swaps (a-b, c-d -> a-d, c-b) keep how many edges each concept has but
    destroy which concepts co-occur. If PCST helps only because of the real co-occurrence
    structure, scores should drop on the shuffled graph.
    """
    import networkx as nx

    graph = nx.Graph()
    graph.add_edges_from(pairs.tolist())
    # About 10 swaps per edge mixes the graph well; the count is a fixed rule, not tuned.
    swaps = 10 * graph.number_of_edges()
    nx.double_edge_swap(graph, nswap=swaps, max_tries=100 * swaps, seed=seed)
    shuffled = sorted((min(a, b), max(a, b)) for a, b in graph.edges())
    return np.array(shuffled, dtype=np.int64).reshape(-1, 2)
```

(Move `import networkx as nx` to the top of the file if `shuffle.py` already imports graph libraries at module level; follow the file's existing style.)
- [ ] **Step 4: Run, expect PASS**; ruff; **Commit** `feat(graph): degree-preserving shuffle of concept pairs`.

---

### Task 4: Overlap score and explanation

**Files:** Create `src/mmxai/models/subgraph_scorer.py`, Test `tests/test_subgraph_scorer.py`

- [ ] **Step 1: Failing tests** `tests/test_subgraph_scorer.py`

```python
"""Stage-1 G-Retriever score: share of the query subgraph an image touches (ADR-0051)."""

import numpy as np

from mmxai.graph.contains import ContainsGraph
from mmxai.graph.pcst import QuerySubgraph
from mmxai.models.subgraph_scorer import (
    explain_overlap,
    image_concept_sets,
    overlap_scores,
)


def contains_graph() -> ContainsGraph:
    # Images 10, 11, 12; concepts 0..3. 10 has {0, 1}, 11 has {2}, 12 has nothing.
    return ContainsGraph(
        image_ids=np.array([10, 11, 12]),
        image_split=np.array(["test", "test", "test"]),
        concept_keys=np.array(["a.n.01", "b.n.01", "c.n.01", "d.n.01"]),
        edge_image=np.array([0, 0, 1]),
        edge_concept=np.array([0, 1, 2]),
    )


def subgraph(concepts: list[int], prizes: list[float]) -> QuerySubgraph:
    return QuerySubgraph(
        np.array(concepts), np.zeros((0, 2), dtype=np.int64), np.array(prizes)
    )


def test_image_concept_sets():
    sets = image_concept_sets(contains_graph())

    assert sets == {10: {0, 1}, 11: {2}, 12: set()}


def test_overlap_counts_bridges_like_prized_concepts():
    sets = image_concept_sets(contains_graph())
    # S_q = {0, 1, 2}; concept 1 is a bridge (prize 0) but still counts.
    subgraphs = [subgraph([0, 1, 2], [2.0, 0.0, 1.0, 0.0])]
    candidate_ids = np.array([[10, 11, 12]])

    scores = overlap_scores(candidate_ids, subgraphs, sets)

    assert np.allclose(scores, [[2 / 3, 1 / 3, 0.0]])


def test_explanation_is_exactly_what_the_score_counts():
    sets = image_concept_sets(contains_graph())
    query_subgraph = subgraph([0, 1, 2], [2.0, 0.0, 1.0, 0.0])
    graph = contains_graph()

    record = explain_overlap(10, query_subgraph, sets, graph.concept_keys)

    assert record["matched"] == [
        {"concept": "a.n.01", "bridge": False},
        {"concept": "b.n.01", "bridge": True},
    ]
    assert record["score"] == 2 / 3
    # Fidelity by construction: dropping one matched concept lowers the score by 1/|S_q|.
    assert record["score_without_each"] == [1 / 3, 1 / 3]
```

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** `src/mmxai/models/subgraph_scorer.py`

```python
"""Stage 1 of the G-Retriever reranker: how much of S_q does an image touch? (ADR-0051)

g(q, i) = |N(i) ∩ S_q| / |S_q|, where N(i) are the concepts of image i (CLIP top-16
edges, ADR-0046, same for every split, no annotation) and S_q is the PCST subgraph of
the query. Every concept of S_q counts the same, bridges included: summing prizes
instead would give bridges (prize 0) no weight and lose the indirect-query mechanism.
No training. The explanation is exactly the set the score counts (ADR-0016).
"""

import numpy as np

from mmxai.graph.contains import ContainsGraph
from mmxai.graph.pcst import QuerySubgraph


def image_concept_sets(graph: ContainsGraph) -> dict[int, set[int]]:
    """Image id -> concept rows it contains (every image appears, possibly with no concept)."""
    sets: dict[int, set[int]] = {}
    for image_id in graph.image_ids.tolist():
        sets[image_id] = set()
    for image_row, concept_row in zip(
        graph.edge_image.tolist(), graph.edge_concept.tolist(), strict=True
    ):
        sets[graph.image_ids[image_row].item()].add(concept_row)
    return sets


def overlap(concepts: set[int], subgraph: QuerySubgraph) -> set[int]:
    return concepts & set(subgraph.concepts.tolist())


def overlap_scores(
    candidate_ids: np.ndarray, subgraphs: list[QuerySubgraph], sets: dict[int, set[int]]
) -> np.ndarray:
    """Score of every candidate, shape (num_queries, top_k); row q uses subgraphs[q]."""
    if len(candidate_ids) != len(subgraphs):
        raise ValueError(f"{len(candidate_ids)} queries but {len(subgraphs)} subgraphs")
    scores = np.zeros(candidate_ids.shape, dtype=np.float32)
    for query, subgraph in enumerate(subgraphs):
        size = len(subgraph.concepts)
        if size == 0:
            continue
        for column, image_id in enumerate(candidate_ids[query].tolist()):
            scores[query, column] = len(overlap(sets[image_id], subgraph)) / size
    return scores


def explain_overlap(
    image_id: int, subgraph: QuerySubgraph, sets: dict[int, set[int]], concept_keys: np.ndarray
) -> dict:
    """The concepts that made the score, which of them are bridges, and the score without each."""
    matched = sorted(overlap(sets[image_id], subgraph))
    size = len(subgraph.concepts)
    score = len(matched) / size if size else 0.0
    items = []
    without_each = []
    for concept_row in matched:
        items.append(
            {"concept": str(concept_keys[concept_row]), "bridge": subgraph.prizes[concept_row] == 0}
        )
        without_each.append((len(matched) - 1) / size)
    return {"matched": items, "score": score, "score_without_each": without_each}
```

Note `bridge` must be a plain Python `bool` (convert with `bool(...)`) so the record is JSON-serialisable; adjust the test if needed.
- [ ] **Step 4: Run, expect PASS**; ruff; **Commit** `feat(models): stage-1 overlap score and explanation for PCST subgraphs`.

---

### Task 5: Select on val, evaluate test once

**Files:** Create `src/mmxai/eval/pcst_rerank.py`, Test `tests/test_pcst_rerank.py`

Reuse from `mmxai.eval.graph_rerank`: `load_inputs` (duck-typed: it reads `config.dataset, features_dir, graph_dir, split_file, concept_edges_dir`, which `PcstRerankConfig` also has — say so in a comment), `split_candidates` (reads `top_k`, `same_text_threshold`), `write_run`-style helpers (`make_run_dir`, `write_run_files`, `collect_env`, `test_run_exists` — import them from where `graph_rerank.py` imports them), `library_versions`. From `mmxai.rerank.fusion`: `sweep_alpha`, `choose_alpha`, `fuse_scores`, `rerank_ranks`. From `mmxai.graph.contains`: `split_subgraph`. From `mmxai.eval.metrics`: `compute_metrics`. Read `src/mmxai/eval/graph_rerank.py` first and copy its patterns (run facts, refusing a second test run).

- [ ] **Step 1: Failing tests** `tests/test_pcst_rerank.py` — keep them on small pure functions (no artifacts):

```python
"""Selection logic of the PCST reranker (ADR-0051). Synthetic numbers only."""

from mmxai.eval.pcst_rerank import choose_setting


def test_choose_setting_takes_the_best_metric():
    results = [
        {"prize_k": 3, "edge_cost": 0.5, "alpha": 0.8, "metric": 0.12},
        {"prize_k": 5, "edge_cost": 0.5, "alpha": 0.7, "metric": 0.13},
    ]

    assert choose_setting(results)["prize_k"] == 5


def test_ties_prefer_larger_alpha_then_smaller_k_then_larger_cost():
    results = [
        {"prize_k": 5, "edge_cost": 0.5, "alpha": 0.8, "metric": 0.13},
        {"prize_k": 3, "edge_cost": 1.0, "alpha": 0.9, "metric": 0.13},
        {"prize_k": 3, "edge_cost": 0.5, "alpha": 0.9, "metric": 0.13},
    ]

    chosen = choose_setting(results)

    assert (chosen["alpha"], chosen["prize_k"], chosen["edge_cost"]) == (0.9, 3, 1.0)
```

Add one more test that `load_pcst_rerank_config` rejects `subgraph_mode` other than `pcst` / `top_k_only` (write a YAML in `tmp_path` from a dict of valid values).

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** `src/mmxai/eval/pcst_rerank.py` with:

```python
@dataclass(frozen=True)
class PcstRerankConfig:
    """Settings of one G-Retriever stage-1 experiment (configs/clip_pcst_vg*.yaml)."""

    experiment: str
    dataset: str
    split_file: str
    features_dir: str
    graph_dir: str
    concept_edges_dir: str
    subgraph_mode: str  # "pcst" (main) or "top_k_only" (ablation 1)
    shuffle_concept_edges: bool  # ablation 2; the shuffle seed is the run seed
    top_k: int
    recall_ks: list[int]
    same_text_threshold: float
    prize_k_grid: list[int]
    edge_cost_grid: list[float]
    alpha_grid: list[float]
    selection_metric: str
    selection_dir: str  # where selection.json is written (artifacts/, gitignored)
    output_dir: str
```

Functions (each small, with a docstring):
- `load_pcst_rerank_config(path)` — `load_yaml_dataclass` + check `subgraph_mode in ("pcst", "top_k_only")`.
- `concept_pairs(inputs, config, seed)` — `undirected_pairs(edges.edge_source, edge_target)` from `inputs.concept_edges`; if `config.shuffle_concept_edges`, `shuffle_concept_pairs(pairs, seed)`.
- `split_subgraphs(candidates, inputs, pairs, config, prize_k, edge_cost)` — one `QuerySubgraph` per query (`query_subgraph` or `no_tree_subgraph` by mode); log progress every 1,000 queries with elapsed seconds (val has ~25,000 queries; the time is not known yet).
- `choose_setting(results)` — best `metric`; ties: larger `alpha`, then smaller `prize_k`, then larger `edge_cost` (fewer, smaller subgraphs: closer to CLIP). Comment the reason.
- `select_on_val(config, seed, log=print) -> Path` — load inputs; val candidates; `image_concept_sets(split_subgraph(inputs.graph, "val"))`; for each `prize_k`, `edge_cost`: subgraphs → `overlap_scores` → `sweep_alpha` → best alpha via `choose_alpha(sweep, config.selection_metric)`; collect results; `choose_setting`; write `<selection_dir>/seed<k>/selection.json` (`prize_k`, `edge_cost`, `alpha`, all grid results, git state, split sha256) and a val run to `results/runs` (split "val"). Refuse to overwrite an existing selection file.
- `evaluate_test(config, seed, log=print) -> Path` — refuse a second test run (`test_run_exists`); read selection; test candidates, subgraphs with the chosen `prize_k`, `edge_cost`; `fuse_scores` with chosen `alpha`; `compute_metrics(rerank_ranks(...))`; add run facts (split, split_sha256, seed, chosen values, subgraph_mode, shuffle flag, graph/concept edge dirs, git commit of graph meta); write the run. Chooses nothing.

Use `log` for progress like `graph_rerank.py`.
- [ ] **Step 4: Run, expect PASS** (3 tests) plus `python -m uv run pytest -q tests/test_pcst.py tests/test_subgraph_scorer.py tests/test_pcst_rerank.py`. Also `python -m uv run python -c "import mmxai.eval.pcst_rerank"` (imports resolve). Ruff.
- [ ] **Step 5: Commit** `feat(eval): select PCST reranker on val and evaluate test once`.

---

### Task 6: CLIs and configs

**Files:** `scripts/select_pcst_rerank.py`, `scripts/evaluate_pcst_rerank.py`, `configs/clip_pcst_vg.yaml`, `configs/clip_pcst_vg_no_tree.yaml`, `configs/clip_pcst_vg_shuffled.yaml`, `configs/README.md`, `scripts/README.md`, test in `tests/test_pcst_rerank.py`

- [ ] **Step 1: Failing test** (append): load all three configs; assert they differ only in `experiment`, `subgraph_mode`, `shuffle_concept_edges`, `selection_dir`; assert main config has `subgraph_mode == "pcst"` and `shuffle_concept_edges is False`. Build paths from `Path(__file__).parent.parent`.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Configs.** `configs/clip_pcst_vg.yaml` (copy paths from `configs/clip_cooc_graph_vg_lr1e4.yaml`):

```yaml
# G-Retriever stage 1 on Visual Genome (ADR-0051, status: Proposed).
# How to run: scripts/select_pcst_rerank.py (val), then scripts/evaluate_pcst_rerank.py (test, once).
experiment: clip_pcst_vg
dataset: visual_genome
split_file: splits/visual_genome/split_seed0.json
features_dir: artifacts/features/visual_genome/ViT-B-32__laion2b_s34b_b79k
graph_dir: artifacts/graphs/visual_genome/contains_clip_top16_v1
# Decision (ADR-0051): PCST runs on the train-counted PMI co-occurrence edges (ADR-0048).
concept_edges_dir: artifacts/graphs/visual_genome/concept_cooccurrence_pmi10_top10_v1
subgraph_mode: pcst
shuffle_concept_edges: false
top_k: 100
recall_ks: [1, 5, 10]
same_text_threshold: 0.9999
# Decision (ADR-0051): grid includes the G-Retriever SceneGraphs values (topk=3, cost_e=0.5).
#   Chosen on val only.
prize_k_grid: [3, 5, 10]
edge_cost_grid: [0.25, 0.5, 1.0]
alpha_grid: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
selection_metric: MRR@100
selection_dir: artifacts/models/visual_genome/clip_pcst_vg
output_dir: results/runs
```

`clip_pcst_vg_no_tree.yaml`: same, `experiment: clip_pcst_vg_no_tree`, `subgraph_mode: top_k_only` (comment: ablation 1, ADR-0051), `selection_dir: .../clip_pcst_vg_no_tree`. `clip_pcst_vg_shuffled.yaml`: same, `experiment: clip_pcst_vg_shuffled`, `shuffle_concept_edges: true` (comment: ablation 2, seed = run seed), `selection_dir: .../clip_pcst_vg_shuffled`. Note: `edge_cost_grid` is irrelevant for `top_k_only`; keep it identical anyway (the test checks equality) and say so in a comment.

- [ ] **Step 4: CLIs.** Both take `--config` and `--seed` (required int). Docstrings in the style of `scripts/train_graph_rerank.py` / `scripts/evaluate_graph_rerank.py` (read them): what it does, prerequisites (features from Kaggle per ADR-0041, `scripts/build_graph.py` with `configs/graph_vg_clip_topk.yaml`, `scripts/build_concept_edges.py` with `configs/concept_edges_vg_cooccurrence.yaml`, `uv sync --group ml --group graph`), run commands for seeds 0 1 2, output. Use raw docstrings if lines end with `\`. `select_pcst_rerank.py` calls `select_on_val`; `evaluate_pcst_rerank.py` calls `evaluate_test`. Add one row each to `configs/README.md` and `scripts/README.md`.
- [ ] **Step 5: Run, expect PASS**; `--help` of both scripts prints the docstring; ruff. **Commit** `feat(eval): CLIs and configs for the PCST reranker and its ablations`.

---

### Task 7: Docs and checks

**Files:** `docs/decisions/0051-g-retriever-chon-do-thi-con-pcst-de-xep-hang-lai.md`, a guide section, `docs/ai-usage-log.md`

- [ ] **Step 1:** ADR-0051: add a "Code:" line under the decision listing the files of this plan.
- [ ] **Step 2:** Add a section "G-Retriever bậc 1 (ADR-0051)" to `docs/guides/huan-luyen-va-xep-hang-lai.md` (Vietnamese, beginner-first; read the guide's style first): ý tưởng (giải thưởng, phí, cầu nối) bằng một ví dụ tổng hợp từ `tests/test_pcst.py` (ghi rõ là dữ liệu giả); công thức `g`; thứ tự chạy (prerequisites → select seed 0 1 2 → evaluate test seed 0 1 2 → summarize); 2 ablation; rò rỉ (PMI train, contains CLIP top-k, chỉ val để chọn); explanation = `explain_overlap`.
- [ ] **Step 3:** One row in `docs/ai-usage-log.md` (self-checked = `Chưa`).
- [ ] **Step 4:** `python -m uv run ruff check`, `python -m uv run ruff format --check` (only .md plans may be listed), `python -m uv run pytest -q` (only the 2 known failures). Paste tails in the report.
- [ ] **Step 5: Commit** `docs: G-Retriever stage 1 guide and ADR code line`.

---

## Out of scope (needs artifacts or later plans)
- Real val selection and test runs: need `artifacts/features/visual_genome/...` (Kaggle, ADR-0041), the contains graph and the PMI concept edges built with the existing scripts. Not present on this machine yet.
- Stage 2 (GNN on S_q), ablation 3 (ConceptNet edges), ablation 4 (drop `kg_evidence` edges), evaluation on the query benchmark v2: plan 2.
- Summary tables in `results/tables/` (existing `scripts/summarize_runs.py` may need the new experiment names).
