# G-Retriever Stage 2 (GraphSAGE on the query subgraph) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a one-layer GraphSAGE whose image vector only aggregates the concepts that lie in the query's PCST subgraph (`h_i(q) = W_self·x_i + W_nb·mean(x_c, c ∈ N(i) ∩ S_q) + b`), choose epoch and `alpha` on val, evaluate test once; plus "no tree" and "shuffled PMI" ablations (ADR-0051, "Thiết kế bậc 2").

**Architecture:** `src/mmxai/graph/pcst_cache.py` stores one `S_q` per text (packed offsets/values) with a checked `meta.json`. `src/mmxai/models/subgraph_sage.py` holds the model, the masked concept mean, the batch loss and candidate scoring. `src/mmxai/eval/pcst_sage_rerank.py` prepares caches, trains/selects on val, evaluates test once, reusing `graph_rerank.build_training_data`, `train.fit` (given a loss function), `fusion.sweep_alpha/choose_alpha`, and the stage-1 helpers in `pcst_rerank.py`.

**Tech Stack:** Python 3.11, numpy 1.26, torch, pcst-fast (already added), pytest, ruff.

**Spec:** `docs/decisions/0051-g-retriever-chon-do-thi-con-pcst-de-xep-hang-lai.md`, section "Thiết kế bậc 2 (2026-11-16)": fixed `k = 3`, `C = 0.5`; empty `N(i) ∩ S_q` → mean = 0; training as ADR-0047 / `configs/clip_graph_vg_lr1e4.yaml`; 3 seeds; exact occlusion explanation.

**Repo rules:** branch `feat/g-retriever-stage2` (checked out, stacked on `feat/g-retriever-pcst-rerank`). Commands as `python -m uv run ...`; format with `python -m uv run ruff format src tests scripts`. English code/comments, Vietnamese docs, simple viva-explainable code, no hard-coded paths/hyperparameters. Commit after each task, never push. Never run anything on test in this plan; no real data on this machine (tests are synthetic).

**Known pre-existing failures (ignore):** 2 tests in `tests/test_notebooks_match_src.py` (Windows symlink).

**Read before starting:** `src/mmxai/eval/graph_rerank.py` (`build_training_data`, `build_split_graph`, `train_and_select`, `graph_only_val_score`, `MODEL_FILE`, `SELECTION_FILE`), `src/mmxai/models/train.py` (`TrainingData`, `sample_negatives`, `batch_loss`, `train_one_epoch`, `fit`), `src/mmxai/models/sage.py` (`ConceptToImageSage` init), `src/mmxai/eval/pcst_rerank.py` (`load_checked_inputs`, `concept_pairs`, `top_concepts_per_query`, `split_subgraphs`, `run_facts`, `write_run`, `has_test_run`), `src/mmxai/graph/pcst.py`, `src/mmxai/graph/contains.py` (`ContainsGraph`, `split_subgraph`).

---

## File map

| File | Responsibility |
|---|---|
| `src/mmxai/graph/pcst_cache.py` (new) | Pack/unpack/pad subgraphs; save/load cache with meta check |
| `src/mmxai/models/train.py` (modify) | `train_one_epoch` and `fit` accept a `loss_fn` (default `batch_loss`, behaviour unchanged) |
| `src/mmxai/models/subgraph_sage.py` (new) | `image_concept_matrix`, `masked_concept_mean`, `SubgraphSage`, `SubgraphTrainingData`, `subgraph_batch_loss`, `candidate_subgraph_scores`, `explain_candidate` |
| `src/mmxai/eval/pcst_sage_rerank.py` (new) | Config, cache preparation, train/select on val, evaluate test once |
| `scripts/prepare_pcst_cache.py`, `scripts/train_pcst_sage.py`, `scripts/evaluate_pcst_sage.py` (new) | Thin CLIs |
| `configs/clip_pcst_sage_vg.yaml`, `..._no_tree.yaml`, `..._shuffled.yaml` (new) | Main + 2 ablations |
| `tests/test_pcst_cache.py`, `tests/test_subgraph_sage.py`, `tests/test_pcst_sage_rerank.py` (new), `tests/test_models_train*.py` (extend existing train test file) | Synthetic tests |

---

### Task 1: Subgraph cache

**Files:** Create `src/mmxai/graph/pcst_cache.py`, Test `tests/test_pcst_cache.py`

- [x] **Step 1: Failing tests**

```python
"""Packed per-text PCST subgraphs and their checked cache (ADR-0051, stage 2)."""

import numpy as np
import pytest

from mmxai.graph.pcst_cache import (
    load_cache,
    pack_subgraphs,
    padded_rows,
    save_cache,
    unpack_row,
)


def test_pack_and_unpack():
    offsets, values = pack_subgraphs([np.array([2, 5]), np.array([], dtype=np.int64), np.array([7])])

    assert offsets.tolist() == [0, 2, 2, 3]
    assert unpack_row(offsets, values, 0).tolist() == [2, 5]
    assert unpack_row(offsets, values, 1).tolist() == []


def test_padded_rows_use_minus_one():
    offsets, values = pack_subgraphs([np.array([2, 5]), np.array([7])])

    assert padded_rows(offsets, values).tolist() == [[2, 5], [7, -1]]


def test_save_refuses_overwrite_and_load_checks_meta(tmp_path):
    offsets, values = pack_subgraphs([np.array([1])])
    meta = {"prize_k": 3, "edge_cost": 0.5, "num_texts": 1}
    save_cache(tmp_path, "val", offsets, values, meta)

    with pytest.raises(FileExistsError):
        save_cache(tmp_path, "val", offsets, values, meta)
    loaded_offsets, loaded_values = load_cache(tmp_path, "val", meta)
    assert loaded_values.tolist() == [1]
    with pytest.raises(ValueError, match="prize_k"):
        load_cache(tmp_path, "val", {"prize_k": 10, "edge_cost": 0.5, "num_texts": 1})
```

- [x] **Step 2: Run, expect FAIL** — `python -m uv run pytest -q tests/test_pcst_cache.py`
- [x] **Step 3: Implement** `src/mmxai/graph/pcst_cache.py`

```python
"""One PCST subgraph S_q per text, stored once and reused by every training epoch and seed.

Running PCST takes milliseconds per text and stage 2 needs S_q for every train text
at every epoch, so subgraphs are computed once and saved (ADR-0051, stage 2). They are
stored packed: values holds every concept row one after another, offsets[t] is where
text t starts. meta.json records how they were made; loading with other settings fails
instead of silently mixing caches.
"""

import json
from pathlib import Path

import numpy as np

PAD = -1


def pack_subgraphs(rows: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros(len(rows) + 1, dtype=np.int64)
    for number, row in enumerate(rows):
        offsets[number + 1] = offsets[number] + len(row)
    values = np.concatenate([np.asarray(row, dtype=np.int64) for row in rows]) if rows else (
        np.zeros(0, dtype=np.int64)
    )
    return offsets, values.astype(np.int64)


def unpack_row(offsets: np.ndarray, values: np.ndarray, text: int) -> np.ndarray:
    return values[offsets[text] : offsets[text + 1]]


def padded_rows(offsets: np.ndarray, values: np.ndarray) -> np.ndarray:
    """(num_texts, longest) matrix of concept rows, PAD (-1) after each text's last concept."""
    lengths = np.diff(offsets)
    width = max(int(lengths.max()), 1) if len(lengths) else 1
    matrix = np.full((len(lengths), width), PAD, dtype=np.int64)
    for text, length in enumerate(lengths.tolist()):
        matrix[text, :length] = unpack_row(offsets, values, text)
    return matrix


def save_cache(cache_dir, split: str, offsets: np.ndarray, values: np.ndarray, meta: dict) -> None:
    cache_dir = Path(cache_dir)
    data_path = cache_dir / f"{split}.npz"
    if data_path.exists():
        raise FileExistsError(f"{data_path} exists; delete it to recompute")
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(data_path, offsets=offsets, values=values)
    with open(cache_dir / f"{split}.meta.json", "w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2, sort_keys=True)


def load_cache(cache_dir, split: str, expected_meta: dict) -> tuple[np.ndarray, np.ndarray]:
    """Load a cache and refuse it if any expected meta value differs."""
    cache_dir = Path(cache_dir)
    with open(cache_dir / f"{split}.meta.json", encoding="utf-8") as file:
        meta = json.load(file)
    for key, value in expected_meta.items():
        if meta.get(key) != value:
            raise ValueError(f"cache {cache_dir}/{split}: {key} is {meta.get(key)!r}, expected {value!r}")
    data = np.load(cache_dir / f"{split}.npz")
    return data["offsets"], data["values"]
```

Rewrite the `values = ... if rows else ...` line as a plain `if`/`else` block (readability rule). Wrap lines > 100 chars.
- [x] **Step 4: Run, expect PASS**; ruff. **Commit** `feat(graph): packed cache of per-text PCST subgraphs`.

---

### Task 2: Let `fit` take a loss function

**Files:** Modify `src/mmxai/models/train.py`; Test: the existing train test file (`ls tests | grep -i train`).

- [x] **Step 1: Failing test** — append:

```python
def test_fit_uses_the_given_loss_function():
    calls = []

    def recording_loss(model, data, text_rows, num_negatives, temperature, generator):
        calls.append(len(text_rows))
        return batch_loss(model, data, text_rows, num_negatives, temperature, generator)

    # reuse this file's tiny TrainingData / model builders (read the file for their names)
    ...
```

Fill `...` with the file's existing tiny-data helper (read the file; use the same builder the current `fit` test uses), call `fit(..., loss_fn=recording_loss)` for 1 epoch and assert `calls` is not empty. Do not invent a new builder if one exists.
- [x] **Step 2: Run, expect FAIL** (unexpected keyword `loss_fn`).
- [x] **Step 3: Implement** — add a keyword parameter `loss_fn: Callable = batch_loss` to `train_one_epoch` (used instead of the direct `batch_loss` call) and to `fit` (passed through). Type hints: `Callable[..., torch.Tensor]`. Docstring line: "loss_fn: batch loss to minimise; the default is the ADR-0046 loss. Stage 2 of ADR-0051 passes its own." No other behaviour change.
- [x] **Step 4: Run** the train test file and `python -m uv run pytest -q tests/test_graph_rerank*.py` (or whatever tests cover `graph_rerank`) → all pass. Ruff. **Commit** `refactor(models): fit accepts a batch loss function`.

---

### Task 3: Masked concept mean and the model

**Files:** Create `src/mmxai/models/subgraph_sage.py`, Test `tests/test_subgraph_sage.py`

- [x] **Step 1: Failing tests**

```python
"""Stage-2 G-Retriever model: GraphSAGE restricted to the query subgraph (ADR-0051)."""

import numpy as np
import torch

from mmxai.graph.contains import ContainsGraph
from mmxai.models.subgraph_sage import SubgraphSage, image_concept_matrix, masked_concept_mean

CONCEPT_X = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])


def test_image_concept_matrix_pads_with_minus_one():
    graph = ContainsGraph(
        image_ids=np.array([10, 11]),
        image_split=np.array(["train", "train"]),
        concept_keys=np.array(["a.n.01", "b.n.01", "c.n.01"]),
        edge_image=np.array([0, 0, 1]),
        edge_concept=np.array([0, 2, 1]),
    )

    assert image_concept_matrix(graph).tolist() == [[0, 2], [1, -1]]


def test_masked_mean_keeps_only_concepts_in_the_subgraph():
    image_concepts = torch.tensor([[[0, 2], [1, -1]]])  # 1 query, 2 images
    subgraph = torch.tensor([[2, 1]])  # S_q = {2, 1}

    mean = masked_concept_mean(CONCEPT_X, image_concepts, subgraph)

    assert torch.allclose(mean[0, 0], CONCEPT_X[2])  # image 0 keeps only concept 2
    assert torch.allclose(mean[0, 1], CONCEPT_X[1])  # image 1 keeps concept 1


def test_no_concept_in_the_subgraph_gives_zero():
    mean = masked_concept_mean(CONCEPT_X, torch.tensor([[[0, -1]]]), torch.tensor([[1, -1]]))

    assert torch.equal(mean, torch.zeros(1, 1, 2))


def test_untrained_model_returns_the_clip_vector():
    model = SubgraphSage(2)
    image_x = torch.tensor([[[0.3, 0.4]]])

    h = model(image_x, torch.tensor([[[5.0, 5.0]]]))

    assert torch.allclose(h, image_x)
```

- [x] **Step 2: Run, expect FAIL.**
- [x] **Step 3: Implement** (start of `subgraph_sage.py`):

```python
"""Stage 2 of the G-Retriever reranker (ADR-0051): GraphSAGE on the query subgraph.

    m_i(q) = mean of x_c over c in N(i) ∩ S_q   (zero vector when the intersection is empty)
    h_i(q) = W_self·x_i + W_nb·m_i(q) + b,       score = cos(q, h_i(q))

Same parameters and start as ADR-0046 (W_self = I, W_nb = 0, b = 0), so the untrained
model scores exactly like CLIP. The only change: an image hears only the concepts that
PCST put in the subgraph of this query, so h depends on the query.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from mmxai.graph.contains import ContainsGraph

PAD = -1


def image_concept_matrix(graph: ContainsGraph) -> np.ndarray:
    """(num_images, max_degree) concept rows of each image (graph row order), padded with -1."""
    per_image: list[list[int]] = [[] for _ in range(len(graph.image_ids))]
    for image_row, concept_row in zip(graph.edge_image.tolist(), graph.edge_concept.tolist(), strict=True):
        per_image[image_row].append(concept_row)
    width = max([len(rows) for rows in per_image] + [1])
    matrix = np.full((len(per_image), width), PAD, dtype=np.int64)
    for image_row, rows in enumerate(per_image):
        matrix[image_row, : len(rows)] = sorted(rows)
    return matrix


def masked_concept_mean(
    concept_x: torch.Tensor, image_concepts: torch.Tensor, subgraph: torch.Tensor
) -> torch.Tensor:
    """m for every (query, image): image_concepts (Q, M, D), subgraph (Q, L) -> (Q, M, dim).

    A concept counts when it is one of the image's concepts AND in the query's S_q.
    Padding (-1) never matches a real concept row.
    """
    real = image_concepts >= 0
    in_subgraph = (image_concepts[..., None] == subgraph[:, None, None, :]).any(dim=-1)
    keep = (real & in_subgraph).float()  # (Q, M, D)
    vectors = concept_x[image_concepts.clamp(min=0)]  # (Q, M, D, dim); padded rows masked out
    total = (vectors * keep[..., None]).sum(dim=2)
    count = keep.sum(dim=2, keepdim=True)
    return total / count.clamp(min=1.0)  # empty -> 0 / 1 = 0


class SubgraphSage(nn.Module):
    """h = W_self x + W_nb m + b. Starts as CLIP (W_self = I, W_nb = 0, b = 0)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.self_weight = nn.Linear(dim, dim, bias=False)
        self.neighbor_weight = nn.Linear(dim, dim, bias=True)
        # Decision (ADR-0046, reused by ADR-0051 stage 2): start exactly at CLIP.
        #   Why: training only moves away from the baseline when that lowers the loss.
        with torch.no_grad():
            self.self_weight.weight.copy_(torch.eye(dim))
            self.neighbor_weight.weight.zero_()
            self.neighbor_weight.bias.zero_()

    def forward(self, image_x: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
        return self.self_weight(image_x) + self.neighbor_weight(mean)
```

Note the memory of `vectors` is `Q × M × D × dim` floats; callers chunk (Task 5). Wrap long lines; keep `zip(... strict=True)` lines ≤ 100 chars.
- [x] **Step 4: Run, expect PASS** (4 tests); ruff. **Commit** `feat(models): GraphSAGE restricted to the query subgraph (G-Retriever stage 2)`.

---

### Task 4: Training data and batch loss

**Files:** Modify `src/mmxai/models/subgraph_sage.py`, Test `tests/test_subgraph_sage.py`

- [x] **Step 1: Failing tests** (append):

```python
from mmxai.models.subgraph_sage import SubgraphTrainingData, subgraph_batch_loss


def tiny_training_data() -> SubgraphTrainingData:
    # 3 train images; text 0 describes image 0 and its S_q = {2}; only image 0 has concept 2.
    return SubgraphTrainingData(
        image_x=torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]]),
        image_concepts=torch.tensor([[2, -1], [0, -1], [1, -1]]),
        concept_x=torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
        text_x=torch.tensor([[0.6, 0.8]]),
        text_image_row=torch.tensor([0]),
        candidate_rows=torch.tensor([[0, 1, 2]]),
        is_negative=torch.tensor([[False, True, True]]),
        text_subgraphs=torch.tensor([[2]]),
    )


def test_loss_is_finite_and_training_lowers_it():
    torch.manual_seed(0)
    data = tiny_training_data()
    model = SubgraphSage(2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    generator = torch.Generator().manual_seed(0)
    rows = torch.tensor([0])

    first = subgraph_batch_loss(model, data, rows, 2, 0.05, generator)
    for _ in range(30):
        loss = subgraph_batch_loss(model, data, rows, 2, 0.05, generator)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert torch.isfinite(first)
    assert loss.item() < first.item()
```

- [x] **Step 2: Run, expect FAIL.**
- [x] **Step 3: Implement** (append to `subgraph_sage.py`; import `sample_negatives` from `mmxai.models.train`):

```python
@dataclass
class SubgraphTrainingData:
    """Train images and texts as in models.train.TrainingData, plus each text's S_q."""

    image_x: torch.Tensor  # (num_images, dim)
    image_concepts: torch.Tensor  # (num_images, D) concept rows, -1 padded
    concept_x: torch.Tensor  # (num_concepts, dim)
    text_x: torch.Tensor  # (num_texts, dim)
    text_image_row: torch.Tensor  # (num_texts,)
    candidate_rows: torch.Tensor  # (num_texts, top_k)
    is_negative: torch.Tensor  # (num_texts, top_k)
    text_subgraphs: torch.Tensor  # (num_texts, L) S_q concept rows, -1 padded


def subgraph_batch_loss(
    model: SubgraphSage,
    data: SubgraphTrainingData,
    text_rows: torch.Tensor,
    num_negatives: int,
    temperature: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """InfoNCE as models.train.batch_loss, but each image vector depends on the text's S_q."""
    columns, valid = sample_negatives(data.is_negative[text_rows], num_negatives, generator)
    negative_rows = torch.gather(data.candidate_rows[text_rows], 1, columns)
    image_rows = torch.cat([data.text_image_row[text_rows, None], negative_rows], dim=1)

    mean = masked_concept_mean(
        data.concept_x, data.image_concepts[image_rows], data.text_subgraphs[text_rows]
    )
    h = model(data.image_x[image_rows], mean)
    query = functional.normalize(data.text_x[text_rows], dim=-1)
    cosine = (functional.normalize(h, dim=-1) * query[:, None, :]).sum(dim=-1)

    logits = cosine / temperature
    always_valid = torch.ones(len(text_rows), 1, dtype=torch.bool)
    logits = logits.masked_fill(~torch.cat([always_valid, valid], dim=1), float("-inf"))
    targets = torch.zeros(len(text_rows), dtype=torch.long)
    return functional.cross_entropy(logits, targets)
```

(Add `from dataclasses import dataclass`.) Keep the structure identical to `batch_loss` on purpose and say so in a comment, so the two can be compared line by line.
- [x] **Step 4: Run, expect PASS**; ruff. **Commit** `feat(models): stage-2 training data and batch loss`.

---

### Task 5: Candidate scores and exact explanation

**Files:** Modify `src/mmxai/models/subgraph_sage.py`, Test `tests/test_subgraph_sage.py`

- [x] **Step 1: Failing tests** (append):

```python
from mmxai.models.subgraph_sage import candidate_subgraph_scores, explain_candidate


def test_candidate_scores_match_a_manual_cosine():
    data = tiny_training_data()
    model = SubgraphSage(2)
    with torch.no_grad():
        model.neighbor_weight.weight.copy_(torch.eye(2))  # make the subgraph matter

    scores = candidate_subgraph_scores(
        model,
        query_x=data.text_x,
        query_subgraphs=data.text_subgraphs,
        candidate_rows=torch.tensor([[0, 1]]),
        image_x=data.image_x,
        image_concepts=data.image_concepts,
        concept_x=data.concept_x,
        chunk=1,
    )

    h0 = data.image_x[0] + data.concept_x[2]  # image 0 hears concept 2
    h1 = data.image_x[1]  # image 1 has no concept in S_q
    q = data.text_x[0]
    expected = torch.stack([torch.cosine_similarity(h0, q, dim=0), torch.cosine_similarity(h1, q, dim=0)])
    assert np.allclose(scores[0], expected.numpy(), atol=1e-6)


def test_explanation_drops_each_matched_concept_and_recomputes():
    data = tiny_training_data()
    model = SubgraphSage(2)
    with torch.no_grad():
        model.neighbor_weight.weight.copy_(torch.eye(2))

    record = explain_candidate(
        model, data.text_x[0], data.text_subgraphs[0], prized_rows={2},
        image_vector=data.image_x[0], image_concept_rows=data.image_concepts[0],
        concept_x=data.concept_x, concept_keys=np.array(["a.n.01", "b.n.01", "c.n.01"]),
    )

    assert [item["concept"] for item in record["matched"]] == ["c.n.01"]
    assert record["matched"][0]["bridge"] is False
    without = torch.cosine_similarity(data.image_x[0], data.text_x[0], dim=0).item()
    assert abs(record["matched"][0]["score_without"] - without) < 1e-6
```

- [x] **Step 2: Run, expect FAIL.**
- [x] **Step 3: Implement** (append):

```python
def candidate_subgraph_scores(
    model: SubgraphSage,
    query_x: torch.Tensor,
    query_subgraphs: torch.Tensor,
    candidate_rows: torch.Tensor,
    image_x: torch.Tensor,
    image_concepts: torch.Tensor,
    concept_x: torch.Tensor,
    chunk: int,
) -> np.ndarray:
    """cos(q, h_i(q)) for every candidate, shape (num_queries, top_k).

    Computed `chunk` queries at a time: the masked mean holds chunk x top_k x D x dim
    numbers (e.g. 256 x 100 x 16 x 512 floats is about 0.8 GB), so the chunk size is a
    memory setting in the config, not a hyperparameter of the method.
    """
    model.eval()
    blocks = []
    with torch.no_grad():
        for start in range(0, len(query_x), chunk):
            rows = candidate_rows[start : start + chunk]
            mean = masked_concept_mean(concept_x, image_concepts[rows], query_subgraphs[start : start + chunk])
            h = functional.normalize(model(image_x[rows], mean), dim=-1)
            query = functional.normalize(query_x[start : start + chunk], dim=-1)
            blocks.append((h * query[:, None, :]).sum(dim=-1).numpy())
    return np.concatenate(blocks) if blocks else np.zeros((0, candidate_rows.shape[1]), dtype=np.float32)


def explain_candidate(
    model: SubgraphSage,
    query_vector: torch.Tensor,
    query_subgraph: torch.Tensor,
    prized_rows: set[int],
    image_vector: torch.Tensor,
    image_concept_rows: torch.Tensor,
    concept_x: torch.Tensor,
    concept_keys: np.ndarray,
) -> dict:
    """Exact occlusion (ADR-0016): the score, and the score with each matched concept removed.

    Removing a concept from S_q and re-running the model is the real counterfactual, so the
    drop is exactly how much that concept moved this candidate's score.
    """
    def score(subgraph: torch.Tensor) -> float:
        mean = masked_concept_mean(concept_x, image_concept_rows[None, None, :], subgraph[None, :])
        h = model(image_vector[None, None, :], mean)[0, 0]
        return torch.cosine_similarity(h, query_vector, dim=0).item()

    model.eval()
    with torch.no_grad():
        full = score(query_subgraph)
        subgraph_set = {int(row) for row in query_subgraph.tolist() if row >= 0}
        matched = sorted(subgraph_set & {int(row) for row in image_concept_rows.tolist() if row >= 0})
        items = []
        for concept_row in matched:
            reduced = query_subgraph.clone()
            reduced[reduced == concept_row] = PAD
            items.append(
                {
                    "concept": str(concept_keys[concept_row]),
                    "bridge": concept_row not in prized_rows,
                    "score_without": score(reduced),
                }
            )
    return {"score": full, "matched": items}
```

Replace the one-line conditional `return ... if blocks else ...` with an explicit `if`/`else`; wrap long lines.
- [x] **Step 4: Run, expect PASS**; ruff. **Commit** `feat(models): stage-2 candidate scores and exact occlusion explanation`.

---

### Task 6: Cache preparation, train/select, evaluate test

**Files:** Create `src/mmxai/eval/pcst_sage_rerank.py`, Test `tests/test_pcst_sage_rerank.py`

Config dataclass:

```python
@dataclass(frozen=True)
class PcstSageConfig:
    """Settings of one G-Retriever stage-2 experiment (configs/clip_pcst_sage_vg*.yaml).

    graph_rerank.load_inputs / split_candidates / build_training_data read: dataset,
    features_dir, graph_dir, split_file, concept_edges_dir, top_k, same_text_threshold.
    """

    experiment: str
    dataset: str
    split_file: str
    features_dir: str
    graph_dir: str
    concept_edges_dir: str
    subgraph_mode: str  # "pcst" or "top_k_only"
    shuffle_concept_edges: bool
    prize_k: int
    edge_cost: float
    cache_dir: str
    model_dir: str
    top_k: int
    recall_ks: list[int]
    same_text_threshold: float
    num_hard_negatives: int
    temperature: float
    learning_rate: float
    batch_size: int
    max_epochs: int
    patience: int
    score_chunk: int  # memory setting for candidate_subgraph_scores
    alpha_grid: list[float]
    selection_metric: str
    output_dir: str
```

Functions (small, docstrings, reuse instead of copy):
- `load_pcst_sage_config(path)`: `load_yaml_dataclass` + `subgraph_mode in ("pcst", "top_k_only")` + `selection_metric == f"MRR@{top_k}"`.
- `cache_path(config, seed) -> Path`: `cache_dir/"shared"` unless `shuffle_concept_edges`, then `cache_dir/f"seed{seed}"` (only the shuffled graph depends on the seed). Comment why.
- `cache_meta(config, inputs, seed, split, num_texts) -> dict`: prize_k, edge_cost, subgraph_mode, shuffle flag, shuffle seed (or None), split, num_texts, split_sha256, graph git commit (from `inputs.graph_meta`), concept-edges dir.
- `compute_subgraph_rows(config, inputs, pairs, queries, log) -> (offsets, values)`: `top_concepts_per_query(queries, inputs.concept_features, config.prize_k)` → `split_subgraphs(top, num_concepts, pairs, mode, prize_k, edge_costs(pairs, edge_cost), log)` → `pack_subgraphs([s.concepts for s in subgraphs])`.
- `prepare_caches(config, seed, log) -> None`: `inputs = load_checked_inputs(config)`; `pairs = concept_pairs(inputs, config, seed)`; train: `train_graph = build_split_graph(inputs, "train")`, `training = build_training_data(config, inputs, train_graph, log)` (duck-typed config; same filter as ADR-0047), compute rows for `training.text_x`, save as split `"train"`; val: `split_candidates(config, inputs, "val")`, rows for `candidates.query_embeddings`, save as `"val"`. **Never touches test.** If a cache exists, skip it with a log line (do not recompute).
- `estimate_pcst_time(config, seed, num_texts, log) -> float`: compute subgraphs for the first `num_texts` train texts only, write nothing, return seconds per text and log the projected total. (For the "measure before running everything" rule.)
- `build_subgraph_training_data(training, inputs, train_graph, train_rows_matrix) -> SubgraphTrainingData`: image concepts from `image_concept_matrix(split_subgraph(inputs.graph, "train"))` (same image row order as `train_graph.image_ids` — assert `np.array_equal(split_subgraph(...).image_ids, train_graph.image_ids)`), `concept_x = torch.from_numpy(inputs.concept_features)`, other fields from `training`.
- `val_graph_scores(config, model, val)`: `candidate_subgraph_scores` for val with the val cache; returns np array.
- `train_and_select(config, seed, log) -> Path`: refuse if `model_dir/seed{k}` exists; `set_seed(seed)`; load inputs (checked), training data, caches (`load_cache` with `cache_meta` expectations incl. `num_texts`); `SubgraphSage(dim)`; `fit(model, data, validate, ..., loss_fn=subgraph_batch_loss)` where `validate` = MRR@100 of graph score alone on val (as `graph_only_val_score`); load best state; `sweep_alpha` → `choose_alpha`; save `model.pt` and `selection.json` (alpha, best_epoch, history, alpha_sweep, git, split_sha256, cache meta); write a val run (`run_facts` from `pcst_rerank` + `prize_k`, `edge_cost`, `best_epoch`, `alpha`).
- `evaluate_test(config, seed, log) -> Path`: refuse a second test run; read selection; refuse if split sha differs; load model; test candidates; compute test S_q (save cache split `"test"` under `cache_path`, refusing overwrite); scores → `fuse_scores` with saved alpha → `compute_metrics(rerank_ranks(...))`; write run. Chooses nothing.

- [ ] **Step 1: Failing tests** `tests/test_pcst_sage_rerank.py`:
  1. `load_pcst_sage_config` rejects a bad `subgraph_mode` (YAML in `tmp_path`).
  2. `cache_path` is `shared` for the main config and `seed1` for the shuffled one with seed 1.
  3. `prepare_caches` never asks for the test split: monkeypatch `load_checked_inputs`, `concept_pairs`, `build_split_graph`, `build_training_data`, `split_candidates`, `compute_subgraph_rows`, `save_cache` in `mmxai.eval.pcst_sage_rerank` with recorders returning tiny objects; assert recorded split names ⊆ {"train", "val"}.
  4. `evaluate_test` refuses when a test run exists and when the selection's `split_sha256` differs (monkeypatch `has_test_run`, `file_sha256`, write a fake `selection.json` in `tmp_path`).
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** the module as specified above.
- [ ] **Step 4: Run, expect PASS**: `python -m uv run pytest -q tests/test_pcst_cache.py tests/test_subgraph_sage.py tests/test_pcst_sage_rerank.py tests/test_pcst_rerank.py`; `python -m uv run python -c "import mmxai.eval.pcst_sage_rerank"`. Ruff. **Commit** `feat(eval): prepare caches, train and select stage 2 on val, evaluate test once`.

---

### Task 7: CLIs and configs

**Files:** `scripts/prepare_pcst_cache.py`, `scripts/train_pcst_sage.py`, `scripts/evaluate_pcst_sage.py`, `configs/clip_pcst_sage_vg.yaml`, `configs/clip_pcst_sage_vg_no_tree.yaml`, `configs/clip_pcst_sage_vg_shuffled.yaml`, `configs/README.md`, `scripts/README.md`, test in `tests/test_pcst_sage_rerank.py`

- [ ] **Step 1: Failing test**: the three configs load and differ only in `experiment`, `subgraph_mode`, `shuffle_concept_edges`, `cache_dir`, `model_dir`; main has `prize_k == 3`, `edge_cost == 0.5`.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Configs.** `configs/clip_pcst_sage_vg.yaml` — copy paths and training values from `configs/clip_graph_vg_lr1e4.yaml` (`num_hard_negatives: 15`, `temperature: 0.05`, `learning_rate: 0.0001`, `batch_size: 512`, `max_epochs: 10`, `patience: 2`, `alpha_grid` 0.0..1.0 step 0.1, `selection_metric: MRR@100`, `top_k: 100`, `recall_ks: [1, 5, 10]`, `same_text_threshold: 0.9999`), plus:

```yaml
# Decision (ADR-0051, stage 2): fixed G-Retriever SceneGraphs values, no grid.
#   Why: one PCST pass over the train texts takes about an hour; a grid is not feasible.
prize_k: 3
edge_cost: 0.5
subgraph_mode: pcst
shuffle_concept_edges: false
cache_dir: artifacts/pcst_cache/clip_pcst_sage_vg
model_dir: artifacts/models/visual_genome/clip_pcst_sage_vg
# Memory setting for scoring (queries per chunk), not a method hyperparameter.
score_chunk: 128
```

`_no_tree`: `subgraph_mode: top_k_only` (ablation 1), own `cache_dir`/`model_dir`. `_shuffled`: `shuffle_concept_edges: true` (ablation 2, seed = run seed), own dirs.
- [ ] **Step 4: CLIs** (docstring style of `scripts/train_graph_rerank.py`; raw docstrings if lines end with `\`):
  - `prepare_pcst_cache.py --config --seed [--estimate N]`: with `--estimate N` call `estimate_pcst_time` and write nothing; else `prepare_caches`. Docstring: run `--estimate 1000` first; for main/no_tree one cache serves all seeds (run with seed 0); shuffled needs seeds 0 1 2.
  - `train_pcst_sage.py --config --seed`: `train_and_select`.
  - `evaluate_pcst_sage.py --config --seed`: `evaluate_test` (only after the team approved ADR-0051 and all val runs are done).
  - Add rows to `configs/README.md` and `scripts/README.md`.
- [ ] **Step 5: Run, expect PASS**; `--help` of each script; ruff. **Commit** `feat(eval): CLIs and configs for G-Retriever stage 2 and its ablations`.

---

### Task 8: Docs and checks

**Files:** ADR-0051, `docs/guides/huan-luyen-va-xep-hang-lai.md`, `docs/ai-usage-log.md`

- [ ] **Step 1:** ADR-0051, section "Thiết kế bậc 2": add a "Code (bậc 2):" line listing this plan's files.
- [ ] **Step 2:** Guide: new section "10. G-Retriever bậc 2" (Vietnamese, beginner-first, match section 9's style): công thức; vì sao bậc 1 thua (link mục kết quả trong ADR) và bậc 2 khác ở đâu; thứ tự chạy trên Colab: `prepare_pcst_cache --estimate 1000` → `prepare_pcst_cache` (seed 0 cho bản chính và no_tree; 0 1 2 cho shuffled) → `train_pcst_sage` seed 0 1 2 cho 3 config → so với `clip_nograph_vg_lr1e4` trên val → chỉ khi nhóm duyệt: `evaluate_pcst_sage`; cache và model nằm trong `artifacts/` (symlink sang Drive); giải thích bằng `explain_candidate`; rò rỉ.
- [ ] **Step 3:** One row in `docs/ai-usage-log.md` (member column: leave the marked blank `(điền tên thành viên)`, self-checked `Chưa`).
- [ ] **Step 4:** `python -m uv run ruff check`, `python -m uv run ruff format --check src tests scripts`, `python -m uv run pytest -q` (only the 2 known failures). Paste tails.
- [ ] **Step 5: Commit** `docs: G-Retriever stage 2 guide and ADR code line`.

---

## Out of scope
- Real runs (Colab): cache, training, val selection, test. Need artifacts.
- Ablation 3 (ConceptNet edges, "KG đã biết") and 4 (drop `kg_evidence` edges): after the KG benchmark (ADR-0050) is merged.
- Evaluation on the query benchmark v2 and result tables.
