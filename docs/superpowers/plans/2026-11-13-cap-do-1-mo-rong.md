# Level-1 Evaluation Extensions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add nDCG@10, an image → text direction and text → image query groups to the evaluation of plain CLIP and CLIP + Graph, then re-run 5 experiments (`_eval2`) on train/val/test once under a pre-registered protocol.

**Architecture:** Extend the one existing evaluation path (`evaluate_clip`, `train_and_select`, `evaluate_test`) through three small new modules: ranked metrics, query groups, and image → text retrieval. `metrics.json` gains new keys; `summarize_runs` writes three new tables. Old configs load and score exactly as before.

**Tech Stack:** Python 3.11, numpy, FAISS (`IndexFlatIP`), PyTorch + PyG (existing models), pytest, uv.

**Spec:** `docs/superpowers/specs/2026-11-13-cap-do-1-mo-rong-design.md` (approved 2026-11-13). Read it first; this plan does not repeat its reasons.

## Global Constraints

- Branch `feat/level1-eval-extensions` (from `feat/level1-ablation-explain`); new PR, not PR #8.
- Code, docstrings, comments in English; docs in Vietnamese; simple code for oral exam (no one-liners/metaprogramming).
- Logic in `src/mmxai/`, thin CLIs in `scripts/`, tests with tiny synthetic data, no network, no `data/`/`artifacts/`.
- No new dependency. Never edit `configs/query_benchmark_vg.yaml`, `splits/`, `queries/`, or rows of `results/tables/main.csv`.
- New config fields default to `None` (`load_yaml_dataclass` treats `default_factory` fields as required), so every existing config still loads.
- nDCG cutoff is 10, binary relevance; relevance definitions exactly as spec §5.1–5.2.
- Group names in code/JSON: grouping `specificity` → `specific` | `generic`; grouping `relation` → `action` | `spatial` | `none` (Vietnamese labels only in docs).
- The spec's `extra_predicates` list is split into `extra_action_predicates` and `extra_spatial_predicates`, so each extra says its group. This is the plan's naming call.
- Every choice (α for image → text, group rule tweaks) on val only; test once per `_eval2` experiment and seed; protocol commit before any `_eval2` training.
- Before reporting: `uv run ruff check` and `uv run pytest` clean, with real output pasted. macOS: libomp symlink fix (guide §4) must be in place.

## Review Focus

1. Group labels misaligned with query rows (text mask differs from `retrieve_candidates`): every label lands on the wrong query. Pinned in Task 2 (`test_relation_groups_follow_query_row_order`).
2. A sampled region id missing from `region_descriptions.json`: must raise naming the id, never fall into group `none`. Pinned in Task 2.
3. Ties in fused scores for image → text: stable sort must keep CLIP order, so α = 1 equals plain CLIP exactly. Pinned in Task 3.
4. Relevant items beyond top 10 / beyond top-k: nDCG stays in [0, 1] and IDCG uses min(#relevant, 10). Pinned in Task 1.
5. Old configs (no new fields): same `metrics.json` keys and values as before the change. Pinned in Tasks 4 and 5.

## File Structure

| File | Responsibility |
|---|---|
| `src/mmxai/eval/metrics.py` (modify) | add `ndcg_at_k`, `ranked_metrics` (R@K, MRR, nDCG from one ranked list) |
| `src/mmxai/eval/query_groups.py` (create) | specificity groups, predicate loading, relation groups, `metrics_by_group` |
| `src/mmxai/eval/image_to_text.py` (create) | text candidates per image, relevance inversion, graph scores for texts, ranking |
| `src/mmxai/eval/extended.py` (create) | `text_to_image_report`, `image_to_text_report`, `query_groupings` shared by CLIP and CLIP + Graph |
| `src/mmxai/utils/runs.py` (modify) | `test_run_exists` shared by both test guards |
| `src/mmxai/eval/clip_baseline.py`, `src/mmxai/eval/graph_rerank.py` (modify) | new optional config fields; call the reports; α for image → text on val |
| `src/mmxai/eval/summary.py`, `scripts/summarize_runs.py` (modify) | nDCG column, groups table, image → text table, experiment-suffix filter |
| `scripts/inspect_query_groups.py` (create) | val-only check of the group rule (spec §5.3) |
| `configs/*_eval2.yaml` (create, 5 files) | spec §5.4 |

---

### Task 1: nDCG@10 and metrics from one ranked list

**Files:**
- Modify: `src/mmxai/eval/metrics.py`
- Test: `tests/test_eval_metrics.py`

**Interfaces:**
- Produces: `ndcg_at_k(retrieved_ids: Sequence[int], relevant_ids: set[int], k: int) -> float`;
  `ranked_metrics(rankings: Sequence[Sequence[int]], relevant: Sequence[set[int]], recall_ks: Sequence[int], top_k: int, ndcg_k: int) -> dict`
  returning exactly `compute_metrics(first ranks, recall_ks, top_k)` plus key `f"nDCG@{ndcg_k}"` (mean over queries).

- [ ] **Step 1: Write the failing tests**

```python
def test_ndcg_one_relevant_at_first_position_is_one():
    assert ndcg_at_k([7, 3, 5], {7}, 10) == pytest.approx(1.0)

def test_ndcg_two_relevant_at_positions_two_and_five():
    dcg = 1 / math.log2(3) + 1 / math.log2(6)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k([9, 1, 8, 6, 2], {1, 2}, 10) == pytest.approx(dcg / idcg)

def test_ndcg_counts_relevant_items_outside_the_list_in_the_ideal():
    # 12 relevant ids, one found at position 1: IDCG uses min(12, 10) = 10 terms.
    idcg = sum(1 / math.log2(j + 1) for j in range(1, 11))
    assert ndcg_at_k([0], set(range(12)), 10) == pytest.approx(1 / idcg)

def test_ndcg_is_zero_without_relevant_in_the_first_k():
    assert ndcg_at_k(list(range(20)), {15}, 10) == 0.0

def test_ranked_metrics_match_compute_metrics_on_the_same_lists():
    rankings = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    relevant = [{2}, {9}, {7, 8}]
    result = ranked_metrics(rankings, relevant, [1, 2], top_k=3, ndcg_k=3)
    expected = compute_metrics([2, None, 1], [1, 2], 3)
    for name, value in expected.items():
        assert result[name] == value
    assert "nDCG@3" in result
```

- [ ] **Step 2: Run and see them fail** — `uv run pytest -q tests/test_eval_metrics.py` → FAIL (`ndcg_at_k` not defined).
- [ ] **Step 3: Implement both functions in `src/mmxai/eval/metrics.py`** — `ranked_metrics` calls `first_relevant_rank`, `compute_metrics` and `ndcg_at_k`; module docstring gains a "Decision (spec 2026-11-13): nDCG@10, binary relevance" note.
- [ ] **Step 4: Run** — `uv run pytest -q tests/test_eval_metrics.py` → all pass.
- [ ] **Step 5: Commit** — `feat(eval): nDCG@10 and metrics from one ranked list`.

### Task 2: Query groups and the val inspection script

**Files:**
- Create: `src/mmxai/eval/query_groups.py`, `scripts/inspect_query_groups.py`
- Test: `tests/test_eval_query_groups.py`

**Interfaces:**
- Consumes: `ranked_metrics` (Task 1); `load_region_texts` (`mmxai.data.visual_genome`).
- Produces:
  - `specificity_groups(relevant: Sequence[set[int]]) -> list[str]` (`specific` if exactly 1 relevant image, else `generic`).
  - `@dataclass(frozen=True) class Predicates: action: list[tuple[str, ...]]; spatial: list[tuple[str, ...]]`.
  - `load_predicates(path: str | Path, extra_action: list[str] | None, extra_spatial: list[str] | None, excluded: list[str] | None) -> Predicates`. Reads `predicate_templates` keys, `predicate_aliases` (alias → canonical, alias joins the canonical's group) and `action_predicates`; action = action_predicates + their aliases; spatial = other template keys + their aliases; extras appended; excluded removed from both; each predicate stored as a tuple of words.
  - `relation_group(text: str, predicates: Predicates) -> str`. Lowercase, replace non-letters with spaces, split; `action` if any action tuple occurs as consecutive words, else `spatial` if any spatial tuple occurs, else `none`.
  - `relation_groups(text_ids: Sequence[int], region_texts: dict[int, str], predicates: Predicates) -> list[str]`. Raises `KeyError` naming the first missing region id.
  - `metrics_by_group(rankings, relevant, labels: Sequence[str], recall_ks, top_k, ndcg_k) -> dict[str, dict]`. One `ranked_metrics` result per group, plus `fraction_queries`.

- [ ] **Step 1: Write the failing tests**: tiny predicate YAML written in `tmp_path` with keys `predicate_templates`, `predicate_aliases`, `action_predicates`.
  - `test_specificity_splits_on_the_number_of_relevant_images`: `[{1}, {1, 2}, {3}]` → `["specific", "generic", "specific"]`.
  - `test_action_wins_over_spatial`: `"man holding umbrella on street"` → `action`.
  - `test_multiword_predicate_matches_only_as_consecutive_words`: `"cat next to dog"` → `spatial`; `"next dog to"` → `none`.
  - `test_predicate_does_not_match_inside_a_word`: `"onion"` → `none`.
  - `test_alias_joins_its_canonical_group`: `"man holds cup"` → `action`.
  - `test_excluded_predicate_is_not_a_relation`: `excluded=["in"]`, `"man in shirt"` → `none`.
  - `test_missing_region_text_raises_with_the_id`: `pytest.raises(KeyError, match="42")`.
  - `test_relation_groups_follow_query_row_order`: text ids `[5, 3]` → labels follow that order.
  - `test_metrics_by_group_cover_every_query_once`: the group `num_queries` sum to the total, and `fraction_queries` sums to 1.
- [ ] **Step 2: Run** — `uv run pytest -q tests/test_eval_query_groups.py` → FAIL (module missing).
- [ ] **Step 3: Implement `src/mmxai/eval/query_groups.py`** per the Interfaces.
- [ ] **Step 4: Implement `scripts/inspect_query_groups.py`**:
  - Args: `--config` (a `_eval2` graph or CLIP config), `--split` with choices `["val"]` only, `--per-group 30`.
  - Prints the size of every group of both groupings, then 30 random sentences per relation group (`numpy.random.default_rng(0)`).
  - Query rows use the same mask as `retrieve_candidates`.
  - Docstring explains it is the spec §5.3 check.
- [ ] **Step 5: Run** — `uv run pytest -q tests/test_eval_query_groups.py` → all pass.
- [ ] **Step 6: Commit** — `feat(eval): text-to-image query groups (specificity, relation) and a val inspection script`.

### Task 3: Image → text retrieval

**Files:**
- Create: `src/mmxai/eval/image_to_text.py`
- Test: `tests/test_eval_image_to_text.py`

**Interfaces:**
- Consumes: `build_image_index`, `search_images` (`mmxai.index.image_index`; generic over any ids); `same_text_relevant_images`.
- Produces:
  - `@dataclass(frozen=True) class TextCandidates`:
    - `query_image_ids: np.ndarray` (num_images,);
    - `text_embeddings: np.ndarray` (the split's texts, rows = text rows);
    - `candidate_rows: np.ndarray` (num_images, top_k) text rows, best first;
    - `clip_scores: np.ndarray` (num_images, top_k);
    - `relevant: list[set[int]]` (per image, its relevant text rows).
  - `retrieve_text_candidates(image_embeddings, image_ids, text_embeddings, text_image_ids, eval_image_ids, top_k, same_text_threshold) -> TextCandidates`.
    - Gallery = texts whose image is in `eval_image_ids`, using the same mask as `retrieve_candidates`.
    - Queries = the split's images in `image_ids` order.
    - Relevance = inversion of `same_text_relevant_images`: text row r is relevant for image i iff i ∈ relevant_images[r].
  - `text_graph_scores(candidates: TextCandidates, image_h: np.ndarray, h_image_ids: np.ndarray) -> np.ndarray`: cos(t, h_i), shape (num_images, top_k).
  - `ranked_text_rows(candidates: TextCandidates, final_scores: np.ndarray) -> list[list[int]]`: stable sort (`np.argsort(-s, kind="stable")`).

- [ ] **Step 1: Write the failing tests**: 3 images × 2 texts, one-hot style vectors so ranks can be computed by hand.
  - `test_relevant_texts_are_the_image_own_texts_and_identical_sentences`: one text duplicated in another image counts for both images.
  - `test_candidates_are_ordered_by_clip_cosine`.
  - `test_alpha_one_gives_exactly_the_clip_order`: fuse with `fuse_scores(clip, graph, 1.0)`, and include equal final scores.
  - `test_text_graph_scores_are_cosines_with_h`: compare against a hand-computed value.
  - `test_gallery_holds_only_texts_of_the_evaluated_images`.
- [ ] **Step 2: Run** — `uv run pytest -q tests/test_eval_image_to_text.py` → FAIL.
- [ ] **Step 3: Implement `src/mmxai/eval/image_to_text.py`** per the Interfaces. Chunk `text_graph_scores` like `candidate_graph_scores`.
- [ ] **Step 4: Run** → pass.
- [ ] **Step 5: Commit** — `feat(eval): image-to-text retrieval with the same pair score`.

### Task 4: Shared reports, test guard and plain CLIP

**Files:**
- Create: `src/mmxai/eval/extended.py`
- Modify: `src/mmxai/utils/runs.py`, `src/mmxai/eval/clip_baseline.py`, `src/mmxai/eval/graph_rerank.py` (`has_test_run` delegates only), `scripts/evaluate_clip.py` (docstring)
- Test: `tests/test_eval_extended.py`, `tests/test_clip_baseline.py`

**Interfaces:**
- Consumes: Tasks 1–3; `Candidates`, `rerank_ranks` (fusion).
- Produces:
  - `test_run_exists(output_dir: str | Path, experiment: str, seed: int) -> bool` in `mmxai.utils.runs`. Body moved from `graph_rerank.has_test_run`, which now calls it.
  - `ranked_image_ids(candidates: Candidates, final_scores: np.ndarray) -> list[list[int]]` in `extended.py`, with the same stable sort as `rerank_ranks`.
  - `text_to_image_report(candidates, final_scores, groupings: dict[str, list[str]] | None, recall_ks, top_k, ndcg_k) -> dict`:
    - returns the `ranked_metrics` keys plus `"groups": {grouping: metrics_by_group(...)}` when `groupings` is given;
    - raises `ValueError` if its R@K/MRR differ from `compute_metrics(rerank_ranks(candidates, final_scores), ...)`.
  - `image_to_text_report(candidates: TextCandidates, final_scores, recall_ks, top_k, ndcg_k) -> dict`.
  - `query_groupings(candidates: Candidates, text_ids: np.ndarray, vg_data_dir: str, predicates: Predicates) -> dict[str, list[str]]`. `text_ids` are the split's query rows.
  - Optional fields for `ClipEvalConfig` (also used by Task 5 for `GraphRerankConfig`), all defaulting to `None`:
    - `ndcg_k: int | None`;
    - `image_to_text: bool | None`;
    - `vg_data_dir: str | None`;
    - `predicates_file: str | None`;
    - `extra_action_predicates: list[str] | None`;
    - `extra_spatial_predicates: list[str] | None`;
    - `excluded_predicates: list[str] | None`.
  - `ndcg_k` set → nDCG; `vg_data_dir` and `predicates_file` set → groups; `image_to_text: true` → `metrics["image_to_text"]` block.

- [ ] **Step 1: Write the failing tests**
  - `test_text_to_image_report_matches_the_pipeline_ranks`.
  - `test_report_raises_when_ranks_disagree`: `monkeypatch.setattr(mmxai.eval.extended, "rerank_ranks", lambda c, s: [None] * len(s))`, then `pytest.raises(ValueError, match="disagree")`.
  - `test_old_clip_config_writes_the_same_metrics_keys_as_before`: `configs/clip_vg.yaml` style config in `tmp_path`, with tiny features as in the existing `tests/test_clip_baseline.py` fixtures. Expected keys are exactly the current ones.
  - `test_clip_eval2_metrics_have_ndcg_groups_and_image_to_text`.
  - `test_second_clip_test_run_is_refused`: `pytest.raises(FileExistsError, match="already run")`.
- [ ] **Step 2: Run** — `uv run pytest -q tests/test_eval_extended.py tests/test_clip_baseline.py` → FAIL.
- [ ] **Step 3: Implement** `extended.py`, `test_run_exists`, the new fields, `evaluate_clip`/`run_clip_evaluation` changes and the CLIP test guard. Plain CLIP uses `final_scores = candidates.clip_scores`, and image → text with α = 1.
- [ ] **Step 4: Run** the files above plus `tests/test_graph_rerank_pipeline.py` → pass.
- [ ] **Step 5: Commit** — `feat(eval): shared evaluation reports, plain-CLIP extensions and test guard`.

### Task 5: CLIP + Graph extensions and α for image → text on val

**Files:**
- Modify: `src/mmxai/eval/graph_rerank.py`
- Test: `tests/test_graph_rerank_pipeline.py`

**Interfaces:**
- Consumes: Task 4 reports; `retrieve_text_candidates`, `text_graph_scores`.
- Produces:
  - `GraphRerankConfig` gains the Task 4 optional fields.
  - `Prepared` gains `val_text_candidates: TextCandidates | None`, built in `prepare` only when `image_to_text` is true.
  - `selection.json` gains `alpha_image_to_text` and `alpha_sweep_image_to_text`. These come from `sweep_alpha`-style metrics over `config.alpha_grid` on val image → text, chosen by `choose_alpha(..., config.selection_metric)`, after the epoch and text → image α are chosen exactly as today.
  - The val run and `evaluate_test` write the `text_to_image_report` keys (with the text → image α) and, if enabled, `metrics["image_to_text"]` (with `alpha_image_to_text`). `evaluate_test` reads both alphas from `selection.json` and chooses nothing.

- [ ] **Step 1: Write the failing tests** (tiny pipeline of the existing file)
  - `test_old_graph_config_keeps_selection_and_metrics_keys`.
  - `test_image_to_text_alpha_is_chosen_on_val_and_reused_on_test`: the test `metrics["image_to_text"]["alpha"] == selection["alpha_image_to_text"]`.
  - `test_eval2_test_metrics_have_ndcg_groups_and_image_to_text`.
  - `test_epoch_and_text_to_image_alpha_do_not_change_when_extensions_are_on`: same seed, old config vs config with extensions, equal `best_epoch` and `alpha`.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** as described. Add an image → text sweep helper next to `sweep_alpha` in `extended.py` or `image_to_text.py` (builder's choice), using `ranked_text_rows` + `ranked_metrics`.
- [ ] **Step 4: Run** `uv run pytest -q` (whole suite) → pass.
- [ ] **Step 5: Commit** — `feat(eval): CLIP + Graph extensions and image-to-text alpha on val`.

### Task 6: Tables

**Files:**
- Modify: `src/mmxai/eval/summary.py`, `scripts/summarize_runs.py`
- Test: `tests/test_eval_summary.py`

**Interfaces:**
- Produces:
  - `metric_names` also returns `nDCG@*`.
  - `summarize_runs(records, split, experiment_suffix: str | None = None)` filters by name suffix.
  - `summarize_groups(records, split, experiment_suffix) -> list[dict]`: rows `experiment, grouping, group, num_queries_mean, fraction_queries_mean, <metric>_mean, <metric>_std`.
  - `summarize_image_to_text(records, split, experiment_suffix) -> list[dict]`: metric mean/std plus `alphas` (space-separated per seed).
  - CLI: `--kind {main,groups,image_to_text}` (default `main`) and `--experiment-suffix`. The default call behaves exactly as today.

- [ ] **Step 1: Failing tests**:
  - `test_default_summary_is_unchanged`;
  - `test_suffix_keeps_only_eval2_experiments`;
  - `test_groups_table_has_one_row_per_group_with_mean_and_std`;
  - `test_image_to_text_table_lists_alpha_of_each_seed`.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → pass. **Step 5: Commit** — `feat(eval): tables for nDCG, query groups and image-to-text`.

### Task 7: `_eval2` configs

**Files:**
- Create: `configs/clip_vg_eval2.yaml`, `configs/clip_nograph_vg_eval2.yaml`, `configs/clip_graph_vg_lr1e4_eval2.yaml`, `configs/clip_cooc_graph_vg_lr1e4_eval2.yaml`, `configs/clip_graph_vg_lr1e4_shuffled_eval2.yaml`
- Modify: `configs/README.md`
- Test: `tests/test_graph_rerank_pipeline.py`, `tests/test_clip_baseline.py`

**Config values:**
- Each file is a copy of its source (spec §5.4).
- `experiment: <source>_eval2`; graph configs `model_dir: artifacts/models/visual_genome/<source>_eval2`.
- `ndcg_k: 10`, `image_to_text: true`, `vg_data_dir: data/visual_genome`, `predicates_file: configs/query_benchmark_vg.yaml`.
- The three predicate lists are `null`.
- `clip_vg_eval2.yaml` keeps `eval_split: val`.

- [ ] **Step 1: Failing test** `test_eval2_configs_differ_from_their_source_only_in_name_model_dir_and_extension_fields`, parametrised over the 5 pairs.
- [ ] **Step 2: Run** → FAIL. **Step 3: Write the configs and the README rows** (Vietnamese). **Step 4: Run** → pass. **Step 5: Commit** — `feat(configs): eval2 copies of the five reported experiments`.

### Task 8: Val checks (runtime, no test split)

- [ ] **Step 1: Reproduce old val numbers.**
  - Run `uv run python scripts/evaluate_clip.py --config configs/clip_vg_eval2.yaml --seed 0`.
  - Expected: R@1/5/10 and MRR@100 equal **exactly** those in `results/runs/20261105-1029_clip_vg_seed0/metrics.json`. If not, stop and debug.
- [ ] **Step 2: Inspect groups.**
  - Run `uv run python scripts/inspect_query_groups.py --config configs/clip_vg_eval2.yaml --split val > <scratch>/groups_val.txt`.
  - Read the sizes and the 30 samples per group.
  - If the rule is clearly wrong: change matching code or the config lists only (spec §5.3), with reasons. Rerun Task 2 tests and this step, then commit (`fix(eval): ...`).
- [ ] **Step 3:** Move the val run folders made in this task out of `results/runs/` into `results/records/<YYYYMMDD>_cap-do-1-mo-rong/pre_lock_checks/`. `summarize_runs` refuses a seed run twice, and the final `_eval2` val runs must come after the lock.

### Task 9: Lock the protocol (before any `_eval2` training)

- [ ] **Step 1:** Write ADR-0049 with the `/new-adr` skill, status "Đề xuất", decided by FluteInTheRain (2026-11-13). It records:
  - evaluation protocol v2 (nDCG@10, image → text with α chosen on val, the two groupings);
  - the 5 re-run conditions (spec §3);
  - any val-based predicate list changes from Task 8.
- [ ] **Step 2:** Add a "Giao thức đánh giá v2" section to `results/tables/README.md`:
  - the 5 `_eval2` rows and the new tables;
  - "report all numbers whatever they show";
  - old rows kept.
- [ ] **Step 3:** Run `uv run pytest -q` and `uv run ruff check` → clean. Commit `docs(adr): lock the evaluation protocol v2 before the re-runs`. `git status` must be clean.

### Task 10: Re-run (runtime)

- [ ] **Step 1:** `uv run python scripts/build_concept_edges.py --config configs/concept_edges_vg_cooccurrence.yaml`.
- [ ] **Step 2:** For each of the 4 graph `_eval2` configs: `uv run python scripts/train_graph_rerank.py --config <cfg> --seeds 0 1 2 --threads 8`.
- [ ] **Step 3:** Plain CLIP val: `evaluate_clip.py --config configs/clip_vg_eval2.yaml --seed {0,1,2}`.
- [ ] **Step 4:** Test once per experiment and seed:
  - graph configs: `evaluate_graph_rerank.py --config <cfg> --seed {0,1,2}`;
  - CLIP: `evaluate_clip.py --config configs/clip_vg_eval2.yaml --split test --seed {0,1,2}`.
  - Expected: every run has `git_dirty: false` at the Task 9 commit.
- [ ] **Step 5:** Write the tables:
  - `summarize_runs.py --experiment-suffix _eval2 --output results/tables/eval2_text_to_image.csv`;
  - `--kind groups --output results/tables/eval2_groups.csv`;
  - `--kind image_to_text --output results/tables/eval2_image_to_text.csv`.
  - Commit `feat(results): eval2 tables`.

### Task 11: Records, docs, review, PR

- [ ] **Step 1:** Write `results/records/<YYYYMMDD>_cap-do-1-mo-rong/README.md` (memory rule: params, per-run numbers, conclusions, reasoning, decisions, reviews, commit timeline). Copy into it:
  - the run files (`run_copies/`);
  - `selection.json`;
  - the Task 8 outputs.
  - Note how far the `_eval2` R@K/MRR numbers differ from `main.csv`.
- [ ] **Step 2:** Update the guide section, the `README.md` checklist (Mở rộng NV1) and `docs/ai-usage-log.md` (`Chưa`).
- [ ] **Step 3:** Run `leakage-auditor`, `experiment-reviewer` and `pr-review-toolkit:pr-test-analyzer` on `git diff feat/level1-ablation-explain...HEAD`. Verify the findings (receiving-code-review), then fix.
- [ ] **Step 4:** Run `uv run pytest -q` and `uv run ruff check` and paste the output. Push. Open the PR with base `feat/level1-ablation-explain`, or `main` if PR #8 is merged.
