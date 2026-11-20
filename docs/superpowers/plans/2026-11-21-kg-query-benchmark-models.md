# Đánh giá các mô hình đã huấn luyện trên bộ truy vấn KG: kế hoạch cài đặt

> Spec: `docs/superpowers/specs/2026-11-21-kg-query-benchmark-models-design.md`. Nhánh `feat/vg-kg-query-benchmark`.
> Mỗi task một commit (Conventional Commits, kết thúc bằng `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`).
> Trước mỗi commit chạy `python -m uv run ruff check` và `python -m uv run pytest -q`. Hai test trong
> `tests/test_notebooks_match_src.py` đã fail sẵn trên Windows; mọi test khác phải pass.

**Cách chạy:** 2 batch, chạy lần lượt.

| Batch | Task | Model thực hiện |
|---|---|---|
| B1 | T1 [M] + T2 [M] | Sonnet |
| B2 | T3 [M] + T4 [S] | Sonnet |

Sau mỗi batch có một lượt review chỉ theo spec (Haiku). Cuối cùng chạy `leakage-auditor`.

## Batch B1

### T1 [M]: ranker cho mô hình đã huấn luyện
- [ ] Tạo `src/mmxai/eval/query_benchmark_models.py` với các hàm và class sau:
  - `clip_candidates(test_embeddings, test_image_ids, query_embeddings, top_k) -> Candidates`
  - `reranked_ids(candidates, final_scores)`: sắp ổn định, khi hoà giữ thứ tự CLIP
  - `GraphRerankRanker`
  - `PcstSageRanker`

  Các ranker cài `QueryRanker.rank(texts, query_embeddings, top_k)` của `mmxai.eval.query_benchmark`. Dùng lại `candidate_graph_scores`, `fuse_scores` (`mmxai.rerank.fusion`), `compute_subgraph_rows`, `split_scoring`, `graph_scores` (`mmxai.eval.pcst_sage_rerank`); không viết lại logic chấm điểm.
- [ ] Thêm test `tests/test_query_benchmark_models.py`, chỉ dùng dữ liệu tổng hợp:
  - top-k và thứ tự của `clip_candidates`;
  - `reranked_ids` với `alpha = 1` giữ thứ tự CLIP;
  - mô hình khởi tạo (`ConceptToImageSage` hoặc `SubgraphSage` chưa huấn luyện) cho đúng thứ tự CLIP;
  - `PcstSageRanker` phải monkeypatch `compute_subgraph_rows`.
- [ ] Commit: `feat(eval): rank the query benchmark with trained re-rankers`.

### T2 [M]: chạy đánh giá, script, config
- [ ] Trong cùng module, thêm:
  - `ModelBenchmarkEvalConfig` và `load_model_benchmark_eval_config` (kiểm `model_kind` thuộc `{"graph_rerank", "pcst_sage"}`);
  - `has_benchmark_run`;
  - `run_model_benchmark_evaluation(config, seed)` theo spec mục 3.
- [ ] Quy tắc bắt buộc:
  - Từ chối chạy lần hai cho cùng (experiment, seed).
  - Kiểm `selection.json["split_sha256"]` bằng `file_sha256(split_file)` của config mô hình.
  - Kiểm `meta.json` của đặc trưng có cùng CLIP với config.
  - Ghi run vào `config.output_dir`, gồm `alpha`, `best_epoch`, `model_sha256`, `benchmark_sha256`, `split_sha256`, `seed`.
- [ ] Tạo `scripts/evaluate_query_benchmark_models.py`: CLI mỏng, docstring theo kiểu `scripts/evaluate_query_benchmark.py`.
- [ ] Tạo 5 config `configs/eval_query_benchmark_vg_kg_<exp>.yaml` cho `clip_nograph_vg`, `clip_graph_vg_lr1e4`, `clip_cooc_graph_vg_lr1e4`, `clip_nograph_vg_lr1e4`, `clip_pcst_sage_vg`:
  - `experiment: <exp>_kg_query_benchmark`;
  - `benchmark_path: queries/visual_genome_kg_test.jsonl`;
  - CLIP `ViT-B-32` / `laion2b_s34b_b79k`;
  - `recall_ks: [1, 5, 10, 100]`;
  - `output_dir: results/query_benchmark/runs`.
- [ ] Thêm test:
  - 5 config nạp được và `model_config` tồn tại;
  - chạy lại bị từ chối;
  - `selection.json` khác split bị từ chối (monkeypatch phần nạp nặng).
- [ ] Commit: `feat(eval): evaluate trained models on the KG query benchmark (ADR-0057)`.

## Batch B2

### T3 [M]: bảng tổng hợp
- [ ] Thêm `summarize_benchmark_runs(runs_dir) -> list[dict]` trong module trên. Hàm gom `metrics.json` theo experiment, tính mean ± std mẫu (`ddof = 1`) của R@1/5/10 và MRR@100 cho:
  - `all`, `direct`, `indirect`;
  - `by_query_group` × loại.

  Hàm từ chối seed trùng trong một experiment, và từ chối các run có `benchmark_sha256` khác nhau.
- [ ] Tạo `scripts/summarize_query_benchmark.py --runs-dir results/query_benchmark/runs --output results/tables/kg_query_benchmark.csv`.
- [ ] Viết test bằng run giả.
- [ ] Commit: `feat(eval): summarize query benchmark runs into one table`.

### T4 [S]: tài liệu
- [ ] Thêm mục "Đánh giá các mô hình đã huấn luyện" vào cuối `docs/query_benchmark.md` (tiếng Việt), gồm:
  - bảng 5 experiment;
  - điều kiện: bộ đã duyệt, có checkpoint;
  - lệnh theo thứ tự: CLIP thuần bằng script cũ, 5 mô hình × 3 seed, tổng hợp;
  - các cell Colab kiểu mục 11 của `docs/guides/huan-luyen-va-xep-hang-lai.md` (bundle, symlink Drive);
  - nguyên tắc chống rò rỉ của spec mục 2.
- [ ] Thêm một dòng vào cuối bảng `docs/ai-usage-log.md` (CRLF, thành viên `Thuat Nguyen`, cột cuối `Chưa`).
- [ ] Commit: `docs: how to evaluate trained models on the KG query benchmark`.
