# Đánh giá các mô hình đã huấn luyện trên bộ truy vấn KG: thiết kế

- **Ngày:** 2026-11-21
- **Nhánh:** `feat/vg-kg-query-benchmark` (PR #14).
- **Liên quan:** ADR-0057 (bộ truy vấn ConceptNet), ADR-0043 (giao thức đánh giá), ADR-0047 (chọn `alpha` trên val), ADR-0051 (G-Retriever), PDF §5 (≥ 20 truy vấn, báo cáo tách trực tiếp / gián tiếp), PDF §6 (bộ truy vấn chỉ để đánh giá).

## 1. Mục tiêu

Hiện `scripts/evaluate_query_benchmark.py` chỉ chấm **CLIP thuần** trên bộ truy vấn. Đề yêu cầu so các mô hình trên bộ này, tách riêng truy vấn trực tiếp và gián tiếp. Cần chấm thêm các mô hình **đã huấn luyện và đã chốt**:

| Experiment | Loại | Vì sao |
|---|---|---|
| `clip_nograph_vg` | `graph_rerank` | Đối chứng không đồ thị chính thức (lr 1e-3) |
| `clip_graph_vg_lr1e4` | `graph_rerank` | Mô hình chính CLIP + GraphSAGE |
| `clip_cooc_graph_vg_lr1e4` | `graph_rerank` | Có cạnh đồng xuất hiện (khái niệm – khái niệm) |
| `clip_nograph_vg_lr1e4` | `graph_rerank` | Đối chứng cùng cấu hình của G-Retriever |
| `clip_pcst_sage_vg` | `pcst_sage` | G-Retriever giai đoạn 2 |

## 2. Nguyên tắc (không được vi phạm)

- **Không chọn gì trên bộ truy vấn.** Mỗi mô hình dùng đúng checkpoint và `alpha` đã lưu trong `selection.json` (chọn trên val). Không đổi top-k, `alpha`, `k`, `C` hay bất kỳ giá trị nào.
- **Cùng 100 ứng viên như CLIP thuần:** cosine chính xác giữa câu truy vấn và 5.000 ảnh test; mô hình chỉ xếp lại 100 ảnh này.
- **Mô hình không đọc** `hidden_concept`, `kg_evidence`, `path_hint` hay `relevant`. Ranker chỉ nhận câu (đã mã hoá CLIP).
- Câu truy vấn mã hoá bằng **đúng CLIP đóng băng** đã tạo đặc trưng ảnh (kiểm `meta.json`).
- Mỗi (experiment, seed) chỉ chạy **một lần**; chạy lại bị từ chối, như `evaluate_test`.
- G-Retriever: đồ thị con `S_q` của mỗi câu truy vấn được tính bằng đúng `compute_subgraph_rows` với `prize_k`, `edge_cost` của config. Không cache vì bộ chỉ có vài chục câu.

## 3. Thiết kế code

**Module mới `src/mmxai/eval/query_benchmark_models.py`:**

- `clip_candidates(test_embeddings, test_image_ids, query_embeddings, top_k) -> Candidates`: cosine chính xác bằng numpy, sắp giảm dần, `relevant` là danh sách tập rỗng (metric tính ở `score_rankings`).
- `reranked_ids(candidates, final_scores) -> np.ndarray`: đổi thứ tự `candidate_ids` theo điểm cuối giảm dần, ổn định khi hoà (giữ thứ tự CLIP).
- `GraphRerankRanker(model_vectors, image_ids, test_embeddings, alpha, top_k)` cài `QueryRanker.rank`. Điểm đồ thị là `candidate_graph_scores`, sau đó `fuse_scores` rồi `reranked_ids`.
- `PcstSageRanker(...)` cài `QueryRanker.rank`. Gọi `compute_subgraph_rows` cho câu truy vấn, rồi `split_scoring` / `graph_scores` của `pcst_sage_rerank`, sau đó `fuse_scores` rồi `reranked_ids`.
- `ModelBenchmarkEvalConfig`: `experiment`, `model_kind` (`graph_rerank` | `pcst_sage`), `model_config`, `benchmark_path`, `model`, `pretrained`, `device`, `text_batch_size`, `recall_ks`, `output_dir`. Dataset, split, đặc trưng, đồ thị và `top_k` đọc từ `model_config`.
- `run_model_benchmark_evaluation(config, seed) -> Path`:
  1. nạp bộ truy vấn đã kiểm (`load_verified_benchmark`);
  2. nạp config và checkpoint của mô hình; kiểm `selection.json` cùng `split_sha256`;
  3. mã hoá câu;
  4. dựng ranker và gọi `evaluate_benchmark`;
  5. ghi `results/query_benchmark/runs/<ngày>_<experiment>_seed<k>/` với `alpha`, `best_epoch`, `model_sha256`, `benchmark_sha256`.

**Script và config:**
- `scripts/evaluate_query_benchmark_models.py --config <cfg> --seed <k>`.
- 5 config `configs/eval_query_benchmark_vg_kg_<experiment>.yaml`.

**Tổng hợp:** `scripts/summarize_query_benchmark.py` gom các run trong `results/query_benchmark/runs` thành `results/tables/kg_query_benchmark.csv`:
- mỗi dòng là một experiment;
- R@1/5/10 và MRR@100, mean ± std mẫu qua seed, cho `all`, `direct`, `indirect` và từng `query_group` × loại;
- từ chối nếu một experiment có seed trùng hoặc thiếu seed.

## 4. Kiểm thử (dữ liệu tổng hợp, không mạng)

- `clip_candidates` ra đúng top-k và điểm giảm dần.
- `reranked_ids` đổi thứ tự đúng; `alpha = 1` giữ nguyên thứ tự CLIP.
- `GraphRerankRanker` với mô hình khởi tạo (`W_self = I`, `W_nb = 0`) cho đúng thứ tự CLIP.
- `PcstSageRanker` với `SubgraphSage` khởi tạo cho đúng thứ tự CLIP. Hàm PCST được monkeypatch để test không phụ thuộc `pcst_fast`.
- `run_model_benchmark_evaluation` từ chối chạy lần hai; từ chối `selection.json` khác split.
- 5 config nạp được, `model_config` trỏ tới file có thật, `benchmark_path` là `queries/visual_genome_kg_test.jsonl`.
- Hàm tổng hợp: mean ± std đúng trên run giả; từ chối seed trùng.

## 5. Chạy thật (người dùng, Colab)

Cần các checkpoint `artifacts/models/visual_genome/<experiment>/seed{0,1,2}/` (`model.pt`, `selection.json`):
- `clip_pcst_sage_vg` và `clip_nograph_vg_lr1e4`: đã có trên Drive của Thuật.
- `clip_nograph_vg`, `clip_graph_vg_lr1e4`, `clip_cooc_graph_vg_lr1e4`: do thành viên khác huấn luyện. Cần xin bản checkpoint. Nếu không có, huấn luyện lại bằng đúng config; quá trình huấn luyện tất định nên tái lập được.

Chỉ chạy **sau** khi bộ truy vấn đã được hai thành viên duyệt và build (`queries/visual_genome_kg_test.jsonl`).

## 6. Ngoài phạm vi

- Ablation đồ thị xáo trên bộ truy vấn.
- Đánh giá giải thích tách theo trực tiếp / gián tiếp.
- Flickr30k: bộ truy vấn chỉ có trên Visual Genome.
