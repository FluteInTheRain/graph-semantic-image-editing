# scripts/ — điểm chạy dòng lệnh (CLI)

**Là gì?** Các file chạy trực tiếp bằng `uv run python scripts/<tên>.py --config configs/<file>.yaml`. Script chỉ **đọc tham số và gọi hàm** trong `src/mmxai/`; logic thật nằm trong package để test được.

Danh sách dự kiến (chưa viết), theo thứ tự chạy pipeline:

| Script | Làm gì | Gọi vào |
|---|---|---|
| `scripts/make_splits.py` (đã có) | Tạo file split train/val/test cố định, ghi vào `splits/`. Cần embedding Visual Genome (ADR-0042) | `mmxai.data.splits` |
| `scripts/extract_features.py` (đã có) | Trích đặc trưng CLIP **một lần**, lưu cache. Hướng dẫn: `docs/guides/chuan-bi-embedding-clip.md` | `mmxai.features.pipeline` |
| `scripts/sync_notebook_code.py` (đã có) | Chép code và config vào notebook Kaggle sau khi sửa | (chỉ đọc file) |
| `scripts/evaluate_clip.py` (đã có) | Baseline CLIP thuần: dựng FAISS trên ảnh của tập đánh giá, đo Recall@1/5/10 + MRR@100 (ADR-0043). Cần `uv sync --group ml` | `mmxai.eval.clip_baseline` |
| `scripts/build_index.py` (đã có) | Lưu vector store CLIP thuần: một FAISS index cho mỗi tập (`val`, `test`) vào `artifacts/vector_store/` | `mmxai.index.vector_store` |
| `scripts/generate_queries.py` (đã có) | Sinh cặp truy vấn trực tiếp/gián tiếp ứng viên từ scene graph Visual Genome + manifest để người kiểm tra (ADR-0044). Hướng dẫn: `docs/query_benchmark.md` | `mmxai.queries.generator` |
| `scripts/build_query_benchmark.py` (đã có) | Dựng `queries/visual_genome_test.jsonl` chỉ từ các cặp đã `verified` trong manifest | `mmxai.queries.validator` |
| `scripts/evaluate_query_benchmark.py` (đã có) | CLIP thuần trên bộ truy vấn đã kiểm tra: Recall@1/5/10/100 + MRR@100 tách direct/indirect. Cần `uv sync --group ml` | `mmxai.eval.query_benchmark` |
| `scripts/summarize_runs.py` (đã có) | Gom `results/runs/*` thành `results/tables/main.csv` (mean ± std, ≥3 seed, bỏ run có code chưa commit) | `mmxai.eval.summary` |
| `scripts/build_graph.py` (đã có) | Dựng đồ thị ảnh – khái niệm, chỉ cạnh `contains`; từ vựng khái niệm đếm trên train (ADR-0045). Hướng dẫn: `docs/guides/xay-do-thi.md`. Cần `uv sync --group ml --group graph` | `mmxai.graph.pipeline` |
| `scripts/train_gnn.py` (đã có) | Huấn luyện GNN một lớp, chỉ trên ảnh train, early stopping trên val (ADR-0046) | `mmxai.models.training` |
| `scripts/evaluate_rerank.py` (đã có) | Xếp hạng lại top 100 của CLIP, chọn `alpha` trên val, đo trên test với `alpha` đã chốt, lưu `scores.npz` | `mmxai.rerank.evaluate` |
| `scripts/setup_and_train.py` (đã có) | **Chạy một lần**: tự tải annotation Visual Genome còn thiếu (`configs/data_vg.yaml`), dựng đồ thị, huấn luyện, chọn `alpha`. Hướng dẫn: `docs/guides/chay-pipeline.md` | `mmxai.graph.pipeline`, `mmxai.models.training`, `mmxai.rerank.evaluate` |
| `scripts/run_pipeline.py` (đã có) | **Chạy nhiều lần**: truy vấn vài câu ở mode `clip` hoặc `clip_graph`, in kết quả + giải thích ảnh top-1 (Occlusion). Hướng dẫn: `docs/guides/chay-pipeline.md` | `mmxai.explain.demo` |
| `scripts/explain.py` | Fidelity/validity/sparsity trên ≥ 10 truy vấn (cấp độ 3) | `mmxai.explain` |
