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
| `scripts/summarize_runs.py` (đã có) | Gom `results/runs/*` thành `results/tables/main.csv` (mean ± std, ≥3 seed, bỏ run có code chưa commit) | `mmxai.eval.summary` |
| `scripts/build_graph.py` | Dựng đồ thị ảnh - đối tượng - khái niệm (cạnh thống kê chỉ từ train) | `mmxai.graph` |
| `scripts/train_gnn.py` | Huấn luyện GNN graph scorer | `mmxai.models` |
| `scripts/evaluate.py` | Re-rank, chọn `alpha` trên val, đo trên test, ≥3 seed | `mmxai.rerank`, `mmxai.eval` |
| `scripts/explain.py` | Xuất đường đi giải thích + fidelity/validity/sparsity | `mmxai.explain` |
