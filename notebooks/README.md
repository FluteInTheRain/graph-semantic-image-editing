# notebooks/ — notebook demo và khám phá

**Là gì?** Notebook để **trình diễn** (nhập truy vấn → top-k + đường đi giải thích) và để khám phá dữ liệu.

Quy tắc (chi tiết: `docs/decisions/0012-chinh-sach-notebook.md`):
- Notebook "mỏng": chỉ `import mmxai` rồi gọi hàm. Không định nghĩa hàm dài trong notebook.
- Output được xoá trước khi commit bằng `nbstripout` (hook trong `.pre-commit-config.yaml`).
- Đặt tên `NN_mo-ta.ipynb`, ví dụ `01_demo_truy_van.ipynb`, `90_kham_pha_vg.ipynb`.
- Ngoại lệ (ADR-0041): `10_prepare_clip_embeddings.ipynb` chạy trên Kaggle nên chứa bản sao code. Sửa code trong `src/mmxai/`, rồi chạy `uv run python scripts/sync_notebook_code.py`.
- Cài môi trường notebook: `uv sync --group ml --group graph --group demo`.

## Danh sách

| Notebook | Làm gì | Chạy ở đâu |
|---|---|---|
| `10_prepare_clip_embeddings.ipynb` | Trích embedding CLIP (ADR-0041); chứa bản sao code | Kaggle |
| `20_xai_level3_kaggle.ipynb` | **xAI cấp độ 3** trên mô hình đã chốt `clip_graph_vg_lr1e4`: lấy embedding, đồ thị, checkpoint từ Kaggle input (thiếu thì dựng/train), chạy `scripts/evaluate_explanations.py` trên val, hiện bảng fidelity/validity/sparsity, rồi chạy test **một lần** sau khi nhóm đã commit lựa chọn. Đặt `DATASET = "flickr30k"` để chạy lại pipeline cấp độ 1 trên tập thứ hai (ADR-0050): dựng đồ thị, train 3 mô hình, test một lần (mục 4b). Notebook mỏng: chỉ gọi script | Kaggle |
