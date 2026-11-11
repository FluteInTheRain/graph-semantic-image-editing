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
| `20_xai_level1_kaggle.ipynb` | **xAI cấp độ 1**: chạy `scripts/setup.py` + `scripts/train.py`, rồi in quan hệ dùng khi xếp hạng cho vài câu tự gõ và phân tích **5 ca thành công + 5 ca thất bại** (chọn theo quy tắc cố định), có ảnh nếu gắn ảnh Visual Genome; lưu `cases.md` / `cases.csv` cho báo cáo. Notebook mỏng: chỉ gọi `mmxai.explain.demo`, `mmxai.explain.figures` | Kaggle (khuyên dùng), Colab, local |
