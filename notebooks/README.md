# notebooks/ — notebook demo và khám phá

**Là gì?** Notebook để **trình diễn** (nhập truy vấn → top-k + đường đi giải thích) và để khám phá dữ liệu.

Quy tắc (chi tiết: `docs/decisions/0012-chinh-sach-notebook.md`):
- Notebook "mỏng": chỉ `import mmxai` rồi gọi hàm. Không định nghĩa hàm dài trong notebook.
- Output được xoá trước khi commit bằng `nbstripout` (hook trong `.pre-commit-config.yaml`).
- Đặt tên `NN_mo-ta.ipynb`, ví dụ `01_demo_truy_van.ipynb`, `90_kham_pha_vg.ipynb`.
- Cài môi trường notebook: `uv sync --group ml --group graph --group demo`.
