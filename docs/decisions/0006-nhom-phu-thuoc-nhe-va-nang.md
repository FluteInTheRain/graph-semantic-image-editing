# ADR-0006: Phụ thuộc mặc định nhẹ, thư viện ML nặng tách thành nhóm tuỳ chọn

- **Trạng thái:** Đã chấp nhận
- **Ngày:** 2026-11-01

## Bối cảnh

`torch` + `open-clip-torch` + `faiss` + `torch-geometric` nặng vài GB và khác nhau theo máy (Mac MPS, CUDA, Colab). Nếu bắt buộc cài hết thì:
- `uv sync` mất nhiều phút, lint/test chậm, hook của công cụ AI chạy chậm theo;
- Bạn chỉ viết tài liệu hoặc chỉ sửa `eval` cũng phải tải torch.

**Dependency group (PEP 735)** = nhóm phụ thuộc có tên trong `pyproject.toml` (`[dependency-groups]`). `uv sync` chỉ cài nhóm `dev`; nhóm khác cài khi gọi `--group <tên>`. `uv.lock` vẫn khoá phiên bản cho **mọi** nhóm.

## Quyết định

| Nhóm | Gói | Cài bằng | Khi nào cần |
|---|---|---|---|
| mặc định | `numpy`, `pyyaml` | `uv sync` | Luôn luôn |
| `dev` | `ruff`, `pytest`, `pre-commit`, `nbstripout` | `uv sync` (tự động) | Luôn luôn |
| `ml` | `torch`, `open-clip-torch`, `faiss-cpu`, `pillow`, `pandas`, `tqdm` | `uv sync --group ml` | Trích đặc trưng, FAISS |
| `graph` | `torch-geometric`, `networkx` | `uv sync --group ml --group graph` | Dựng đồ thị, huấn luyện GNN |
| `nlp` | `sentence-transformers` | `uv sync --group nlp` | Liên kết khái niệm (tuỳ chọn) |
| `demo` | `jupyterlab`, `ipykernel`, `matplotlib`, `gradio` | `uv sync --group demo` | Notebook, UI demo |

Cài tất cả: `uv sync --all-groups`.

Quy tắc code: module trong `src/mmxai/` **import torch/faiss/open_clip bên trong hàm** hoặc chỉ ở module dành riêng cho bước đó, để `import mmxai` và test mặc định không cần thư viện nặng.

Theo nền tảng:

| Máy | Thiết bị | Ghi chú |
|---|---|---|
| macOS Apple Silicon | `mps` hoặc `cpu` | Wheel torch mặc định trên PyPI có MPS; PyG lõi chạy được; tránh `pyg-lib` |
| Linux/Windows có NVIDIA | `cuda` | Nếu cần bản CUDA cụ thể, cấu hình `[tool.uv.sources]` trỏ index PyTorch, ghi ADR mới |
| Google Colab | `cuda` | Colab có sẵn torch; chỉ `pip install -e .` + gói thiếu, **không** cài lại torch |

Code chọn thiết bị qua một hàm duy nhất (`cuda` → `mps` → `cpu`), không hard-code.

## Các phương án đã cân nhắc

| Phương án | Ưu | Nhược | Lý do không chọn |
|---|---|---|---|
| Mặc định nhẹ + dependency groups | `uv sync` vài giây, test nhanh, vẫn khoá đủ | Phải nhớ thêm `--group` | **Được chọn** |
| Tất cả vào `dependencies` | Một lệnh duy nhất | Chậm, vỡ trên máy không hợp torch | Làm chậm mọi người |
| `optional-dependencies` (extras) | Chuẩn cũ, pip hiểu | Extras là metadata xuất bản; phù hợp thư viện phát hành hơn | Groups là chuẩn mới cho môi trường dev |
| requirements-*.txt riêng | Quen | Không có lock chung, dễ lệch | Mất lợi ích `uv.lock` |

## Hệ quả

- CI/hook chỉ cần `uv sync` → nhanh.
- Ai chạy pipeline thật phải nhớ cài nhóm `ml`/`graph`; `README.md` ghi rõ lệnh.
- Gói mới nặng phải vào đúng nhóm (`uv add --group ml <gói>`), không vào mặc định.

## Nguồn tham khảo

- PEP 735 Dependency Groups: https://peps.python.org/pep-0735/
- uv, dependency groups: https://docs.astral.sh/uv/concepts/projects/dependencies/#dependency-groups
- PyTorch trên Apple MPS: https://pytorch.org/docs/stable/notes/mps.html
