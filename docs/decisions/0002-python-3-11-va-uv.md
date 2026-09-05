# ADR-0002: Dùng Python 3.11 và uv để quản lý môi trường

- **Trạng thái:** Đã chấp nhận
- **Ngày:** 2026-11-01

## Bối cảnh

**Môi trường** = phiên bản Python + đúng các thư viện với đúng phiên bản. Nếu mỗi máy một kiểu, kết quả không tái lập được.

Ràng buộc:
- Đề (§8) yêu cầu "file requirements/environment và hướng dẫn chạy lại"; §9 yêu cầu chạy lại thí nghiệm nhỏ ngay tại buổi chấm.
- Python hệ thống trên máy macOS của nhóm là 3.9, quá cũ (PyG, open_clip mới cần ≥ 3.10).
- Máy thành viên khác nhau: macOS Apple Silicon, Windows/Linux, Google Colab.

**uv là gì?** Công cụ (viết bằng Rust) thay cho `pip` + `venv` + `pyenv`: tự tải đúng bản Python, tạo `.venv`, cài thư viện, và ghi **file khoá** `uv.lock` chứa phiên bản chính xác của mọi gói.

## Quyết định

| Hạng mục | Lựa chọn |
|---|---|
| Phiên bản Python | 3.11, ghi trong `.python-version`; `pyproject.toml` khai `requires-python = ">=3.11"` |
| Công cụ | `uv` (trên macOS: `/opt/homebrew/bin/uv`) |
| Khai báo phụ thuộc | `pyproject.toml` (chuẩn PEP 621 + PEP 735) |
| Khoá phiên bản | `uv.lock` — **commit vào git** |
| Lệnh chạy | luôn qua `uv run ...` (không cần `activate`) |

Lệnh cơ bản:

```bash
uv sync                      # tạo .venv, cài gói mặc định + nhóm dev
uv run pytest                # chạy test
uv add <gói>                 # thêm phụ thuộc (tự cập nhật pyproject + uv.lock)
uv export --format requirements-txt > requirements.txt   # khi cần nộp file requirements
```

## Các phương án đã cân nhắc

| Phương án | Ưu | Nhược | Lý do không chọn |
|---|---|---|---|
| uv + `uv.lock` | Rất nhanh, tự cài Python, khoá chính xác, đa nền tảng | Công cụ mới, một số bạn chưa quen | **Được chọn** |
| pip + venv + requirements.txt | Ai cũng biết | Không tự cài Python 3.11, không có lock chuẩn, dễ lệch phiên bản | Không giải quyết vấn đề Python 3.9 |
| conda / mamba | Quen trong ML, cài được CUDA | Chậm, file environment.yml không khoá chặt, nặng | Chậm, khó tái lập chính xác |
| Poetry | Có lock | Chậm hơn uv, không quản lý bản Python | uv làm được mọi thứ Poetry làm, nhanh hơn |
| Python 3.12 / 3.13 | Mới hơn | Một số wheel ML (faiss, pyg-lib) ra chậm hơn | 3.11 là điểm an toàn cho stack ML |

## Hệ quả

- Clone repo → `uv sync` → chạy được. Không cần cài Python tay.
- `uv.lock` thay đổi khi thêm gói: commit cùng `pyproject.toml` trong cùng một commit.
- Colab: dùng `pip install -e .` hoặc `uv pip install --system -e .` (Colab có sẵn torch, xem ADR-0006).
- Nếu đề yêu cầu `requirements.txt`, sinh ra bằng `uv export`, không viết tay.

## Nguồn tham khảo

- Tài liệu uv: https://docs.astral.sh/uv/
- PEP 621 (metadata trong pyproject): https://peps.python.org/pep-0621/
- Đề bài `Project03_Multimodal_XAI.pdf`, §8 và §9.
