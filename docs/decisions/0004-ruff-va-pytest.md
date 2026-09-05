# ADR-0004: Dùng ruff (lint + format) và pytest

- **Trạng thái:** Đã chấp nhận
- **Ngày:** 2026-11-01

## Bối cảnh

- **Lint** = tự động tìm lỗi/thói quen xấu trong code (biến không dùng, import thừa...).
- **Format** = tự động định dạng code theo một kiểu thống nhất, để diff chỉ chứa thay đổi thật.
- **Test** = đoạn code kiểm tra code khác chạy đúng.

5 người cùng sửa code: không thống nhất kiểu viết thì PR đầy thay đổi khoảng trắng, khó review. Đề (§9) còn yêu cầu giải thích được mọi đoạn mã — code gọn, có test thì dễ giải thích hơn. Ngoài ra, công cụ AI (Claude Code) trong repo được nối hook chạy đúng ba lệnh bên dưới, nên ba lệnh này phải luôn chạy được.

## Quyết định

Ba lệnh chuẩn, chạy trước mỗi commit/PR:

```bash
uv run ruff check           # lint
uv run ruff format          # format (dùng --check để chỉ kiểm tra)
uv run pytest               # test
```

Cấu hình trong `pyproject.toml`:

| Mục | Giá trị | Ý nghĩa |
|---|---|---|
| `[tool.ruff] line-length` | 100 | Dòng tối đa 100 ký tự |
| `target-version` | `py311` | Khớp ADR-0002 |
| `extend-include` | `*.ipynb` | Lint cả notebook |
| `[tool.ruff.lint] select` | `E, F, W, I, B, UP, SIM, NPY` | Lỗi cơ bản, sắp xếp import, bug phổ biến, cú pháp mới, đơn giản hoá, numpy |
| `[tool.pytest.ini_options] testpaths` | `tests` | Chỉ tìm test trong `tests/` |

Quy ước test:
- File `tests/test_<module>.py`. Test mặc định phải **nhanh và không cần torch/dữ liệu thật** (chạy được ngay sau `uv sync`).
- Hiện có `tests/test_smoke.py`: kiểm tra `import mmxai` và mọi subpackage có docstring.
- Test cần GPU/dữ liệu lớn: sẽ đánh dấu riêng (marker) khi xuất hiện, mặc định bỏ qua.
- Hook tuỳ chọn trong `.pre-commit-config.yaml` (ruff, nbstripout, chặn file > 1 MB): bật bằng `uv run pre-commit install`.

## Các phương án đã cân nhắc

| Phương án | Ưu | Nhược | Lý do không chọn |
|---|---|---|---|
| ruff (lint + format) | Một công cụ, cực nhanh, thay flake8 + isort + black | Ít quen hơn black với một số bạn | **Được chọn** |
| black + flake8 + isort | Lâu đời | 3 công cụ, 3 cấu hình, chậm | ruff tương thích black, gọn hơn |
| pylint | Kiểm tra rất kỹ | Chậm, nhiều cảnh báo nhiễu | Quá ồn cho đồ án |
| pytest | Chuẩn de facto, `assert` thuần, fixture `tmp_path` | — | **Được chọn** |
| unittest | Có sẵn | Dài dòng (class, `self.assertEqual`) | Khó viết nhanh |
| Thêm mypy | Bắt lỗi kiểu | Thêm thời gian, torch typing phức tạp | Chưa cần; xem lại sau |

## Hệ quả

- Mọi PR phải qua 3 lệnh trên; reviewer từ chối PR làm hỏng chúng.
- Dòng `# noqa: <mã>` chỉ dùng khi có lý do, ghi lý do ngay cạnh.
- Phiên bản ruff trong `.pre-commit-config.yaml` phải khớp bản trong `uv.lock`.

## Nguồn tham khảo

- Ruff: https://docs.astral.sh/ruff/
- pytest: https://docs.pytest.org/
- pre-commit: https://pre-commit.com/
