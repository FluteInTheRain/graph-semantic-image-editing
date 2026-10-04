# ADR-0012: Notebook mỏng, xoá output trước khi commit bằng nbstripout

- **Trạng thái:** Đã chấp nhận
- **Ngày:** 2026-11-01

## Bối cảnh

**Jupyter notebook** (`.ipynb`) tiện để thử và demo, nhưng có ba vấn đề khi làm nhóm:
1. File JSON chứa cả output (ảnh, bảng) → mỗi lần chạy lại tạo diff hàng nghìn dòng, xung đột merge khó gỡ.
2. Code trong notebook không test được, dễ chạy sai thứ tự ô.
3. Ảnh trong output làm repo phình to.

Đề (§3, §8) yêu cầu "notebook hoặc giao diện nhỏ: nhập truy vấn, trả về top-k và hiển thị đường đi giải thích".

## Quyết định

| Quy tắc | Chi tiết |
|---|---|
| Notebook mỏng | Chỉ `import mmxai` và gọi hàm; hàm > ~10 dòng phải chuyển vào `src/mmxai/` |
| Xoá output | `nbstripout` chạy qua hook pre-commit (`.pre-commit-config.yaml`); bật bằng `uv run pre-commit install` |
| Đặt tên | `notebooks/NN_mo-ta.ipynb`; `0x` = demo chính thức, `9x` = khám phá cá nhân |
| Chủ sở hữu | Mỗi notebook một người chịu trách nhiệm chính, tránh hai người sửa cùng lúc |
| Lint | ruff kiểm tra cả `.ipynb` (`extend-include` trong `pyproject.toml`) |
| Kết quả cần giữ | Hình/bảng xuất ra `results/figures/`, `results/tables/` thay vì để trong output |

Notebook demo chính (sẽ viết): `notebooks/01_demo_truy_van.ipynb` — nhập truy vấn → top-k ảnh → điểm cosine, điểm đồ thị, điểm cuối → đường đi giải thích.

## Các phương án đã cân nhắc

| Phương án | Ưu | Nhược | Lý do không chọn |
|---|---|---|---|
| Notebook mỏng + nbstripout (pre-commit) | Diff sạch, logic test được | Phải cài hook một lần | **Được chọn** |
| nbstripout dạng git filter (`nbstripout --install`) | Tự động, không cần pre-commit | Cấu hình ẩn trong `.git/config`, khó thấy | Gom chung vào pre-commit cho dễ giải thích |
| Jupytext (lưu `.py` song song) | Diff rất sạch | Hai file cho một notebook, dễ lệch | Thêm khái niệm |
| Giữ output | Xem được kết quả trên GitHub | Diff khổng lồ, xung đột | Không hợp làm nhóm |
| Không dùng notebook | Sạch | Đề cần demo trực quan | Notebook là cách demo được đề chấp nhận |

## Hệ quả

- Muốn xem demo phải chạy notebook (cần `uv sync --group ml --group graph --group demo` + artifact).
- Trước buổi chấm: chạy notebook demo, xuất bản HTML có output vào `reports/` làm bản dự phòng.

## Nguồn tham khảo

- nbstripout: https://github.com/kynan/nbstripout
- pre-commit: https://pre-commit.com/
- Đề bài, §3 và §8.
