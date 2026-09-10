# ADR-0003: src layout với package `mmxai` chia theo các bước pipeline

- **Trạng thái:** Đã chấp nhận
- **Ngày:** 2026-11-01

## Bối cảnh

**Layout** = cách sắp xếp thư mục. **src layout** = đặt code vào `src/<tên_package>/` thay vì để ở gốc repo. Lợi ích chính: test luôn chạy trên package **đã cài** (`uv sync` cài ở chế độ editable), nên lỗi "chạy được trên máy tôi vì đang đứng ở thư mục gốc" không xảy ra.

Pipeline theo đề (§4.2): truy vấn → mã hoá CLIP → top-k theo cosine (FAISS) → điểm đồ thị (GNN) → xếp hạng lại → giải thích → đánh giá. Người đọc mới cần nhìn cây thư mục là thấy ngay pipeline.

## Quyết định

Tên package: `mmxai` (multimodal + xAI). Mỗi bước pipeline là một subpackage:

| Subpackage | Trách nhiệm | Mục đề bài |
|---|---|---|
| `src/mmxai/data/` | Đọc Visual Genome, FB15k-237-IMG, MKG-W/Y, MMKG tự xây; đọc file split | §4.1, §6 |
| `src/mmxai/features/` | Trích đặc trưng CLIP/OpenCLIP đóng băng, lưu cache | §4.1 |
| `src/mmxai/index/` | Dựng/nạp chỉ mục FAISS, lấy top-k ứng viên | §4.1 |
| `src/mmxai/graph/` | Dựng đồ thị (ảnh - đối tượng - khái niệm), liên kết thực thể | §4.2 |
| `src/mmxai/models/` | GNN (GCN/GAT/GraphSAGE), graph scorer, phương pháp cấp độ 2 | §5 |
| `src/mmxai/rerank/` | Kết hợp điểm `alpha·cosine + (1−alpha)·graph` | §4.2 |
| `src/mmxai/explain/` | Đường đi/quan hệ giải thích; fidelity, validity, sparsity | §5 cấp 3 |
| `src/mmxai/eval/` | Recall@K, MRR, nDCG; tách truy vấn trực tiếp/gián tiếp | §7 |
| `src/mmxai/utils/` | Seed, đọc config, I/O, logging | — |

Thư mục cấp gốc:

| Thư mục | Nội dung | Commit? |
|---|---|---|
| `configs/` | YAML thí nghiệm (ADR-0007) | có |
| `scripts/` | CLI mỏng gọi vào `mmxai` | có |
| `notebooks/`, `app/` | Demo (ADR-0012, ADR-0013) | có |
| `tests/` | pytest | có |
| `splits/`, `queries/` | File chia, bộ truy vấn đánh giá (ADR-0009) | có |
| `mmkg/`, `data_cards/` | MMKG tiếng Việt, data card (ADR-0015) | có (trừ ảnh) |
| `results/`, `reports/` | Bảng kết quả, báo cáo, slide (ADR-0010) | có (trừ `results/runs/`) |
| `data/`, `artifacts/` | Dữ liệu thô, cache, checkpoint (ADR-0008) | **không** |
| `docs/decisions/` | ADR | có |

Quy tắc phụ thuộc: các bước sau được import bước trước, không ngược lại (vd. `eval` không import `models`). `utils` không import subpackage nào khác.

## Các phương án đã cân nhắc

| Phương án | Ưu | Nhược | Lý do không chọn |
|---|---|---|---|
| src layout, chia theo bước pipeline | Khớp đề, dễ chia việc theo người, test chạy trên bản cài | Nhiều thư mục lúc đầu còn rỗng | **Được chọn** |
| Flat layout (`mmxai/` ở gốc) | Đơn giản | Dễ import nhầm code chưa cài, lỗi ẩn khi đóng gói | Hướng dẫn PyPA và Scientific Python khuyên src |
| Cookiecutter Data Science v2 nguyên mẫu | Chuẩn cộng đồng | Thiên về data science dạng bảng, thiếu chỗ cho splits/queries/explain | Mượn ý tưởng (data/, reports/, notebooks/) nhưng tuỳ biến |
| Chia theo loại (`train.py`, `models.py`, `utils.py` ở gốc) | Ít file | Phình to, khó chia việc, khó test | Không mở rộng được khi thêm cấp 2, 3 |

## Hệ quả

- Mỗi thành viên có thể "sở hữu" 1–2 subpackage → lịch sử commit rõ ràng (ADR-0014).
- Import luôn là `from mmxai.<bước> import ...`; script/notebook không chứa logic.
- Thêm bước mới (vd. video) = thêm subpackage, ghi ADR mới.

## Nguồn tham khảo

- PyPA, "src layout vs flat layout": https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/
- Scientific Python Development Guide: https://learn.scientific-python.org/development/guides/packaging-simple/
- Cookiecutter Data Science v2: https://cookiecutter-data-science.drivendata.org/
- Đề bài, §4.2 (quy trình tổng thể).
