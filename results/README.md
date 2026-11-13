# results/ — kết quả thí nghiệm

**Là gì?** Nơi lưu số liệu sau khi chạy đánh giá.

| Đường dẫn | Nội dung | Commit? |
|---|---|---|
| `results/runs/<ngày>_<thí-nghiệm>_seed<k>/` | Mỗi lần chạy: `config.yaml` đã dùng, `metrics.json`, `env.json`, log | Không (gitignore) |
| `results/tables/*.csv` | Bảng tổng hợp mean ± std qua ≥3 seed, dùng cho báo cáo | Có |
| `results/records/<ngày>_<việc>/` | Hồ sơ một phiên: bản chép `config.yaml`/`metrics.json`/`env.json` của các run, `selection.json`, meta đồ thị, và `README.md` ghi tham số, kết quả, suy luận, quyết định, review. Để truy vết, không thay `results/tables/` | Có |
| `results/figures/*.png` | Hình cho báo cáo/slide | Có (nếu nhỏ) |

Chi tiết: `docs/decisions/0010-theo-doi-thi-nghiem-bang-file.md`.

## Đọc `results/tables/main.csv`

- Sinh bằng `uv run python scripts/summarize_runs.py`, không sửa tay. Ngoại lệ duy nhất: dòng `clip_graph_vg_lr1e4_shuffled`. Dòng này được chép nguyên văn từ đầu ra của script ghi ra file tạm, vì máy chạy ablation không có run test của các dòng khác (`results/tables/README.md`, mục "Ablation đồ thị ngẫu nhiên"). Mỗi dòng là một thí nghiệm trên tập `test`, gồm `<metric>_mean` và `<metric>_std` (std mẫu, `ddof=1`) qua các seed, kèm hash file split và commit đã chạy.
- **`clip_vg` có std = 0** vì CLIP đóng băng và FAISS Flat là tất định: 3 seed cho cùng một số, không phải 3 lần đo độc lập. Khi so sánh với CLIP + Graph, không dùng tiêu chí "chênh lệch < std" cho dòng này; cần kiểm định theo từng truy vấn (ADR-0043).
- Giao thức (truy vấn, đáp án đúng, `top_k = 100`, MRR@100): ADR-0043. Recall tuyệt đối là cận dưới vì chỉ xét 5 region đã lấy mẫu mỗi ảnh.
