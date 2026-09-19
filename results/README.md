# results/ — kết quả thí nghiệm

**Là gì?** Nơi lưu số liệu sau khi chạy đánh giá.

| Đường dẫn | Nội dung | Commit? |
|---|---|---|
| `results/runs/<ngày>_<thí-nghiệm>_seed<k>/` | Mỗi lần chạy: `config.yaml` đã dùng, `metrics.json`, `env.json`, log | Không (gitignore) |
| `results/tables/*.csv` | Bảng tổng hợp mean ± std qua ≥3 seed, dùng cho báo cáo | Có |
| `results/figures/*.png` | Hình cho báo cáo/slide | Có (nếu nhỏ) |

Chi tiết: `docs/decisions/0010-theo-doi-thi-nghiem-bang-file.md`.
