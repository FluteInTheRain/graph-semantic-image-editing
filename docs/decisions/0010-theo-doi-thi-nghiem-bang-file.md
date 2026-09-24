# ADR-0010: Theo dõi thí nghiệm bằng file JSON/CSV, không dùng W&B/MLflow

- **Trạng thái:** Đã chấp nhận
- **Ngày:** 2026-11-01

## Bối cảnh

**Theo dõi thí nghiệm (experiment tracking)** = ghi lại mỗi lần chạy: config nào, seed nào, ra số bao nhiêu. Thiếu nó thì cuối kỳ không biết số trong báo cáo đến từ lần chạy nào.

Nhu cầu của đồ án: vài chục lần chạy (CLIP, CLIP+Graph, 3 GNN, 3 seed, ablation, độ nhạy `alpha`), cần bảng **mean ± std** cho báo cáo, chạy trên nhiều máy (Mac, Colab).

## Quyết định

Mỗi lần chạy tạo một thư mục:

```
results/runs/<YYYYMMDD-HHMM>_<experiment>_seed<k>/   # gitignore
    config.yaml     # config thực sự đã dùng (sau ghi đè)
    metrics.json    # {"split": "test", "R@1": ..., "R@5": ..., "R@10": ..., "MRR": ..., "nDCG@10": ...,
                    #  "by_type": {"direct": {...}, "indirect": {...}}, "alpha": ...}
    env.json        # git commit, python, torch, open_clip, thiết bị
    run.log
```

Bảng tổng hợp **được commit**:

| File | Nội dung |
|---|---|
| `results/tables/main.csv` | CLIP vs CLIP+Graph, mean ± std qua ≥ 3 seed |
| `results/tables/ablation.csv` | Bỏ cạnh / xáo cạnh / đồ thị ngẫu nhiên |
| `results/tables/alpha_sensitivity.csv` | Metric theo `alpha` (trên val) |
| `results/tables/explain.csv` | Fidelity, validity, sparsity |
| `results/figures/*.png` | Hình cho báo cáo |

Script tổng hợp (sẽ viết) đọc mọi `metrics.json` → sinh CSV. Số trong báo cáo **chỉ** lấy từ `results/tables/`.

TensorBoard: chỉ dùng tạm khi debug đường cong loss, không bắt buộc, log nằm trong `runs/` (gitignore).

## Các phương án đã cân nhắc

| Phương án | Ưu | Nhược | Lý do không chọn |
|---|---|---|---|
| File JSON/CSV trong repo | Không cần tài khoản, chạy offline, diff được, nộp kèm được | Không có giao diện đẹp | **Được chọn** |
| Weights & Biases | Dashboard đẹp, so sánh dễ | Cần tài khoản/mạng, dữ liệu ở server ngoài, khó nộp | Phụ thuộc dịch vụ ngoài |
| MLflow | Mã nguồn mở, có UI | Thêm server/thư mục `mlruns`, học thêm | Nặng so với nhu cầu |
| TensorBoard làm chính | Có sẵn với torch | Hợp đường cong, không hợp bảng tổng hợp | Chỉ dùng phụ |

## Hệ quả

- Giảng viên có thể truy từ số trong báo cáo → CSV → `metrics.json` → `config.yaml` + git commit.
- Mỗi thành viên chạy trên máy mình vẫn ra cùng định dạng; gộp bằng cách copy thư mục run.
- Thư mục run lớn không commit; nếu cần làm minh chứng, nén và đưa lên link cloud cùng artifact.

## Nguồn tham khảo

- Papers with Code, ML Reproducibility Checklist: https://github.com/paperswithcode/releasing-research-code
- Đề bài, §6 (≥ 3 seed) và §7 (bảng đánh giá).
