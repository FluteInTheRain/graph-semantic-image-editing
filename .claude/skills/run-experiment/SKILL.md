---
name: run-experiment
description: Chạy một thực nghiệm từ file config với ≥3 seed, tổng hợp mean ± std cho Recall@1/5/10 và MRR, ghi kết quả có thể truy vết (config, commit, seed, split). Chỉ chạy khi người dùng gọi /run-experiment.
argument-hint: "[configs/<file>.yaml] [seeds, mặc định 0 1 2]"
disable-model-invocation: true
---

# Chạy thực nghiệm có kiểm soát

Tham số: `$ARGUMENTS` – tham số đầu là file config; phần còn lại (nếu có) là danh sách seed.
Mặc định lấy khoá `seeds` trong config, không có thì `0 1 2` (ADR-0011). Ít hơn 3 seed thì dừng và hỏi lại (PDF §6 yêu cầu ≥ 3 seed cho thực nghiệm chính).

## Bước 1 – Kiểm tra trước khi chạy (không bỏ qua)

1. Đọc file config. Xác nhận nó trỏ tới file split trong `splits/` (đã commit) và **không** chứa
   giá trị chọn trên test.
2. Rà checklist rò rỉ trong `CLAUDE.md` cho đường đi của config này (train → val chọn `alpha`/siêu
   tham số → test chạy một lần). Nghi ngờ → dừng, đề xuất gọi subagent `leakage-auditor`.
3. `git status --short`: nếu có thay đổi chưa commit trong `src/` hoặc `configs/`, cảnh báo rằng kết
   quả sẽ không truy được về một commit; hỏi người dùng có muốn commit trước không.
4. Tìm điểm chạy trong `scripts/` (xem `scripts/README.md`). Nếu script cần thiết chưa tồn tại:
   dừng và báo – **không** tự viết pipeline mới trong skill này.

## Bước 2 – Chạy từng seed

Với mỗi seed, chạy lệnh theo quy ước của script, ví dụ:

```bash
uv run python scripts/evaluate.py --config <config> --seed <seed>
```

Output từng lần chạy theo ADR-0010: `results/runs/<YYYYMMDD-HHMM>_<experiment>_seed<k>/` (gitignored)
gồm `config.yaml`, `metrics.json`, `env.json` (commit hash, phiên bản thư viện, thiết bị), `run.log`. Dán lệnh và vài dòng cuối log thật.
Lệnh chạy lâu (trích đặc trưng, huấn luyện GNN) → chạy nền và báo người dùng, không chờ mù.

## Bước 3 – Tổng hợp

- Tính mean ± std mẫu (`ddof=1`, ADR-0011) cho Recall@1, Recall@5, Recall@10, MRR trên test
  với cấu hình đã chọn trên validation. Nếu có nhóm truy vấn trực tiếp/gián tiếp, báo tách riêng.
- Ghi/cập nhật bảng trong `results/tables/` (được commit) theo ADR-0010, ví dụ `main.csv`, `ablation.csv`.
- So sánh với baseline CLIP thuần trên **cùng split** nếu đã có; chênh lệch < std thì nói rõ là chưa có ý nghĩa.

## Bước 4 – Báo lại

Bảng kết quả, đường dẫn file, lệnh chạy lại được. Nhắc người dùng thêm dòng vào
`docs/ai-usage-log.md` nếu AI đã viết/sửa code cho thực nghiệm này, và cân nhắc gọi `experiment-reviewer`.
