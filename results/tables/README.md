# Bảng kết quả

Số trong báo cáo chỉ lấy từ các file ở đây (ADR-0010). Mỗi dòng là mean ± std mẫu qua 3 seed, trên tập **test**, cùng file split (`split_sha256` trong cột cuối).

## `main.csv`: dòng nào là cấu hình chính

Quy tắc (chốt 2026-11-12, commit `6c97c4c`, trước khi chạy test `*_lr1e4`): mỗi mô hình dùng `lr` có điểm kết hợp **val** cao nhất.

| Dòng (`experiment`) | `lr` | Vai trò |
|---|---|---|
| `clip_vg` | – | Baseline CLIP thuần (ADR-0043) |
| `clip_nograph_vg` | 1e-3 | **Chính**: đối chứng không đồ thị (val: 1e-3 tốt hơn 1e-4) |
| `clip_graph_vg_lr1e4` | 1e-4 | **Chính**: CLIP + Graph một lớp (ADR-0046) |
| `clip_cooc_graph_vg_lr1e4` | 1e-4 | **Chính**: CLIP + Graph có cạnh đồng xuất hiện (ADR-0048) |
| `clip_graph_vg` | 1e-3 | Test cũ, chạy **trước** khi có quy tắc; không phải cấu hình chính. Giữ lại để minh bạch |

Không có dòng `clip_nograph_vg_lr1e4` hay `clip_cooc_graph_vg` (1e-3): theo quy tắc, các bản đó không được chạy test.

Vì sao có hai dòng cho mô hình một lớp, và giới hạn của việc đó: `docs/guides/huan-luyen-va-xep-hang-lai.md`, mục 7.

## Ablation đồ thị ngẫu nhiên (chốt 2026-11-13, trước khi huấn luyện và chạy test)

| Dòng (`experiment`) | Vai trò |
|---|---|
| `clip_graph_vg_lr1e4_shuffled` | **Ablation** (ADR-0046 §6, PDF §5 cấp độ 1): mô hình chính `clip_graph_vg_lr1e4`, huấn luyện y hệt, trên đồ thị đã xáo cạnh `contains` giữ bậc trong từng split (`configs/graph_vg_clip_topk_shuffled.yaml`, `shuffle_seed: 0`) |

Quy tắc, ghi lại trước khi có bất kỳ số nào của ablation:

- Siêu tham số giống hệt mô hình chính (test `test_shuffled_ablation_config_differs_from_the_main_config_only_in_the_graph` kiểm). Không chỉnh gì cho ablation, kể cả khi số val xấu.
- Huấn luyện, early stopping và chọn `alpha` trên **val** như ADR-0047; sau đó chạy test **đúng một lần** mỗi seed (0, 1, 2) và báo cáo dòng này dù kết quả ra sao.
- Chỉ một đồ thị xáo (`shuffle_seed: 0`). Không thử thêm seed xáo hay biến thể khác rồi chọn bản đẹp.
- So sánh cần đọc: `clip_graph_vg_lr1e4 − clip_graph_vg_lr1e4_shuffled` (vai trò của cấu trúc quan hệ thật) và `clip_graph_vg_lr1e4_shuffled − clip_nograph_vg`.
- Dòng này được thêm vào `main.csv` bằng tay từ đầu ra của `scripts/summarize_runs.py --output <file tạm>`, vì `results/runs/` trên máy chạy ablation không có run test của các dòng khác; các dòng cũ giữ nguyên.
