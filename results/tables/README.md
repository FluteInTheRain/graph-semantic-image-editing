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
