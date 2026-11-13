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

## Giao thức đánh giá v2 (chốt 2026-11-13, trước khi huấn luyện và chạy test `_eval2`)

Quyết định đầy đủ: ADR-0049. Đây là phần mở rộng của Nhiệm vụ 1 (PDF §7): nDCG@10, chiều ảnh → văn bản, và đánh giá theo nhóm truy vấn.

| Bảng | Nội dung |
|---|---|
| `eval2_text_to_image.csv` | Văn bản → ảnh: R@1/5/10, MRR@100, nDCG@10 của 5 experiment `_eval2` |
| `eval2_image_to_text.csv` | Ảnh → văn bản: cùng các chỉ số; cột `alphas` là `α` chọn trên val của từng seed (CLIP thuần: 1) |
| `eval2_groups.csv` | Văn bản → ảnh tách theo nhóm: `specificity` (`specific`/`generic`) và `relation` (`action`/`spatial`/`none`) |

Quy tắc, ghi lại trước khi có bất kỳ số nào của `_eval2`:

- 5 experiment: `clip_vg_eval2`, `clip_nograph_vg_eval2`, `clip_graph_vg_lr1e4_eval2`, `clip_cooc_graph_vg_lr1e4_eval2`, `clip_graph_vg_lr1e4_shuffled_eval2`. Config giống hệt bản gốc, chỉ khác tên, `model_dir` và các trường mở rộng; có test kiểm.
- Huấn luyện và mọi lựa chọn (epoch, `α` hai chiều) trên **val**. Sau đó chạy test **đúng một lần** cho mỗi seed 0, 1, 2. Báo cáo mọi số, dù kết quả ra sao.
- Luật chia nhóm đã chốt trên val (ADR-0049, mục 3); không đổi sau thời điểm này.
- Các dòng cũ trong `main.csv` giữ nguyên. Nếu R@K/MRR của `_eval2` lệch so với dòng cũ, ghi mức lệch và lý do (huấn luyện lại, khác máy) vào hồ sơ `results/records/<ngày>_cap-do-1-mo-rong/`.
- Ba bảng sinh bằng `scripts/summarize_runs.py --experiment-suffix _eval2` (tham số `--kind main | groups | image_to_text`), không sửa tay.
