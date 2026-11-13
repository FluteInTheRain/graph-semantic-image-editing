# Hồ sơ phiên 2026-11-13: mở rộng đánh giá cấp độ 1 (`_eval2`)

Thư mục này lưu **toàn bộ** tham số, kết quả từng lần chạy, bằng chứng kiểm tra trước khi khóa giao thức, `selection.json`, kết luận và kết quả review của nhánh `feat/level1-eval-extensions`.

- Phạm vi: bổ sung **nDCG@10** (nhị phân), chiều **ảnh → văn bản** (chọn `alpha_image_to_text` trên `val`), và **phân nhóm truy vấn** (`specificity`: `specific`/`generic`; `relation`: `action`/`spatial`/`none`), sau đó chạy lại 5 thực nghiệm (`_eval2`) dưới giao thức đã khóa tại ADR-0049 (`46bff84`).
- Bảng số liệu chính thức cho báo cáo nằm ở:
  - `results/tables/eval2_text_to_image.csv`
  - `results/tables/eval2_image_to_text.csv`
  - `results/tables/eval2_groups.csv`

---

## 1. Nội dung thư mục

| Đường dẫn | Nội dung |
|---|---|
| `pre_lock_checks/` | Kết quả kiểm tra trên `val` trước khi khóa giao thức (Task 8): `clip_vg_eval2_val_time.txt`, `inspect_query_groups_val.txt`, và bản chạy `20261113-1359_clip_vg_eval2_seed0` (khớp chính xác R@1/5/10 và MRR@100 với `clip_vg` seed 0) |
| `run_copies/` | Bản chép nguyên văn 30 thư mục chạy từ `results/runs/*_eval2*` (5 thực nghiệm × 3 `val` + 3 `test`), tất cả ở commit `46bff84`, `git_dirty: false` |
| `selection/` | 12 file `selection.json` (4 thực nghiệm có huấn luyện × 3 seed), lưu `best_epoch`, `alpha`, `alpha_image_to_text`, `alpha_sweep`, `alpha_sweep_image_to_text` |

---

## 2. Quyết định và giao thức

- **Spec & Plan:** `docs/superpowers/specs/2026-11-13-cap-do-1-mo-rong-design.md`, `docs/superpowers/plans/2026-11-13-cap-do-1-mo-rong.md`.
- **Khóa giao thức (ADR-0049, commit `46bff84`):**
  - nDCG@10 nhị phân, IDCG dùng `min(#relevant, 10)`.
  - Chiều `image_to_text` dùng cùng công thức điểm cặp `fuse_scores` và chọn `alpha_image_to_text` trên `val` theo `MRR@100` (tách biệt với `alpha` của `text_to_image`).
  - Phân nhóm truy vấn trên `val` trước khi khóa (`inspect_query_groups_val.txt`): `specific` 88,4% / `generic` 11,6%; `action` 8,5% / `spatial` 39,5% / `none` 52,0%. Giữ nguyên danh sách vị từ đóng băng từ ADR-0044 (không sửa rule).
  - Mọi config `configs/*_eval2.yaml` giữ nguyên siêu tham số từ bản gốc, chỉ thêm trường mở rộng và đổi `experiment`/`model_dir`.
  - Chạy đúng 3 seed (`0, 1, 2`), mỗi `(experiment, seed)` chạy `test` đúng 1 lần.

---

## 3. Tham số & cấu hình

5 config trong `configs/`:
- `clip_vg_eval2.yaml` (plain CLIP, `alpha_image_to_text = 1.0`)
- `clip_nograph_vg_eval2.yaml` (`use_graph: false`)
- `clip_graph_vg_lr1e4_eval2.yaml` (đồ thị `contains` top-16)
- `clip_cooc_graph_vg_lr1e4_eval2.yaml` (đồ thị `contains` + `cooccurrence` dựng trên train)
- `clip_graph_vg_lr1e4_shuffled_eval2.yaml` (ablation xáo cạnh `shuffle_seed: 0`)

Dữ liệu & split: `splits/visual_genome/split_seed0.json` (`sha256`: `0c65b3e5fa9fedc3c225e77806dc9832d5e2135cadba973fd3506d80a002c4ee`).

---

## 4. Kết quả từng lần chạy (`val` và `test`)

### 4.1 Lựa chọn trên `val` (`selection/*.json`)

| Experiment | Seed | `best_epoch` | `alpha` (t2i) | `alpha_image_to_text` |
|---|---|---|---|---|
| `clip_nograph_vg_eval2` | 0 / 1 / 2 | 1 / 1 / 1 | 0.3 / 0.4 / 0.2 | 0.7 / 0.6 / 0.6 |
| `clip_graph_vg_lr1e4_eval2` | 0 / 1 / 2 | 6 / 8 / 6 | 0.0 / 0.0 / 0.1 | 0.6 / 0.6 / 0.6 |
| `clip_cooc_graph_vg_lr1e4_eval2` | 0 / 1 / 2 | 5 / 5 / 6 | 0.0 / 0.1 / 0.0 | 0.6 / 0.6 / 0.6 |
| `clip_graph_vg_lr1e4_shuffled_eval2` | 0 / 1 / 2 | 7 / 9 / 8 | 0.1 / 0.2 / 0.1 | 0.6 / 0.6 / 0.6 |

### 4.2 Kết quả `test` từng seed (`run_copies/`, commit `46bff84`, `git_dirty: false`)

| Experiment | Seed | t2i R@1 | t2i R@5 | t2i R@10 | t2i MRR@100 | t2i nDCG@10 | i2t R@1 | i2t R@10 | i2t MRR@100 | i2t nDCG@10 |
|---|---|---|---|---|---|---|---|---|---|---|
| `clip_vg_eval2` | 0, 1, 2 | 0.0584 | 0.1422 | 0.2018 | 0.1076 | 0.1169 | 0.1280 | 0.3630 | 0.2068 | 0.0930 |
| `clip_nograph_vg_eval2` | 0 | 0.0691 | 0.1693 | 0.2409 | 0.1259 | 0.1384 | 0.1486 | 0.3886 | 0.2280 | 0.1026 |
| `clip_nograph_vg_eval2` | 1 | 0.0693 | 0.1727 | 0.2480 | 0.1268 | 0.1394 | 0.1468 | 0.3890 | 0.2276 | 0.1020 |
| `clip_nograph_vg_eval2` | 2 | 0.0682 | 0.1803 | 0.2480 | 0.1271 | 0.1382 | 0.1498 | 0.3910 | 0.2288 | 0.1029 |
| `clip_graph_vg_lr1e4_eval2` | 0 | 0.0707 | 0.1738 | 0.2480 | 0.1279 | 0.1384 | 0.1508 | 0.3954 | 0.2310 | 0.1043 |
| `clip_graph_vg_lr1e4_eval2` | 1 | 0.0689 | 0.1807 | 0.2510 | 0.1282 | 0.1389 | 0.1524 | 0.3988 | 0.2323 | 0.1050 |
| `clip_graph_vg_lr1e4_eval2` | 2 | 0.0716 | 0.1749 | 0.2488 | 0.1290 | 0.1399 | 0.1512 | 0.3942 | 0.2311 | 0.1043 |
| `clip_cooc_graph_vg_lr1e4_eval2` | 0 | 0.0698 | 0.1781 | 0.2474 | 0.1272 | 0.1379 | 0.1514 | 0.3974 | 0.2311 | 0.1045 |
| `clip_cooc_graph_vg_lr1e4_eval2` | 1 | 0.0714 | 0.1806 | 0.2466 | 0.1285 | 0.1390 | 0.1522 | 0.3942 | 0.2311 | 0.1042 |
| `clip_cooc_graph_vg_lr1e4_eval2` | 2 | 0.0714 | 0.1791 | 0.2470 | 0.1279 | 0.1379 | 0.1530 | 0.3954 | 0.2325 | 0.1049 |
| `clip_graph_vg_lr1e4_shuffled_eval2` | 0 | 0.0684 | 0.1755 | 0.2485 | 0.1268 | 0.1386 | 0.1520 | 0.3944 | 0.2307 | 0.1038 |
| `clip_graph_vg_lr1e4_shuffled_eval2` | 1 | 0.0686 | 0.1760 | 0.2489 | 0.1274 | 0.1392 | 0.1528 | 0.3968 | 0.2324 | 0.1044 |
| `clip_graph_vg_lr1e4_shuffled_eval2` | 2 | 0.0688 | 0.1763 | 0.2493 | 0.1274 | 0.1392 | 0.1510 | 0.3958 | 0.2305 | 0.1041 |

### 4.3 Đối chiếu `eval2_text_to_image.csv` với `results/tables/main.csv`

Các số R@1/5/10 và MRR@100 tái lập gần như trùng khớp với `main.csv` (lệch tối đa 0,0001 do sai số dấu phẩy động / môi trường chạy lại):
- `clip_vg_eval2` vs `clip_vg`: khớp 100% (`0.0584 / 0.1422 / 0.2018 / 0.1076`).
- `clip_nograph_vg_eval2` vs `clip_nograph_vg`: R@1/5/MRR khớp 100%, R@10 `0.2457` vs `0.2457` (chỉ std `0.0041` vs `0.0043`).
- `clip_graph_vg_lr1e4_eval2` vs `clip_graph_vg_lr1e4`: R@1 `0.0704` vs `0.0705`, R@5 `0.1765` vs `0.1765`, R@10 `0.2493` vs `0.2492`, MRR `0.1284` vs `0.1284`.
- `clip_cooc_graph_vg_lr1e4_eval2` vs `clip_cooc_graph_vg_lr1e4`: R@1 `0.0708` vs `0.0709`, R@5 `0.1793` vs `0.1792`, R@10 `0.2470` vs `0.2469`, MRR `0.1279` vs `0.1280`.
- `clip_graph_vg_lr1e4_shuffled_eval2` vs `clip_graph_vg_lr1e4_shuffled`: khớp 100% (`0.0686 / 0.1759 / 0.2489 / 0.1272`).

---

## 5. Kết luận và lưu ý diễn giải

- Cả 4 biến thể có học tham số đều vượt `clip_vg_eval2` rõ rệt ở cả hai chiều `text→image` và `image→text`, cũng như trên nDCG@10.
- Trên toàn tập `test`, chênh lệch giữa `clip_graph_vg_lr1e4_eval2`, `clip_cooc_graph_vg_lr1e4_eval2`, `clip_graph_vg_lr1e4_shuffled_eval2` và `clip_nograph_vg_eval2` khá nhỏ (nhiều cặp chênh nhỏ hơn `std`, ví dụ nDCG@10 `0.1391 ± 0.0007` vs `0.1390 ± 0.0003` vs `0.1387 ± 0.0006`). Không kết luận vượt trội trên toàn tập khi chênh lệch nhỏ hơn `std`.
- Khi tách nhóm (`eval2_groups.csv`), khác biệt rõ nhất nằm ở nhóm **`generic`** (11,28% số truy vấn `test`, 2.820 truy vấn): `clip_graph_vg_lr1e4_eval2` (`R@1 = 0.0424 ± 0.0080`, `MRR = 0.1082 ± 0.0040`) và `clip_cooc_graph_vg_lr1e4_eval2` (`R@1 = 0.0457 ± 0.0055`, `MRR = 0.1072 ± 0.0039`) cao hơn rõ so với `shuffled` (`R@1 = 0.0274 ± 0.0007`, `MRR = 0.0951 ± 0.0018`) và `nograph` (`R@1 = 0.0280 ± 0.0027`, `MRR = 0.0914 ± 0.0092`).

---

## 6. Review độc lập (trên `git diff feat/level1-ablation-explain...HEAD`)

- **`leakage-auditor`:** `KHÔNG THẤY RÒ RỈ` (đạt 7/7 quy tắc PDF §6: test tách biệt hoàn toàn, cạnh đồng xuất hiện chỉ đếm trên train, `alpha` và `alpha_image_to_text` chọn trên `val`, test chạy 1 lần, cùng `split_sha256`, 3 seed).
- **`experiment-reviewer`:** Đạt tất cả mục kỹ thuật (3 seed, mean ± std, đủ 5 mô hình/ablation, công thức nDCG/MRR/R@K đúng). Yêu cầu hoàn tất hồ sơ phiên (`README.md` + `run_copies/`) và lưu ý không suy diễn quá dữ liệu ở các cặp chênh `< std`.
- **`pr-test-analyzer`:** Không có lỗi Critical; chỉ ra 6 lỗ hổng test/guard mức Important (kiểm giá trị `old_config`, unit test `check_extension_settings`, lỗi và multi-chunk của `image_to_text`, đường lỗi/dirty run của `summarize_image_to_text`, guard `evaluate_clip(..., image_to_text=True, ndcg_k=None)`, và guard thiếu `alpha_image_to_text` trong `evaluate_test`). Đã kiểm chứng và bổ sung đầy đủ ở commit `4d3603a`.
- **Independent PR #9 Reviewer + PR Code Updater (vòng review lần 2):** Không có lỗi Blocking hay làm lệch số 30 run `_eval2`; chỉ ra 4 mục Important và 2 mục Minor về guard & validation (`_require_same_keys` trong `summarize_runs` và kiểm cột đồng nhất trong `write_table`; chặn `ndcg_k < 1` hoặc `ndcg_k > top_k`; gọi `check_extension_settings` trong `load_clip_eval_config` và chặn `group_inputs` thiếu `ndcg_k` ở đầu `evaluate_clip`; đối chiếu `config.yaml` trong `test_run_exists` để tránh khớp nhầm hậu tố tên experiment; chặn đầu vào rỗng/lệch độ dài trong `query_groups.py`, `image_to_text.py` và `extended.py`). Đã cập nhật toàn bộ kèm 12 unit test mới.

---

## 7. Dòng thời gian commit

- `352236b` — Thiết kế mở rộng đánh giá cấp độ 1
- `828fac3` — Kế hoạch triển khai
- `af6037d`..`4362560` — Tasks 1–7: `ranked_metrics`, `query_groups`, `image_to_text`, `extended`, `clip_baseline`/`graph_rerank`, `summary`, configs `_eval2`, `inspect_query_groups.py`
- `46bff84` — Task 9: Khóa giao thức v2 (ADR-0049, `results/tables/README.md`) trước khi chạy `_eval2`
- `88d29bb` — Task 10: Xuất 3 bảng `eval2_*.csv` sau khi chạy xong 5 thực nghiệm × (3 val + 3 test)
- `4d3603a` — Task 11: Bổ sung test và guard theo góp ý `pr-test-analyzer`
- `a87cd38` — Task 11: Hồ sơ phiên, cập nhật guide, checklist và nhật ký AI
