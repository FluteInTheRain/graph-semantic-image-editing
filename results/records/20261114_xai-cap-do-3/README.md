# Hồ sơ 2026-11-14: đánh giá giải thích cấp độ 3 trên val

Bằng chứng truy vết cho kết quả val của xAI cấp độ 3. Chạy trên Kaggle bằng `notebooks/20_xai_level3_kaggle.ipynb`
(commit `41e4c9e`), output công khai: https://www.kaggle.com/code/thiennp/20-xai-level3-kaggle/output

| Thư mục trong `run_copies/` | Là gì |
|---|---|
| `20261114-0453_explain_vg_val/` | Đánh giá giải thích trên val: `summary.md` (bảng mean ± std qua 3 seed), `evaluation.json` (540 bản ghi), `config.yaml`, `env.json` |
| `20261116-2252_explain_vg_val/` | Chạy lại val trên máy thành viên với **đúng checkpoint của lần test** (commit `59d088c`). Cùng lựa chọn, số trùng lần Kaggle tới 4 chữ số (hạng tụt ngẫu nhiên lệch ≤ 0,001). Đây là nguồn dòng `val` trong `results/tables/xai_level3.csv` |
| `20261116-2245_explain_vg_test/` | Lần chạy test **duy nhất** với `integrated_gradients`, `m = 10` (chốt trên val, commit `2ba21cf`); chạy trên máy thành viên, commit `8a2f137` |
| `20261114-04xx_clip_graph_vg_lr1e4_seed{0,1,2}/` | Lần train lại mô hình `clip_graph_vg_lr1e4` trên Kaggle (chỉ val) |

## Tóm tắt

- Occlusion và Integrated Gradients: bỏ cạnh trong giải thích làm điểm giảm và ảnh tụt hạng rõ hơn xoá ngẫu nhiên
  (Wilcoxon một phía, p ≤ 0.002, mọi `m`). GNNExplainer không hơn ngẫu nhiên.
- Chốt trên val: `integrated_gradients`, `m = 10` (characterization cao nhất; IG gần như hoà Occlusion).
- Test (2026-11-16): Fid+ 0,0225 ± 0,0010 so với 0,0006 khi xoá ngẫu nhiên; hạng tụt 2,57 ± 0,29 so với 0,26
  (p = 0,00049). Bảng: `results/tables/xai_level3.csv`.

## Lưu ý

- Mô hình được **train lại** trên Kaggle, đồ thị được dựng lại; không phải checkpoint của
  `results/records/20261113_cap-do-1/`. `alpha` (0, 0, 0.1) và `best_epoch` (6, 8, 6) trùng lần train gốc; MRR@100 val
  lệch ≤ 2e-5 (khác torch/phần cứng), nên không trùng từng bit.
- Số cho báo cáo lấy từ `results/tables/` (ADR-0010); thư mục này chỉ là bằng chứng truy vết.
