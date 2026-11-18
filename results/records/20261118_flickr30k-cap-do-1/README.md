# Hồ sơ: pipeline cấp độ 1 trên Flickr30k (val + test)

Hồ sơ này lưu kết quả của lần chạy lại pipeline cấp độ 1 trên tập thứ hai là Flickr30k. Pipeline giữ nguyên, chỉ đổi dữ liệu (ADR-0050).

- Máy chạy: Kaggle, notebook `notebooks/20_kaggle_graph_pipeline.ipynb` với `DATASET = "flickr30k"` và `RUN_LEVEL1_TEST = True`.
- Commit: `fb62f69`, nhánh `feat/run-pipeline-on-flickr30k`. Mọi run đều có `git_dirty: false`.
- Đã chạy: huấn luyện 3 config × 3 seed, CLIP thuần, giải thích cơ bản trên val, và **test một lần** mỗi seed cho cả 4 mô hình.
- Bảng: `results/tables/level1_flickr30k_test.csv`, `results/tables/level1_flickr30k_val.csv`, sinh bằng `scripts/summarize_runs.py --runs-dir <run_copies> --split {test,val}`; số không sửa tay.
- Số val trùng từng chữ số với lần chạy Kaggle trước (20261116), vì huấn luyện tất định với cùng seed.

## Nội dung thư mục

| Đường dẫn | Là gì |
|---|---|
| `run_copies/` | Bản chép nguyên văn `results/runs/*flickr30k*` từ Kaggle: `config.yaml`, `metrics.json`, `env.json` |
| `run_copies/*_explain_seed0/` | `scripts/explain.py` trên val, seed 0, `--num-edges 3`: `analysis.{json,md}` |
| `graphs/*.local.meta.json` | Meta của hai đồ thị dựng trên máy local (đồ thị gốc và đồ thị xáo `shuffle_seed: 0`). Kaggle dựng lại đồ thị bằng cùng config |

## Kết quả test (25.000 truy vấn, 5.000 ảnh, 3 seed)

| Mô hình | Epoch | `alpha` | R@1 | R@5 | R@10 | MRR@100 |
|---|---|---|---|---|---|---|
| `clip_flickr30k` | – | 1 | 0,4901 ± 0,0000 | 0,7426 ± 0,0000 | 0,8250 ± 0,0000 | 0,6045 ± 0,0000 |
| `clip_nograph_flickr30k` | 1/1/1 | 0,6/0,6/0,6 | 0,5332 ± 0,0006 | 0,7825 ± 0,0005 | 0,8556 ± 0,0011 | 0,6447 ± 0,0003 |
| `clip_graph_flickr30k_lr1e4` | 6/5/2 | 0,5/0,5/0,5 | 0,5309 ± 0,0036 | 0,7811 ± 0,0022 | 0,8553 ± 0,0026 | 0,6432 ± 0,0028 |
| `clip_graph_flickr30k_lr1e4_shuffled` | 3/3/2 | 0,5/0,5/0,4 | 0,5276 ± 0,0014 | 0,7775 ± 0,0014 | 0,8526 ± 0,0009 | 0,6399 ± 0,0015 |

## Kết quả val

| Mô hình | R@1 | R@5 | R@10 | MRR@100 |
|---|---|---|---|---|
| `clip_flickr30k` (seed 0) | 0,4891 | 0,7368 | 0,8168 | 0,6016 |
| `clip_nograph_flickr30k` | 0,5288 ± 0,0003 | 0,7785 ± 0,0010 | 0,8519 ± 0,0008 | 0,6403 ± 0,0000 |
| `clip_graph_flickr30k_lr1e4` | 0,5298 ± 0,0026 | 0,7782 ± 0,0038 | 0,8517 ± 0,0028 | 0,6408 ± 0,0026 |
| `clip_graph_flickr30k_lr1e4_shuffled` | 0,5269 ± 0,0002 | 0,7753 ± 0,0013 | 0,8495 ± 0,0012 | 0,6383 ± 0,0007 |

## MRR@100 ghép cặp theo seed

| Tập | Hiệu | Seed 0 | Seed 1 | Seed 2 | Mean ± std |
|---|---|---|---|---|---|
| Test | Thật − không đồ thị | +0,00032 | −0,00006 | −0,00474 | −0,00149 ± 0,00282 |
| Test | Thật − xáo | +0,00444 | +0,00354 | +0,00184 | +0,00327 ± 0,00132 |
| Test | Xáo − không đồ thị | −0,00412 | −0,00360 | −0,00658 | −0,00477 ± 0,00159 |
| Val | Thật − không đồ thị | +0,00222 | +0,00191 | −0,00253 | +0,00053 ± 0,00266 |
| Val | Thật − xáo | +0,00400 | +0,00328 | +0,00023 | +0,00250 ± 0,00200 |
| Val | Xáo − không đồ thị | −0,00178 | −0,00137 | −0,00276 | −0,00197 ± 0,00072 |

## Giải thích cơ bản (val, seed 0)

- MRR@100 trong phân tích là 0,6424968…, bằng số lúc huấn luyện seed 0 (chốt chặn `check_reproduces_training` chạy qua).
- 25.000 truy vấn: hạng tăng 7.306, giảm 3.557, giữ nguyên 13.288; 849 truy vấn có ảnh đúng ngoài top 100.
- Câu truy vấn không hiện vì input Kaggle thiếu `dataset_flickr30k.json`; truy vấn ghi bằng id ảnh và vị trí caption.
- Xoá 3 cạnh do Occlusion chọn làm điểm đồ thị giảm ở cả 15 ảnh được giải thích.
- Bỏ mọi cạnh: 5 ảnh thành công tụt từ hạng 5/8/11/14/9 xuống 37/25/15/44/50; ở 5 ca thất bại ảnh đúng tụt thêm 81→89, 83→97, 78→90, 70→95, 61→84.
