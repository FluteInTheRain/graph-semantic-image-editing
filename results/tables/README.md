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

## xAI cấp độ 3: đánh giá giải thích (chốt explainer trên val 2026-11-16, trước khi chạy test)

| Bảng | Nội dung |
|---|---|
| `xai_level3.csv` | Mỗi dòng là một (explainer, `m`): Fid+, Fid+ ngẫu nhiên, hạng tụt, hạng tụt ngẫu nhiên, Fid−, characterization, robust Fid+, sparsity, validity (`connected_share`, `all_in_reach_share`), p-value Wilcoxon so với ngẫu nhiên. Mean ± std qua 3 seed, 20 truy vấn. 9 dòng `val` (3 explainer × `m` ∈ {3, 5, 10}) và 1 dòng `test` |
| `xai_level3_test_queries.csv` | Bằng chứng theo từng truy vấn trên test (20 truy vấn, trung bình 3 seed): Fid+, hạng tụt, cùng số đó khi xoá ngẫu nhiên |

Quy tắc:

- Explainer và `m` chọn trên **val** bằng characterization trung bình (`statistics.choose_on_validation`): `integrated_gradients`, `m = 10`. Ghi vào `configs/explain_vg.yaml` ở commit `2ba21cf`, **trước** khi chạy test.
- Test chạy **đúng một lần** (`results/runs/20261116-2245_explain_vg_test`, commit `8a2f137`, `git_dirty: false`). Script từ chối lần chạy test thứ hai.
- Dòng `val` lấy từ lần chạy lại trên val (`20261116-2252_explain_vg_val`, commit `59d088c`) với **đúng checkpoint của lần test**, nên val và test dùng cùng mô hình. Lần chạy lại cho cùng lựa chọn (`integrated_gradients`, `m = 10`) như lần val đầu trên Kaggle (`20261114-0453_explain_vg_val`, checkpoint train lại trên Kaggle); hai lần lệch nhau ≤ 0,001 ở hạng tụt ngẫu nhiên, các cột khác trùng tới 4 chữ số.
- Bản sao các run: `results/records/20261114_xai-cap-do-3/run_copies/`. Hai CSV sinh từ `evaluation.json` của `20261116-2252_explain_vg_val` và `20261116-2245_explain_vg_test`, không sửa tay số.

## Flickr30k: pipeline cấp độ 1 trên tập thứ hai

Quyết định: ADR-0050. Pipeline giữ nguyên, chỉ đổi dữ liệu.

| Bảng | Nội dung |
|---|---|
| `level1_flickr30k_test.csv` | Mean ± std qua 3 seed trên **test** Flickr30k: `clip_flickr30k`, `clip_nograph_flickr30k`, `clip_graph_flickr30k_lr1e4`, `clip_graph_flickr30k_lr1e4_shuffled` |
| `level1_flickr30k_val.csv` | Như trên nhưng trên **val** (tập chọn epoch và `alpha`), chỉ 3 mô hình có học |

- Test chạy **đúng một lần** mỗi seed, cùng phiên Kaggle với huấn luyện (commit `fb62f69`, `git_dirty: false`), bằng checkpoint và `alpha` đã chọn trên val.
- CLIP thuần chỉ chạy seed 0 trên val, nên không có trong bảng val (`summarize_runs.py` cần ≥ 3 seed). Số của nó nằm trong hồ sơ.
- Hai bảng sinh bằng `scripts/summarize_runs.py --split {val,test}` trên bản chép các run Kaggle (`results/records/20261118_flickr30k-cap-do-1/run_copies/`); không sửa tay.

## `g_retriever_test.csv`: G-Retriever bậc 2 (ADR-0051)

Tạo bằng `uv run python scripts/summarize_runs.py --runs-dir <thư mục chỉ chứa 6 run test dưới đây> --split test --output results/tables/g_retriever_test.csv`. Không ghi đè `main.csv`, vì máy chạy không có các run cũ.

| Dòng (`experiment`) | `lr` | Vai trò |
|---|---|---|
| `clip_pcst_sage_vg` | 1e-4 | G-Retriever bậc 2: GraphSAGE chỉ gộp khái niệm trong đồ thị con PCST (`k = 3`, `C = 0,5` cố định theo bài gốc) |
| `clip_nograph_vg_lr1e4` | 1e-4 | Đối chứng **cùng cấu hình huấn luyện** với dòng trên, đúng như thiết kế bậc 2 trong ADR-0051 |

Ghi chú minh bạch:
- Theo quy tắc chọn `lr` ở mục `main.csv`, đối chứng không đồ thị **chính thức** vẫn là `clip_nograph_vg` (1e-3, val tốt hơn 1e-4).
- Dòng `clip_nograph_vg_lr1e4` ở đây được chạy test ngày 2026-11-17, **sau khi nhóm đồng ý**, chỉ để so cùng cấu hình với G-Retriever. Không có lựa chọn nào dựa trên số test này.
- Run: `20261117-1300/1305/1309_clip_pcst_sage_vg_seed0/1/2`, `20261117-1301/1305/1310_clip_nograph_vg_lr1e4_seed0/1/2`. Commit code `4afb770`, cùng `split_sha256` với `main.csv`.

