# Hồ sơ phiên 2026-11-13: hoàn thiện cấp độ 1

Thư mục này lưu **toàn bộ** tham số, kết quả, kết luận, suy luận và quyết định của phiên làm việc 2026-11-13 trên
nhánh `feat/level1-ablation-explain`. Mục đích: không có gì chỉ nằm trong lịch sử chat.

- Phiên này làm ba việc: ablation đồ thị xáo cạnh, giải thích bằng Occlusion, và phân tích 5 thành công + 5 thất bại.
- Người làm: FluteInTheRain, có Claude Code (Opus 5.5) hỗ trợ. Nhật ký AI: `docs/ai-usage-log.md`, dòng 2026-11-13.
- Đường dẫn tính từ gốc repo. Số dòng code có thể lệch khi code đổi; nếu lệch, tìm theo tên hàm.
- Số cho báo cáo vẫn lấy từ `results/tables/main.csv` (ADR-0010). Thư mục này là **bằng chứng truy vết**.

## Mục lục

1. [Nội dung thư mục](#1-nội-dung-thư-mục)
2. [Quyết định và ai quyết định](#2-quyết-định-và-ai-quyết-định)
3. [Môi trường và dữ liệu](#3-môi-trường-và-dữ-liệu)
4. [Tham số](#4-tham-số)
5. [Kết quả: huấn luyện lại mô hình chính (chỉ val)](#5-kết-quả-huấn-luyện-lại-mô-hình-chính-chỉ-val)
6. [Kết quả: ablation đồ thị xáo cạnh](#6-kết-quả-ablation-đồ-thị-xáo-cạnh)
7. [Kết quả: giải thích và 5 + 5 truy vấn](#7-kết-quả-giải-thích-và-5--5-truy-vấn)
8. [Kết luận và suy luận](#8-kết-luận-và-suy-luận)
9. [Review độc lập và cách xử lý](#9-review-độc-lập-và-cách-xử-lý)
10. [Sửa lỗi môi trường và test](#10-sửa-lỗi-môi-trường-và-test)
11. [Việc chưa làm](#11-việc-chưa-làm)
12. [Dòng thời gian commit](#12-dòng-thời-gian-commit)

## 1. Nội dung thư mục

| Đường dẫn | Là gì |
|---|---|
| `run_copies/<thời điểm>_<experiment>_seed<k>/` | Bản chép nguyên văn của `results/runs/` (thư mục đó bị gitignore): `config.yaml` đã dùng, `metrics.json`, `env.json` (commit, phiên bản thư viện). Tên `run_copies` chứ không phải `runs`, vì `.gitignore` bỏ qua mọi thư mục tên `runs/` |
| `run_copies/*_explain_seed0/` | Ba lần chạy `scripts/explain.py`: `1002` (trước khi sửa lỗi câu trùng), `1004` (sau khi sửa), `1036` (sau review PR #8: thêm "bỏ mọi cạnh" và chốt chặn khớp `selection.json`; truy vấn, giải thích và kiểm tra xoá giống hệt `1004`) |
| `diagnostics/global_noedges.py` + `_output.txt` | Chẩn đoán (chỉ val): MRR@100 của từng seed khi bỏ mọi cạnh của đồ thị val, cùng trọng số. Chạy một lần để trả lời review, không chọn gì từ nó |
| `selection/<experiment>_seed<k>.json` | `selection.json` khi huấn luyện: epoch tốt nhất, lịch sử val theo epoch, MRR@100 val theo từng `alpha` |
| `graphs/*.meta.json` | Meta của đồ thị gốc và đồ thị xáo (số cạnh, hash split, hash `graph.npz` gốc, commit) |

Không lưu `model.pt`, đồ thị `.npz` hay đặc trưng: chúng nằm trong `artifacts/` (gitignore, `CLAUDE.md` cấm commit).
Hash của từng mô hình đã chạy test nằm trong `metrics.json` (`model_sha256`).

## 2. Quyết định và ai quyết định

| Ngày | Quyết định | Ai |
|---|---|---|
| 2026-11-13 | Làm 3 mục cấp độ 1 còn thiếu (ablation, giải thích, 5 + 5), chia việc cho agent | FluteInTheRain |
| 2026-11-13 | Quy tắc ablation chốt **trước khi chạy** (commit `ea7068f`, `results/tables/README.md`): siêu tham số giống hệt mô hình chính, 1 bản xáo `shuffle_seed: 0`, chọn trên val, test đúng 1 lần mỗi seed, báo cáo dù kết quả ra sao | Claude Code đề xuất và thực hiện theo yêu cầu "làm chính xác" của FluteInTheRain |
| 2026-11-13 | Chọn 5 + 5 truy vấn trên **val** (không phải test), seed 0, `--num-edges 3`. Lý do: test chỉ chạy một lần trong `evaluate_graph_rerank.py`; chấm lại test để phân tích cần cả nhóm quyết định | Claude Code đề xuất; nhóm chưa duyệt |
| 2026-11-13 | Thêm quy tắc "câu trùng chỉ chọn một lần" sau lần chạy đầu (`fc26470`) | Claude Code; lý do ở mục 7 |
| 2026-11-13 | Chấp nhận dòng `clip_graph_vg_lr1e4_shuffled` trong `main.csv` được chép tay từ đầu ra `summarize_runs.py --output <file tạm>` | FluteInTheRain |
| 2026-11-13 | **Không** chạy thêm bản xáo seed 1, 2 (experiment-reviewer có đề xuất). Giữ đúng quy tắc đã chốt: 1 bản xáo | FluteInTheRain |
| 2026-11-13 | Push nhánh và mở PR; nội dung (bản nháp 5 + 5, câu chữ kết luận) nhóm duyệt sau | FluteInTheRain |
| 2026-11-13 | Cho một agent mới, không mang ngữ cảnh phiên này, review lại code và báo cáo của PR #8, rồi sửa theo phản hồi (mục 9, dòng cuối) | FluteInTheRain |

## 3. Môi trường và dữ liệu

| Mục | Giá trị |
|---|---|
| Máy | MacBook arm64, macOS 26.4.1, 8 luồng CPU, không GPU |
| Thư viện | Python 3.11.17, torch 2.15.0, torch_geometric 2.8.1, faiss 1.15.1, numpy 2.4.6 (`run_copies/*/env.json`) |
| Split | `splits/visual_genome/split_seed0.json`, sha256 `0c65b3e5fa9fedc3c225e77806dc9832d5e2135cadba973fd3506d80a002c4ee` |
| Đặc trưng CLIP | `artifacts/features/visual_genome/ViT-B-32__laion2b_s34b_b79k/` (có sẵn trên máy, ADR-0041) |
| `objects.json` v1.4 | tải 2026-11-13, sha256 `94238b6187c0ea1f97acc1fdaec106bd3dee12bb0f81b23dc8b07e9173514985` (khớp `docs/guides/xay-do-thi.md`) |
| `region_descriptions.json` v1.2 | tải 2026-11-13 (712 MB), sha256 `19624f10afb10bf69e30ecf48d85b1a71420c7432e8c88f1011a3c4fd60ec768`. Repo chưa ghi hash chuẩn để đối chiếu |
| Đồ thị gốc | `artifacts/graphs/visual_genome/contains_clip_top16_v1`, dựng tại commit `6a76986`, 1.729.232 cạnh, `graph.npz` sha256 `33fb4c6b95eaed9ad34ed56b423006b068840f969742d91d4b2fef7a5edc7c48` |
| Sửa môi trường | Trỏ `faiss/.dylibs/libomp.dylib` sang bản của torch trong `.venv` (mục 10) |

## 4. Tham số

### 4.1 Mô hình chính và ablation

`configs/clip_graph_vg_lr1e4.yaml` và `configs/clip_graph_vg_lr1e4_shuffled.yaml` chỉ khác `experiment`,
`graph_dir`, `model_dir`. Test `test_shuffled_ablation_config_differs_from_the_main_config_only_in_the_graph`
trong `tests/test_graph_rerank_pipeline.py` kiểm điều này.

| Tham số | Giá trị | Nguồn quyết định |
|---|---|---|
| Mô hình | GraphSAGE một lớp, khái niệm → ảnh (`ConceptToImageSage`, `src/mmxai/models/sage.py`) | ADR-0046 |
| Cạnh `contains` | CLIP top-16 cho mọi split | ADR-0046 |
| `top_k` | 100 ứng viên từ FAISS | ADR-0043 |
| `num_hard_negatives`, `temperature` | 15, 0,05 | ADR-0046 (Claude Code đề xuất, nhóm chưa duyệt) |
| `learning_rate` | 1e-4 | qphulong, 2026-11-12 |
| `batch_size`, `max_epochs`, `patience` | 512, 10, 2 | ADR-0047 |
| `alpha_grid` | 0,0; 0,1; …; 1,0 | ADR-0016 |
| Chọn epoch và `alpha` | MRR@100 trên val; hoà thì lấy `alpha` lớn hơn | ADR-0047 |
| Seed huấn luyện | 0, 1, 2 | ADR-0011 |
| `--threads` | 8 | số luồng của máy |

### 4.2 Đồ thị xáo

| Tham số | Giá trị |
|---|---|
| Config | `configs/graph_vg_clip_topk_shuffled.yaml` |
| Thuật toán | `shuffle_contains_edges` trong `src/mmxai/graph/shuffle.py`. Mỗi split: hoán vị đầu khái niệm của các cạnh, rồi đổi chỗ ngẫu nhiên để xoá cặp (ảnh, khái niệm) trùng. Giữ đúng bậc của mỗi ảnh và của mỗi khái niệm trong từng split; không cạnh nào vượt split |
| `shuffle_seed` | 0 (Claude Code đề xuất; nhóm quyết định không thêm seed khác) |
| Đầu ra | `artifacts/graphs/visual_genome/contains_clip_top16_shuffled_seed0`, `edge_source: clip_topk_shuffled` |
| Lưu ý khi đọc meta | `meta.json` của đồ thị xáo chép nguyên meta gốc, nên khoá `config` (có `output_dir: .../contains_clip_top16_v1`, `edge_source: clip_topk`) là config **của lần dựng đồ thị gốc**. Thông tin của lần xáo nằm ở khoá `shuffle` và `edge_source` ở cấp ngoài. Không sửa code vì đồ thị đã được dùng cho kết quả |
| ADR-0016 §6 | Mục này yêu cầu ablation và fidelity dùng chung hàm can thiệp (`mark_remaining_edges`). Xáo cạnh là viết lại đồ thị **trước khi huấn luyện**, không phải can thiệp lúc chấm điểm, nên không dùng được hàm đó. Phép "bỏ cạnh" lúc chấm điểm (kiểm tra xoá, "bỏ mọi cạnh") thì có dùng |

### 4.3 Giải thích

| Tham số | Giá trị |
|---|---|
| Lệnh | `uv run python scripts/explain.py --config configs/clip_graph_vg_lr1e4.yaml --seed 0 --split val --num-edges 3 --vg-data-dir data/visual_genome` |
| Explainer | Occlusion (`src/mmxai/explain/occlusion.py`): bỏ từng cạnh vào ảnh, chạy lại mô hình, đo độ giảm `cos(q, h_i)` |
| Mô hình được giải thích | Checkpoint seed 0 huấn luyện lại trên máy này (mục 5), `alpha = 0.0` đọc từ `selection.json` |
| `--num-edges` | 3: số quan hệ hiển thị mỗi phía, và số cạnh bỏ cùng lúc trong kiểm tra xoá. **Chưa chọn bằng fidelity trên val** (cấp độ 3) |
| Quy tắc chọn 5 + 5 | Docstring `src/mmxai/eval/query_ranks.py` (6 quy tắc), cũng ghi trong `analysis.json` → `settings.selection_rule` |

## 5. Kết quả: huấn luyện lại mô hình chính (chỉ val)

Mục đích: cần checkpoint để giải thích, vì checkpoint gốc nằm trên máy của đồng đội. **Không chạy lại test** cho
`clip_graph_vg_lr1e4` (dòng test trong `main.csv` là của lần chạy gốc, commit `6c97c4c`).

| Seed | Epoch tốt nhất | `alpha` | Val MRR@100 | Val R@1 / R@5 / R@10 |
|---|---|---|---|---|
| 0 | 6 | 0,0 | 0,1286 | 0,0742 / 0,1705 / 0,2372 |
| 1 | 8 | 0,0 | 0,1285 | 0,0749 / 0,1696 / 0,2382 |
| 2 | 6 | 0,1 | 0,1272 | 0,0699 / 0,1716 / 0,2412 |
| mean ± std | | | **0,1281 ± 0,0008** | |

Kết quả khớp **tới 4 chữ số** (mức chi tiết duy nhất có để so) với số đồng đội ghi trong `docs/guides/huan-luyen-va-xep-hang-lai.md` mục 7: 0,1281 ± 0,0008,
epoch 6/8/6, `alpha` 0/0/0,1. Kết luận: checkpoint trên máy này thay được checkpoint gốc khi giải thích. Chưa đối
chiếu `model_sha256` với run test gốc. Nguồn: `run_copies/20261113-09{49,50,51}_*`, `selection/clip_graph_vg_lr1e4_seed*.json`.

## 6. Kết quả: ablation đồ thị xáo cạnh

### 6.1 Theo từng run

| Run | Split | Seed | Epoch | `alpha` | R@1 | R@5 | R@10 | MRR@100 | `model_sha256` |
|---|---|---|---|---|---|---|---|---|---|
| `0955_..._shuffled_seed0` | val | 0 | 7 | 0,1 | 0,0677 | 0,1716 | 0,2391 | 0,1240 | |
| `0956_..._shuffled_seed1` | val | 1 | 9 | 0,2 | 0,0665 | 0,1724 | 0,2402 | 0,1240 | |
| `0957_..._shuffled_seed2` | val | 2 | 8 | 0,1 | 0,0670 | 0,1714 | 0,2395 | 0,1241 | |
| `0957_..._shuffled_seed0` | test | 0 | 7 | 0,1 | 0,0684 | 0,1755 | 0,2485 | 0,1268 | `7811ab7c0b77…` |
| `0957_..._shuffled_seed1` | test | 1 | 9 | 0,2 | 0,0686 | 0,1760 | 0,2489 | 0,1274 | `4574b7a02771…` |
| `0958_..._shuffled_seed2` | test | 2 | 8 | 0,1 | 0,0688 | 0,1763 | 0,2493 | 0,1274 | `0b6dbec5fdbc…` |

Tổng hợp test (`results/tables/main.csv`, dòng `clip_graph_vg_lr1e4_shuffled`): R@1 0,0686 ± 0,0002, R@5 0,1759 ± 0,0004,
R@10 0,2489 ± 0,0004, MRR@100 **0,1272 ± 0,0003**. Script tổng hợp chạy lại trên máy này ra dòng `clip_vg` giống
hệt dòng đã commit, đây là một phép kiểm chéo.

### 6.2 MRR@100 val theo `alpha` (từ `selection/*.json`)

| `alpha` | Thật s0 | Thật s1 | Thật s2 | Xáo s0 | Xáo s1 | Xáo s2 |
|---|---|---|---|---|---|---|
| 0,0 | **0,1286** | **0,1285** | 0,1258 | 0,1238 | 0,1240 | 0,1237 |
| 0,1 | 0,1266 | 0,1266 | **0,1272** | **0,1240** | 0,1240 | **0,1241** |
| 0,2 | 0,1267 | 0,1264 | 0,1258 | 0,1240 | **0,1240** | 0,1236 |
| 0,5 | 0,1228 | 0,1236 | 0,1233 | 0,1219 | 0,1229 | 0,1223 |
| 1,0 (CLIP thuần) | 0,1061 | 0,1061 | 0,1061 | 0,1061 | 0,1061 | 0,1061 |

In đậm là `alpha` được chọn. Ở bản xáo, các `alpha` 0,0–0,2 gần như bằng nhau ở 4 chữ số; code so sánh số đầy đủ.

### 6.3 So sánh theo cặp seed (test, MRR@100)

Số của mô hình chính và đối chứng lấy từ `docs/guides/huan-luyen-va-xep-hang-lai.md` mục 7, chỉ có 4 chữ số. Số của
bản xáo lấy đầy đủ từ `run_copies/*_shuffled_seed*/metrics.json` (test). Hiệu, mean, std (ddof = 1) và t đều tính
trên các số đó **trước khi làm tròn**, nên trung bình của cột hiệu đã làm tròn có thể lệch 1 ở chữ số cuối.

| Seed | Thật | Xáo | Không đồ thị (`lr` 1e-3) | Thật − xáo | Xáo − không đồ thị |
|---|---|---|---|---|---|
| 0 | 0,1279 | 0,12681 | 0,1259 | +0,00109 | +0,00091 |
| 1 | 0,1283 | 0,12735 | 0,1269 | +0,00095 | +0,00045 |
| 2 | 0,1291 | 0,12736 | 0,1271 | +0,00174 | +0,00026 |
| mean ± std | 0,1284 ± 0,0006 | 0,12718 ± 0,00031 | 0,1266 ± 0,0006 | **+0,00126 ± 0,00042** | +0,00054 ± 0,00034 |

t ghép cặp (thật − xáo) = 5,16 với df = 2 (p ≈ 0,04); xáo − không đồ thị: t = 2,80 (p ≈ 0,11). Đây là **số mô tả**:
chỉ đo dao động theo seed huấn luyện trên một tập test cố định và **một** bản xáo, không gồm dao động do cách xáo hay
do mẫu truy vấn. Kiểm định theo truy vấn (paired bootstrap) chưa làm (mục 11).

### 6.4 Tỉ lệ cạnh xáo trùng cạnh gốc

Đo sau khi chạy, để tìm hiểu bất thường val/test. Đây không phải lựa chọn nào cả.

| Split | Số cạnh | Trùng cạnh gốc | Tỉ lệ |
|---|---|---|---|
| train | 1.569.232 | 29.428 | 1,88% |
| val | 80.000 | 1.421 | 1,78% |
| test | 80.000 | 1.510 | 1,89% |

## 7. Kết quả: giải thích và 5 + 5 truy vấn

| Mục | Giá trị |
|---|---|
| Kiểm tra khớp pipeline | MRR@100 val trong phân tích = 0,1285508567…, bằng **từng bit** `alpha_sweep["0.0"]` trong `selection.json` seed 0. Từ `9689c79`, `analyse_split` tự kiểm điều này và dừng nếu khác (`check_reproduces_training` trong `src/mmxai/explain/query_analysis.py`) |
| Thống kê val (25.000 truy vấn) | Hạng tăng 7.387, giảm 4.620, giữ nguyên 1.463; với 11.530 truy vấn, ảnh đúng ngoài top 100 của CLIP |
| CLIP thuần trên val | R@1 0,0559, R@5 0,1443, R@10 0,2046, MRR@100 0,1061 |
| CLIP + Graph trên val (seed 0) | R@1 0,0742, R@5 0,1705, R@10 0,2372, MRR@100 0,1286 |
| Khớp văn bản ↔ ảnh | Ở cả 10 truy vấn, ảnh nguồn của câu nằm trong danh sách ảnh đúng |
| Kết quả chi tiết | `run_copies/20261113-1036_clip_graph_vg_lr1e4_explain_seed0/analysis.{json,md}` (bản cuối); bản nháp phân tích: `reports/case_studies/5-thanh-cong-5-that-bai.md` |

**Vì sao chạy hai lần:** lần `1002` chọn câu "part of a trouser" 3 lần và "tall green and tan grass" 2 lần. Visual
Genome lặp region description, các câu trùng có cùng thứ hạng, nên chỉ còn 3 thành công và 4 thất bại khác nhau.
Sửa ở `fc26470`: câu đã chọn (không phân biệt hoa/thường, khoảng trắng) thì bỏ qua. Việc sửa chỉ loại câu trùng,
không đổi hướng chọn, và chỉ chạy trên val. Lần `1004` cho 5 + 5 câu khác nhau.

Hai phép can thiệp trên ảnh đúng, mỗi lần chấm lại mọi ứng viên của truy vấn với `alpha` đã lưu:
"xoá 3" bỏ 3 cạnh đẩy lên mạnh nhất; "bỏ mọi cạnh" bỏ cả 16 cạnh vào ảnh, chỉ ảnh đó (các ảnh khác giữ cạnh).

| Nhóm | Truy vấn | Hạng CLIP → CLIP + Graph | Xoá 3: điểm đồ thị; hạng | Bỏ mọi cạnh: điểm đồ thị; hạng |
|---|---|---|---|---|
| Thành công | "man in a grey shirt and blue jeans" | 99 → 1 | 0,2046 → 0,1962; 1 → 1 | 0,2046 → 0,1679; 1 → 7 |
| Thành công | "Blue jacket with orange stripes" | 100 → 3 | 0,1771 → 0,1717; 3 → 5 | 0,1771 → 0,1658; 3 → 5 |
| Thành công | "part of a trouser" | 100 → 3 | 0,0727 → 0,0693; 3 → 4 | 0,0727 → 0,0987; 3 → 1 |
| Thành công | "a girl wearing blue jeans." | 99 → 3 | 0,1428 → 0,1357; 3 → 3 | 0,1428 → 0,1369; 3 → 3 |
| Thành công | "the grass beside the car" | 98 → 3 | 0,1761 → 0,1722; 3 → 3 | 0,1761 → 0,1603; 3 → 8 |
| Thất bại | "scattered waves in water" | 7 → 100 | 0,0930 → 0,0868; 100 → 100 | 0,0930 → 0,1043; 100 → 96 |
| Thất bại | "tall green and tan grass" | 15 → 96 | 0,0306 → 0,0257; 96 → 96 | 0,0306 → 0,0518; 96 → 94 |
| Thất bại | "wheel of car" | 15 → 96 | 0,0020 → −0,0041; 96 → 97 | 0,0020 → 0,0331; 96 → 82 |
| Thất bại | "a wooden sign" | 12 → 92 | 0,0014 → −0,0081; 92 → 95 | 0,0014 → 0,0721; 92 → 34 |
| Thất bại | "handicap parking sign on metal post" | 15 → 95 | 0,0325 → 0,0223; 95 → 98 | 0,0325 → 0,0724; 95 → 75 |

Với 5 ảnh hạng 1 của các truy vấn thất bại, bỏ mọi cạnh thì cả 5 vẫn đứng hạng 1.

**Chẩn đoán toàn cục** (`diagnostics/global_noedges_output.txt`, val): bỏ mọi cạnh của đồ thị val, giữ nguyên trọng số.
- Mô hình chính: MRR@100 0,12855 / 0,12846 / 0,12718 → 0,12730 / 0,12738 / 0,12395, tức trung bình 0,12806 → 0,12621.
- Mô hình đồ thị xáo: 0,12401 / 0,12403 / 0,12408 → 0,12252 / 0,12381 / 0,12284.
- CLIP thuần: 0,1061.

Ở cả 15 ảnh được giải thích (10 ảnh đúng và 5 ảnh hạng 1 của truy vấn thất bại), bỏ 3 cạnh đẩy lên đều làm điểm đồ
thị **giảm**. Không có ca nào đi sai hướng.

## 8. Kết luận và suy luận

### 8.1 Ablation

- **Kết luận:** đồ thị thật hơn đồ thị xáo ở cả 3 seed, +0,00126 ± 0,00042 MRR@100 test.
  - So với đối chứng không đồ thị **`lr = 1e-3`** (cấu hình chính của đối chứng), xáo cạnh làm mất khoảng 70% phần
    chênh +0,0018. Đối chứng khác `lr` với hai mô hình kia, nên con số 70% chỉ là ước lượng thô.
  - Trên val, so với đối chứng **cùng** `lr = 1e-4` (0,1246), đồ thị xáo (0,1240) còn thấp hơn, tức mất hơn toàn bộ
    phần chênh.
- Kết quả **phù hợp** với giả thuyết rằng một phần nhỏ cải thiện đến từ quan hệ ảnh – khái niệm thật. Không viết
  "chứng minh": chỉ có 3 seed, 1 bản xáo, hiệu ứng tuyệt đối khoảng 1% MRR; t và p ở mục 6.3 chỉ là số mô tả.
- Theo từng chỉ số (test): R@1 thật − xáo = 0,0705 − 0,0686 = +0,0019, lớn hơn std của mô hình chính (0,0014) nhưng
  vẫn sát mức nhiễu. R@5 (+0,0006) và R@10 (+0,0003) chênh nhỏ hơn std, không kết luận được.
- **Không** viết "đồ thị xáo tốt hơn không đồ thị": chênh +0,00054, p ≈ 0,11, và hai mô hình dùng `lr` khác nhau.
- **Suy luận phụ:** với đồ thị xáo, `alpha` chọn được cao hơn (0,1/0,2/0,1 so với 0/0/0,1) và epoch tốt nhất muộn hơn
  (7/9/8 so với 6/8/6). Cả hai khớp với ý "cạnh ngẫu nhiên mang ít thông tin hơn, nên mô hình dựa vào CLIP nhiều hơn".
- **Bất thường chưa giải thích được:**
  - Trên val, đồ thị xáo mất toàn bộ phần chênh (thật − xáo = +0,0040). Trên test chỉ mất +0,0013.
  - Bản xáo tăng từ val lên test +0,0031 (0,12404 → 0,12718, số đầy đủ), trong khi mô hình chính chỉ tăng +0,0003 và
    đối chứng +0,0004.
  - Tỉ lệ cạnh trùng gốc gần như bằng nhau ở val và test (1,78% và 1,89%, mục 6.4), nên điều này **không** giải thích
    được bất thường.
  - Với 1 bản xáo, không tách được dao động do cách xáo. Nhóm đã quyết định không thêm bản xáo, nên ghi đây là giới
    hạn trong báo cáo.

### 8.2 Giải thích và 5 + 5

- Giải thích **trung thực về nguồn gốc**: tính trên đúng mô hình, đúng cạnh của split val, đúng `alpha` đã dùng để
  xếp hạng. Có hai chốt chặn:
  - `check_reproduces_training` (`src/mmxai/explain/query_analysis.py`): MRR val tính lại phải bằng đúng số
    `selection.json` lưu lúc huấn luyện. Chốt này nối lần giải thích với lần xếp hạng đã báo cáo (sai checkpoint,
    sai đồ thị hay sai `alpha` đều bị bắt).
  - `explain_candidate` (`src/mmxai/explain/ranking_model.py`): điểm Occlusion dùng (đủ cạnh) phải bằng điểm xếp
    hạng của chính lần chạy đó. Chốt này chỉ so hai nhánh code trong cùng một lần chạy.
  - Không có heuristic chọn đường đi (review faithfulness, mục 9).
- **Thành công chủ yếu do vector của chính ảnh (phép chiếu học được `W_self x_i + b`), không do cạnh.**
  - Bằng chứng đúng là phép "bỏ mọi cạnh", không phải đóng góp từng cạnh. Với phép trung bình trên 16 cạnh, bỏ 1
    hay 3 cạnh thì điểm gần như chắc chắn chỉ đổi rất ít, dù nhánh láng giềng mạnh đến đâu (review PR #8, B2).
  - Khi bỏ mọi cạnh, 5 ảnh thành công vẫn đứng hạng 7 / 5 / 1 / 3 / 8, so với 99 / 100 / 100 / 99 / 98 theo CLIP.
  - Cạnh giúp 3/5 ca (hạng 7 → 1, 5 → 3, 8 → 3), không đổi 1 ca (3 → 3), và **kéo xuống** 1 ca ("part of a trouser":
    không cạnh thì hạng 1, có cạnh thì hạng 3).
  - Ở mức toàn cục, cùng trọng số mà bỏ mọi cạnh thì val MRR giảm từ 0,12806 xuống 0,12621. Vậy mô hình có dùng cạnh,
    nhưng phần lớn mức tăng so với CLIP (0,1061) vẫn còn khi không có cạnh.
  - Lưu ý: mô hình huấn luyện trên **đồ thị xáo** cũng mất khoảng 0,001 khi bỏ cạnh (0,12404 → 0,12306). Vì vậy riêng
    việc "bỏ cạnh thì điểm giảm" không chứng minh vai trò của quan hệ; phép so có ý nghĩa vẫn là đồ thị thật so với
    đồ thị xáo (mục 8.1).
- **Thất bại chủ yếu do `alpha = 0` và vector của ảnh; cạnh làm nặng thêm ở một số ca.**
  - `alpha = 0` nghĩa là bỏ hạng tử cosine của CLIP trong điểm cuối. Vector CLIP của ảnh vẫn nằm trong `h_i` qua
    `W_self`, nên **không** nên viết "điểm CLIP bị bỏ hoàn toàn".
  - Ảnh đúng vốn hạng 7–15 theo CLIP, có cạnh thì rơi xuống 92–100, bỏ mọi cạnh vẫn chỉ ở 96 / 94 / 82 / 34 / 75.
    Nghĩa là ngay cả khi không có cạnh, vector của ảnh qua phép chiếu đã xếp nó xa hơn nhiều so với CLIP.
  - Cạnh kéo xuống thêm rõ nhất ở "a wooden sign" (34 → 92), "handicap parking sign on metal post" (75 → 95) và
    "wheel of car" (82 → 96). Ở 2 ca còn lại cạnh gần như không đổi gì (96 → 100, 94 → 96).
  - Ảnh đứng hạng 1 của 5 truy vấn thất bại vẫn hạng 1 khi bỏ mọi cạnh, tức chúng thắng nhờ vector của chính ảnh.
    Giải thích bằng cạnh vì vậy chỉ giải thích được một phần các ca thất bại; báo cáo cần nói rõ điều này.
  - Hướng cải tiến (cấp độ 4): kết hợp điểm thích nghi theo truy vấn, thay vì một `alpha` cố định.
- **Cạnh đúng nghĩa có thể kéo ảnh xuống:**
  - Ví dụ: với "tall green and tan grass", cạnh `grass.n.01` → ảnh đúng có đóng góp −0,0031.
  - Lý do: mô hình lấy trung bình 16 khái niệm láng giềng rồi tính cosine, nên đóng góp không cộng dồn và không
    luôn cùng chiều với nghĩa.
- **Cạnh CLIP top-16 có nhiễu:**
  - Ví dụ: `screen_saver.n.01`, `body_waste.n.01`, `adam.n.01`, `amour_propre.n.01`.
  - Giải thích hiển thị đúng đồ thị mà mô hình đã dùng, kể cả phần nhiễu này. Đây là cái giá của việc không dùng
    annotation cho cạnh (ADR-0046, mục A2).

## 9. Review độc lập và cách xử lý

Mỗi reviewer là một subagent chỉ đọc, chạy độc lập và nhận diff `4cdfe19...feat/level1-ablation-explain`.

| Reviewer | Kết luận | Phát hiện chính | Xử lý |
|---|---|---|---|
| `leakage-auditor` | Không rò rỉ | (a) 3 split dùng chung một RNG, xáo theo thứ tự test → train → val, nên cách xáo của train phụ thuộc vào *số cạnh* của test, nhưng không mang nhãn/nội dung test. (b) Dòng `main.csv` chép tay | (a) Không sửa vì đồ thị đã chốt trước khi chạy; ghi giới hạn trong guide. (b) FluteInTheRain chấp nhận; ghi trong `results/README.md` |
| `experiment-reviewer` | Quy trình đúng và công bằng; kết quả ủng hộ giả thuyết, hiệu ứng nhỏ | Các số ở mục 6.3, 8.1. Đề xuất: thêm bản xáo seed 1, 2; kiểm định theo truy vấn (bootstrap); đối chiếu `model_sha256`; bảng per-seed | Số và cách diễn đạt đã đưa vào guide. **Không** thêm bản xáo (FluteInTheRain quyết định). Bootstrap và đối chiếu sha: chưa làm (mục 11). Bảng per-seed: ở mục 6 của file này |
| `explanation-faithfulness-reviewer` | "Chưa chứng minh được", **không phải** hợp lý hoá sau: nguồn gốc trung thực, thiếu bằng chứng fidelity | Cao: chưa có đối chứng xoá ngẫu nhiên; chưa có số fidelity tổng hợp (≥ 10 truy vấn × 3 seed). Trung bình: `--num-edges 3` chưa chọn trên val; chưa gắn cờ khi xoá nhiều cạnh làm điểm đổi sai hướng. Thấp: câu mô tả dấu đóng góp bị ngược; chưa có can thiệp riêng cho ca thất bại; test end-to-end chỉ dùng `alpha = 0` | Sửa câu mô tả dấu (`fc26470`). Kiểm trên dữ liệu thật: 15/15 ca xoá đều làm điểm giảm (mục 7). Các mục fidelity còn lại thuộc cấp độ 3: chưa làm (mục 11) |
| Review độc lập lần 2 (agent mới, không mang ngữ cảnh, sau khi mở PR #8) | Không có lỗi critical, không rò rỉ; code xáo cạnh và giải thích đúng. Vấn đề lớn nhất ở hồ sơ | **Major:** thư mục `runs/` của hồ sơ không được commit vì `.gitignore` có `runs/`. **Minor:** chốt chặn chỉ so hai nhánh code của cùng lần chạy; docstring nói quy tắc 6 "cố định trước"; dẫn sai bằng chứng cho "thành công do phép chiếu" (đóng góp từng cạnh không trả lời được câu hỏi); số làm tròn lệch nhau; "70%" và p dễ đọc quá mức. **Nit:** tên test hứa "tests once"; khoá `config` trong meta đồ thị xáo; ADR-0016 §6; "khớp hoàn toàn"; mô tả nội dung ảnh chưa xem; ghi chú macOS thiếu cách hoàn tác; thiếu R@1 | Tất cả đã xử lý: đổi `runs/` → `run_copies/`; thêm `check_reproduces_training` và phép "bỏ mọi cạnh" + test (`9689c79`); viết lại mục 6.3, 8.1, 8.2 bằng số đầy đủ; ghi chú meta và ADR-0016 ở mục 4.2; sửa guide và case study. Số "bỏ mọi cạnh" của reviewer (bỏ cạnh của **mọi** ảnh) khác số trong code (bỏ cạnh của **riêng** ảnh đó); cả hai đều được ghi (mục 7) |
| `pr-test-analyzer` | Coverage tốt, không có lỗ hổng mức nghiêm trọng (≥ 8/10) | (7/10) chưa test khớp văn bản ↔ ảnh; (6/10) `alpha` chỉ test ở 0; (6/10) kiểm tra xoá chưa test với ≥ 2 cạnh hỗ trợ; vài nhánh lỗi chưa test | Thêm assert khớp văn bản ↔ ảnh (`fc26470`) và kiểm trên dữ liệu thật (10/10 khớp). Hai mục 6/10: chưa làm (mục 11) |

## 10. Sửa lỗi môi trường và test

1. **`Fatal Python error: Aborted` trong FAISS (chỉ macOS).**
   - Tái hiện bằng script nhỏ: nạp `torch` và `faiss` cùng lúc thì ra `OMP: Error #15 ... libomp.dylib already initialized`.
   - Nguyên nhân: hai wheel mỗi bên mang một bản `libomp.dylib` (`faiss/.dylibs/` và `torch/lib/`).
   - Đã thử `KMP_DUPLICATE_LIB_OK=TRUE` (bị treo hơn 10 phút) và `OMP_NUM_THREADS=1` (vẫn crash).
   - Sửa: trỏ bản của faiss sang bản của torch (symlink trong `.venv`; bản gốc sao lưu ngoài repo). Sau đó mọi thứ tự
     import đều chạy.
   - Phải làm lại sau mỗi lần `uv sync` cài lại faiss. Ghi trong `docs/guides/huan-luyen-va-xep-hang-lai.md` mục 4.
2. **8 test của `tests/test_feature_pipeline.py` fail khi có torch.** Lỗi có sẵn trên `main`.
   - Nguyên nhân: `tests/test_clip_encoder.py` import module thật, gắn thuộc tính `clip_encoder` vào package
     `mmxai.features`. `from mmxai.features import clip_encoder` đọc thuộc tính này trước `sys.modules`, nên fixture
     giả bị bỏ qua.
   - Sửa fixture: thay cả thuộc tính package (`f850268`).
   - Chạy riêng file thì 18/18 pass; chạy sau `test_clip_encoder.py` thì trước khi sửa 8 fail, sau khi sửa pass.
3. **3 lỗi ruff I001** đã có sẵn trên `main`: thiếu dòng trống giữa nhóm import (`6a76986`).

Kết quả cuối trên nhánh: `uv run pytest` → 424 passed, 1 skipped. Test bị skip có từ trước: `queries/` còn rỗng.
`uv run ruff check` → All checks passed.

## 11. Việc chưa làm

| Việc | Vì sao chưa | Thuộc |
|---|---|---|
| Thêm bản xáo seed 1, 2 | FluteInTheRain quyết định không làm (giữ quy tắc đã chốt) | – |
| Fidelity so với xoá ngẫu nhiên; validity, sparsity tổng hợp trên ≥ 10 truy vấn × 3 seed; chọn `--num-edges` trên val | Ngoài phạm vi cấp độ 1 | Cấp độ 3 |
| Kiểm định theo truy vấn (paired bootstrap) giữa đồ thị thật và đồ thị xáo trên test | Chưa được yêu cầu; nếu làm thì không được chọn gì dựa trên kết quả | Tuỳ chọn |
| Đối chiếu `model_sha256` của run test gốc (máy đồng đội) với checkpoint huấn luyện lại | Không có run test gốc trên máy này | Truy vết |
| Test với `alpha` khác 0 và test xoá ≥ 2 cạnh hỗ trợ (pr-test-analyzer, 6/10) | Ưu tiên thấp hơn; chưa làm | Chất lượng test |
| Nhóm duyệt bản nháp 5 + 5 và câu chữ kết luận; tự kiểm tra code rồi sửa cột cuối của `docs/ai-usage-log.md` | FluteInTheRain: duyệt sau | – |

## 12. Dòng thời gian commit

| Commit | Nội dung |
|---|---|
| `f850268` | Sửa fixture `clip_encoder` |
| `6a76986` | Sửa import (ruff I001). Đồ thị gốc và mô hình chính huấn luyện lại ở commit này |
| `f7c2f88` … `dc9d469` | Agent A: hàm xáo, CLI, config ablation, README |
| `ea7068f` | **Chốt quy tắc ablation** (trước khi xáo và huấn luyện). Mọi run ablation ghi commit này |
| `44de3ec` | Dòng test ablation trong `main.csv` |
| `adcb57b` … `a2ed697`, merge `2264087` | Agent B: thứ hạng theo truy vấn, giải thích, `scripts/explain.py` |
| `fc26470` | Sửa chọn câu trùng, câu mô tả dấu, assert khớp văn bản ↔ ảnh. Lần chạy giải thích `1004` ở commit này |
| `bb092ff` | Tài liệu: guide, bản nháp 5 + 5, checklist README, nhật ký AI |
| `4c77b2a` | Hồ sơ phiên này (bản đầu) |
| `9689c79` | Sau review lần 2: chốt chặn khớp `selection.json`, phép "bỏ mọi cạnh", docstring quy tắc 6, test "test chỉ một lần". Lần chạy giải thích `1036` ở commit này |
| (commit tài liệu kế tiếp) | Sau review lần 2: sửa hồ sơ, guide, case study; thêm `run_copies/`, `diagnostics/` |
