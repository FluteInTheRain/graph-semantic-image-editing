# Thiết kế: mở rộng cấp độ 1 (nDCG, chiều ảnh → văn bản, đánh giá theo nhóm truy vấn)

Ngày: 2026-11-13. Người yêu cầu và quyết định: FluteInTheRain. Trạng thái: đã duyệt (FluteInTheRain, 2026-11-13).
Đường dẫn tính từ gốc repo. Số dòng có thể lệch khi code đổi; tìm theo tên hàm.

## 1. Mục đích

PDF §7, bảng "Đánh giá kết quả", hàng **Nhiệm vụ 1 – Truy vấn**, cột **Mở rộng** có ba mục. Nhóm làm cả ba để báo
cáo cấp độ 1 đầy đủ hơn:

1. **nDCG**: thêm một chỉ số xếp hạng bên cạnh Recall@1/5/10 và MRR@100.
2. **Chiều truy vấn thứ hai (ảnh → văn bản)**: đưa ảnh vào, tìm câu mô tả đúng.
3. **Đánh giá theo nhóm truy vấn**: tách kết quả văn bản → ảnh theo hai cách chia nhóm.

Mọi mục đều làm cho **CLIP thuần** và **CLIP + Graph**, vì PDF §7 bắt buộc so sánh hai mô hình này.

## 2. Mục tiêu, không làm, không được xảy ra

**Mục tiêu**
- Có số trên **test** (mean ± std, 3 seed) cho cả ba mục, với 5 experiment: CLIP thuần, đối chứng không đồ thị,
  CLIP + Graph một lớp, CLIP + Graph đồng xuất hiện, đồ thị xáo.
- Mọi số truy vết được: config, commit, file run được chép vào `results/records/`.

**Không làm**
- Các mục mở rộng của hàng Nhiệm vụ 2 (độ nhạy α, so sánh cách tạo cạnh). Đề ghi hàng này là "cấp độ 2 trở lên".
- Đánh giá theo nhóm cho chiều ảnh → văn bản.
- Huấn luyện riêng cho chiều ảnh → văn bản; đổi siêu tham số; xoá hay sửa dòng cũ trong `results/tables/main.csv`.
- Khảo sát người dùng, video, cách chia nhóm thứ ba (thể loại đối tượng qua WordNet).

**Không được xảy ra** (nếu xảy ra thì kết quả hỏng dù code chạy):
- Bất kỳ lựa chọn nào (α, ngưỡng, cách chia nhóm, luật nhận vị từ) làm sau khi đã thấy số **test** của lần chạy mới.
- Hai đường chấm test khác nhau cho cùng một mô hình (một đường cũ, một đường mới) cho ra số không khớp.
- nDCG và R@K của cùng một run được tính từ hai thứ hạng khác nhau.

## 3. Ràng buộc

- PDF §6, `CLAUDE.md`, ADR-0009: chọn mọi thứ trên val. Test chạy một lần cho mỗi experiment **mới**.
- **Chạy lại test cho các mô hình đã có dòng test** là một ngoại lệ của quy tắc nhóm, nên phải thoả 5 điều kiện
  (FluteInTheRain đồng ý 2026-11-13):
  1. Giữ nguyên config: split, siêu tham số, seed 0/1/2.
  2. Commit quy tắc chạy và báo cáo **trước khi chạy**. Báo cáo mọi số mới, dù ra sao.
  3. Không xoá dòng cũ. Dòng mới mang tên experiment mới (hậu tố `_eval2`).
  4. Một ADR mới (ADR-0049, trạng thái "Đề xuất" tới khi nhóm xác nhận) ghi giao thức đánh giá v2 và ngoại lệ này.
  5. Chạy lại **toàn bộ** (cả huấn luyện) trên cùng một máy và cùng một commit; chép file run vào `results/records/`.
- Code đơn giản, đọc được khi vấn đáp; logic trong `src/mmxai/`, CLI mỏng trong `scripts/`, test bằng dữ liệu tổng
  hợp nhỏ (`.claude/rules/tests.md`). Không thêm dependency.
- Tài liệu tiếng Việt, code và docstring tiếng Anh.

## 4. Cách làm đã chọn và các cách đã cân nhắc

| Cách | Nội dung | Kết quả |
|---|---|---|
| **A. Mở rộng pipeline hiện có** | Đường chấm hiện tại (`evaluate_clip`, `train_and_select`, `evaluate_test`) tính thêm nDCG, số theo nhóm và chiều ảnh → văn bản, ghi vào `metrics.json`; `summarize_runs` gom thêm bảng | **Đã chọn** (FluteInTheRain). Lý do: chỉ có một đường chấm test, dùng lại cơ chế chặn chạy test lần hai |
| B. Script đánh giá riêng sau huấn luyện | Nạp checkpoint, tính mọi thứ ở một chỗ mới | Bỏ: thành hai đường chấm test song song, dễ lệch nhau |
| C. Lưu xếp hạng thô rồi tính offline | Mỗi run lưu 100 ứng viên × mọi truy vấn | Bỏ: file lớn, không commit được; lợi ích chưa cần |

Cũng đã cân nhắc **không chạy lại test** (nDCG và số theo nhóm của chiều văn bản → ảnh chỉ báo trên val). FluteInTheRain
chọn chạy lại để mọi mục có số test và hai chiều dùng cùng một mô hình.

## 5. Thiết kế

### 5.1 nDCG@10 (cả hai chiều)

- Relevance nhị phân. Một kết quả là đúng theo đúng định nghĩa đáp án của ADR-0043 (chiều văn bản → ảnh) hoặc của mục
  5.2 (chiều ảnh → văn bản).
- `DCG@10 = Σ_{vị trí p ≤ 10, kết quả đúng} 1 / log2(p + 1)`;
  `IDCG@10 = Σ_{j=1}^{min(số đáp án đúng, 10)} 1 / log2(j + 1)`. nDCG@10 = DCG / IDCG. Đáp án đúng nằm ngoài top 100
  vẫn được tính vào IDCG.
- Tính từ **cùng** danh sách đã xếp hạng (sau khi kết hợp điểm với α) mà R@K và MRR dùng. Vì vậy hàm metric mới phải
  nhận danh sách id đã xếp và tập đáp án, không chỉ hạng đầu tiên như `compute_metrics` hiện nay
  (`src/mmxai/eval/metrics.py`).
- Phải có test tính tay: một đáp án ở vị trí 1 thì nDCG = 1; hai đáp án ở vị trí 2 và 5; không có đáp án trong top 10
  thì nDCG = 0.

### 5.2 Chiều ảnh → văn bản

Một mô hình duy nhất chấm điểm một **cặp (câu t, ảnh i)**:
`điểm(t, i) = α · cos(t, x_i) + (1 − α) · cos(t, h_i)`. `x_i` là vector CLIP của ảnh; `h_i` là vector đồ thị do
mô hình tạo ra (`image_vectors` trong `src/mmxai/eval/graph_rerank.py`). Chiều văn bản → ảnh cố định t và xếp hạng ảnh.
Chiều ảnh → văn bản cố định i và xếp hạng câu. **Không huấn luyện thêm.**

| Thành phần | Quy định |
|---|---|
| Truy vấn | Mọi ảnh của tập đánh giá (test: 5.000 ảnh) |
| Kho tìm kiếm | Mọi câu (5 region description đã lấy mẫu mỗi ảnh, ADR-0041) của ảnh thuộc tập đánh giá (test: 25.000 câu) |
| Đáp án đúng | Câu t là đúng với ảnh i nếu i nằm trong tập ảnh đúng của t theo ADR-0043 (cosine văn bản ≥ `same_text_threshold`), tức 5 câu của chính ảnh, cộng mọi câu giống hệt một trong 5 câu đó. Dùng lại `same_text_relevant_images` rồi đảo chiều ánh xạ |
| Ứng viên | FAISS `IndexFlatIP` trên vector câu, top 100 câu theo `cos(x_i, t)`. CLIP thuần dừng ở đây (α = 1) |
| Xếp hạng lại (CLIP + Graph) | Với mỗi ảnh truy vấn i và 100 câu ứng viên: `α_i2t · cos(t, x_i) + (1 − α_i2t) · cos(t, h_i)`; sắp xếp ổn định, hoà giữ thứ tự CLIP (như `rerank_ranks`) |
| `α_i2t` | Chọn **trên val**, sau khi huấn luyện xong, cùng lưới `alpha_grid` và cùng luật: MRR@100 cao nhất, hoà thì α lớn hơn (ADR-0047). Lưu vào `selection.json` (khoá mới, ví dụ `alpha_image_to_text` và `alpha_sweep_image_to_text`). Epoch vẫn chọn theo chiều văn bản → ảnh như cũ |
| Metric | R@1/5/10 và MRR@100 theo câu đúng đầu tiên, nDCG@10, số truy vấn |

### 5.3 Nhóm truy vấn (chỉ chiều văn bản → ảnh)

Hai cách chia **độc lập**, mỗi truy vấn thuộc đúng một nhóm trong mỗi cách. Báo cáo cho mỗi nhóm: số truy vấn, tỉ lệ
truy vấn, R@1/5/10, MRR@100, nDCG@10.

**Cách 1: câu riêng hay câu chung**
- `riêng`: truy vấn có đúng 1 ảnh đúng (chỉ ảnh gốc).
- `chung`: truy vấn có ≥ 2 ảnh đúng, tức câu giống hệt câu của ảnh khác, ví dụ "a man".
- Dữ liệu có sẵn trong `Candidates.relevant`; không cần văn bản gốc.

**Cách 2: câu có quan hệ hay không**
- Cần văn bản gốc của câu: đọc từ `region_descriptions.json` qua `load_region_texts` (`mmxai.data.visual_genome`), như
  `scripts/explain.py` đã làm. Thêm một tham số đường dẫn dữ liệu vào config.
- Chuẩn hoá: chữ thường, chỉ giữ chữ cái, tách theo khoảng trắng. Một vị từ khớp khi dãy từ của nó xuất hiện liền
  nhau trong câu.
- Danh sách vị từ lấy từ `configs/query_benchmark_vg.yaml`: `predicate_templates` (khoá), `predicate_aliases` (khoá)
  và `action_predicates`. Danh sách này đã được chốt trên train cho ADR-0044. Đọc từ file đó, không chép tay.
- Ba nhóm, xét theo thứ tự:
  - `hành động`: câu chứa ít nhất một vị từ trong `action_predicates`, hoặc alias của nó (ví dụ "man **holding**
    umbrella");
  - `vị trí`: không thuộc nhóm trên, nhưng chứa một vị từ khác trong danh sách (ví dụ "sign **on** pole");
  - `không quan hệ`: còn lại (ví dụ "red car").
- **Kiểm tra trên val trước khi chốt:** in kích thước ba nhóm và 30 câu ngẫu nhiên (seed 0) của mỗi nhóm trên val; người
  làm đọc để chắc luật không sai rõ ràng. Mọi chỉnh sửa luật chỉ dựa trên val, phải xong và được commit trước bước
  chốt quy tắc (mục 6, bước 3). Kết quả kiểm tra này được ghi vào hồ sơ.
- **Nếu luật sai thì sửa ở đâu** (quyết định của Claude Code, 2026-11-13; FluteInTheRain duyệt cùng bản thiết kế):
  - **Không bao giờ** sửa `configs/query_benchmark_vg.yaml`. Đó là config đã đóng băng của bộ truy vấn ADR-0044.
  - Được sửa cách chuẩn hoá và cách khớp trong code.
  - Được thêm vào config của phần chia nhóm hai danh sách nhỏ: `extra_predicates` (thêm vào nhóm `hành động` hoặc
    `vị trí`, ví dụ một dạng chia động từ mà alias chưa có) và `excluded_predicates` (không tính là quan hệ). Ban đầu
    cả hai danh sách đều rỗng. Mỗi mục thêm vào phải kèm lý do và ví dụ câu val, và được ghi vào ADR-0049.
  - Trường hợp "man in a grey shirt" ("in" mang nghĩa mặc): giữ ở nhóm `vị trí` trừ khi kiểm tra val cho thấy trường hợp
    này làm nhóm mất ý nghĩa. Nếu loại thì phải loại cả từ, không xử lý riêng từng câu.

### 5.4 Chạy lại (5 experiment, hậu tố `_eval2`)

| Experiment mới | Copy từ | Ghi chú |
|---|---|---|
| `clip_vg_eval2` | `configs/clip_vg.yaml` | Không huấn luyện; chạy val rồi test 3 seed như cũ (ADR-0011) |
| `clip_nograph_vg_eval2` | `configs/clip_nograph_vg.yaml` (`lr = 1e-3`) | Cấu hình chính của đối chứng (`results/tables/README.md`) |
| `clip_graph_vg_lr1e4_eval2` | `configs/clip_graph_vg_lr1e4.yaml` | Mô hình chính |
| `clip_cooc_graph_vg_lr1e4_eval2` | `configs/clip_cooc_graph_vg_lr1e4.yaml` | Cần dựng cạnh đồng xuất hiện trên máy này (`scripts/build_concept_edges.py`) |
| `clip_graph_vg_lr1e4_shuffled_eval2` | `configs/clip_graph_vg_lr1e4_shuffled.yaml` | Dùng lại đồ thị xáo đã có (`shuffle_seed: 0`) |

- Mỗi config mới chỉ khác bản gốc ở `experiment`, `model_dir` và các trường mới cần cho mở rộng (đường dẫn dữ liệu văn
  bản, đường dẫn danh sách vị từ, `extra_predicates`, `excluded_predicates`). Trường mới phải có giá trị mặc định để
  config cũ vẫn nạp được với `load_yaml_dataclass`. Có test kiểm điều này, như các test so config hiện có trong `tests/test_graph_rerank_pipeline.py`.
- Không chạy lại `clip_graph_vg` (`lr = 1e-3`): không phải cấu hình chính.
- Chặn chạy test lần hai: giữ cơ chế `has_test_run` cho CLIP + Graph. Thêm cơ chế tương tự cho `evaluate_clip`, vì
  hiện CLIP thuần chỉ được nhắc trong docstring mà không bị chặn bằng code.

### 5.5 Bảng kết quả

- Không sửa `results/tables/main.csv`.
- Bảng mới (sinh bằng `scripts/summarize_runs.py`, mean ± std mẫu qua 3 seed, chỉ run có `git_dirty: false`):
  - `results/tables/eval2_text_to_image.csv`: R@1/5/10, MRR@100, nDCG@10 của 5 experiment.
  - `results/tables/eval2_image_to_text.csv`: cùng các chỉ số cho chiều ảnh → văn bản, kèm `α_i2t` của từng seed.
  - `results/tables/eval2_groups.csv`: dạng dài, mỗi dòng là (experiment, cách chia, nhóm), có số truy vấn và các chỉ số.
- `results/tables/README.md` mô tả bảng mới và quan hệ với `main.csv`. Nếu số R@K/MRR của `_eval2` lệch với dòng cũ,
  ghi rõ mức lệch và lý do (khác máy, huấn luyện lại).

## 6. Thứ tự thực hiện

1. Code và test: nDCG@10, chiều ảnh → văn bản, nhóm truy vấn, bảng mới, chặn test lần hai cho CLIP thuần. Commit từng
   phần nhỏ.
2. Kiểm tra luật nhóm trên val (mục 5.3); chỉnh nếu cần, chỉ dựa trên val.
   Kiểm tra code mới tái tạo đúng số val cũ: R@K và MRR@100 val của `clip_vg_eval2` phải bằng đúng số của `clip_vg` trên
   val (CLIP tất định). Nếu khác thì dừng và tìm nguyên nhân trước khi đi tiếp (quyết định của Claude Code).
3. **Chốt quy tắc:** ADR-0049 + 5 config `_eval2` + mục "Giao thức đánh giá v2" trong `results/tables/README.md`, commit
   **trước** mọi lần huấn luyện và chạy test mới.
4. Dựng cạnh đồng xuất hiện; huấn luyện và chọn trên val cho 4 mô hình đồ thị × 3 seed; chạy val cho CLIP thuần.
5. Chạy test đúng một lần cho mỗi experiment `_eval2` và mỗi seed.
6. Sinh bảng; hồ sơ `results/records/<ngày>_cap-do-1-mo-rong/` (tham số, số theo run, suy luận, quyết định, review, bản
   chép file run); cập nhật guide, checklist `README.md` mục Mở rộng, `docs/ai-usage-log.md`.
7. Review: `leakage-auditor`, `experiment-reviewer`, `pr-test-analyzer`; sửa theo review; mở PR riêng (không gộp vào
   PR #8).

## 7. Để người làm tự quyết

- Tên hàm, module và vị trí chính xác của code mới trong `src/mmxai/eval/`, miễn giữ một đường chấm test duy nhất.
- Định dạng khoá mới trong `metrics.json` và `selection.json`, miễn bảng ở mục 5.5 sinh được từ đó.
- Cách viết bảng của `summarize_runs` cho số theo nhóm.
- Có tách PR #8 và việc này thành nhánh khác không: mặc định nhánh mới tách từ `feat/level1-ablation-explain`, vì cần
  code giải thích và đồ thị xáo của PR #8.
