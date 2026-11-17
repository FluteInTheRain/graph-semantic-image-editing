# G-Retriever bậc 2 trên Flickr30k: thiết kế

- **Ngày:** 2026-11-17
- **Nhánh:** `feat/g-retriever-flickr30k`, tách từ `feat/g-retriever-stage2` rồi gộp `origin/feat/run-pipeline-on-flickr30k`.
- **Liên quan:** ADR-0051 (G-Retriever), ADR-0048 (cạnh PMI), ADR Flickr30k của nhánh `feat/run-pipeline-on-flickr30k` (đang đánh số 0050). ADR mới cho thiết kế này: **ADR-0052**.

## 1. Mục tiêu

Chạy G-Retriever bậc 2 (`clip_pcst_sage_*`, ADR-0051) trên tập thứ hai Flickr30k, **giống cách nhánh Flickr30k chạy lại cấp độ 1**: chỉ đổi dữ liệu, không chỉnh siêu tham số nào. Kèm đối chứng cùng cấu hình `clip_nograph_flickr30k_lr1e4`, như trên VG.

## 2. Vấn đề

PCST chạy trên cạnh khái niệm–khái niệm PMI. Cạnh này đếm từ annotation đối tượng của train (ADR-0048), nhưng Flickr30k không có annotation đối tượng.

## 3. Quyết định (nhóm chọn 2026-11-17)

**Dùng lại cạnh PMI của VG** (`artifacts/graphs/visual_genome/concept_cooccurrence_pmi10_top10_v1`).

- Đồ thị Flickr30k chép nguyên từ vựng của VG: cùng 3.880 khái niệm, cùng thứ tự (`src/mmxai/graph/vocabulary_transfer.py`). Vì vậy hàng của cạnh PMI khớp ngay với khái niệm của đồ thị Flickr30k. Code hiện có đã kiểm điều này bằng `np.array_equal(edges.concept_keys, graph.concept_keys)`.
- Chống rò rỉ: cạnh PMI chỉ đếm trên **train VG**, hoàn toàn không dùng dữ liệu nào của Flickr30k.
- Hai phương án bị loại:
  - đếm PMI trên caption train Flickr30k: là một cách dựng cạnh mới, cần thêm code và khác cách làm trên VG;
  - đếm PMI từ cạnh CLIP top-16: cạnh vẫn suy từ CLIP, đúng điểm yếu đã thấy trên VG.

## 4. Thay đổi code (nhỏ)

`load_concept_edges_for` (`src/mmxai/eval/graph_rerank.py`) hiện bắt cạnh PMI phải được đếm trên **chính** `split_file` của thí nghiệm, nên sẽ báo lỗi với cạnh của VG. Cách sửa:

- Thêm trường tuỳ chọn `concept_edges_split_file: str | None = None` vào `GraphRerankConfig`, `PcstRerankConfig` và `PcstSageConfig`. Cả ba đều đi qua `load_inputs`.
- `load_concept_edges_for` so `split_sha256` trong meta của cạnh với `concept_edges_split_file` nếu trường này được đặt, ngược lại so với `split_file`. Hành vi cũ không đổi khi trường để trống.
- Config VG **không sửa**. Run cũ vẫn tái lập được.

## 5. Config mới

| File | Bản gốc VG | Khác bản gốc ở |
|---|---|---|
| `configs/clip_pcst_sage_flickr30k.yaml` | `clip_pcst_sage_vg.yaml` | `experiment`, `dataset`, `split_file`, `features_dir`, `graph_dir`, `cache_dir`, `model_dir`, thêm `concept_edges_split_file: splits/visual_genome/split_seed0.json`. `concept_edges_dir` **giữ nguyên** (cạnh VG) |
| `configs/clip_nograph_flickr30k_lr1e4.yaml` | `clip_nograph_vg_lr1e4.yaml` | `experiment`, `dataset`, `split_file`, `features_dir`, `graph_dir`, `model_dir` |

Một test (`tests/test_flickr30k_pcst_configs.py`) kiểm hai điều: mọi khoá khác giữ đúng giá trị của bản VG, và các đường dẫn dữ liệu trỏ tới `flickr30k`.

## 6. Ngoài phạm vi

- G-Retriever bậc 1, ablation `no_tree` và `shuffled` trên Flickr30k.
- Chạy thật: người dùng chạy trên Colab theo hướng dẫn. Plan chỉ viết code, config, test và tài liệu.
- Bộ truy vấn trực tiếp/gián tiếp trên Flickr30k.

## 7. Thời gian chạy ước tính

Khoảng 25 ms cho mỗi lần gọi PCST:
- train khoảng 105k caption: khoảng 45 phút;
- val 25k và test 25k: mỗi tập khoảng 10 phút.

Cache dùng chung cho 3 seed.

## 8. Rủi ro và ghi chú

- **Trùng số ADR:** nhánh Flickr30k dùng ADR-0050, PR #14 cũng dùng ADR-0050 (ConceptNet). Một trong hai phải đổi số trước khi cả hai vào `main`. Nhóm quyết định; plan này không đổi số.
- Từ vựng VG có thể thiếu khái niệm hay gặp trong Flickr30k. Giới hạn này đã được ghi trong ADR Flickr30k.
