# ADR-0052: Chạy G-Retriever bậc 2 trên Flickr30k bằng cách dùng lại cạnh PMI của Visual Genome

- **Trạng thái:** Chấp nhận (nhóm chấp nhận 2026-11-17)
- **Ngày:** 2026-11-17

## Bối cảnh

ADR-0051 định nghĩa G-Retriever bậc 2 (`clip_pcst_sage_*`): PCST chọn đồ thị con `S_q` của truy vấn trên **cạnh khái niệm–khái niệm PMI**, rồi `SubgraphSage` học điểm đồ thị từ `S_q`. Cạnh PMI được đếm từ **annotation đối tượng của train** (ADR-0048).

ADR-0050 chạy lại pipeline cấp độ 1 trên tập thứ hai Flickr30k, **chỉ đổi dữ liệu, không chỉnh siêu tham số**. Nhóm muốn làm giống vậy cho G-Retriever bậc 2, kèm đối chứng cùng cấu hình `clip_nograph_flickr30k_lr1e4` như trên VG.

Vấn đề: Flickr30k **không có annotation đối tượng**, nên không đếm được PMI kiểu ADR-0048 trên chính Flickr30k.

Khái niệm cho người mới:

- **PMI**: điểm cho biết hai khái niệm xuất hiện chung nhiều hơn mức ngẫu nhiên.
- **Từ vựng khái niệm**: danh sách đỉnh của đồ thị. Đồ thị Flickr30k chép nguyên từ vựng của VG (`src/mmxai/graph/vocabulary_transfer.py`): cùng 3.880 khái niệm, cùng thứ tự. Nhờ vậy hàng `i` của cạnh PMI VG cũng là khái niệm `i` của đồ thị Flickr30k.

## Quyết định

Nhóm chọn ngày 2026-11-17: **dùng lại cạnh PMI của VG** (`artifacts/graphs/visual_genome/concept_cooccurrence_pmi10_top10_v1`) cho Flickr30k.

| Mục | Quy định |
|---|---|
| Cạnh PMI | Của VG, không dựng lại. Chỉ đếm trên **train VG** (ADR-0048) |
| Đồ thị ảnh–khái niệm | Của Flickr30k: `artifacts/graphs/flickr30k/contains_clip_top16_v1`, dựng bằng `scripts/build_graph_from_vocabulary.py --config configs/graph_flickr30k_clip_topk.yaml` |
| Kiểm khớp từ vựng | Code hiện có kiểm `np.array_equal(edges.concept_keys, graph.concept_keys)` |
| Siêu tham số | Giữ nguyên bản VG: `k = 3`, `C = 0,5`, `lr = 1e-4`... Không chỉnh gì |
| Đối chứng | `configs/clip_nograph_flickr30k_lr1e4.yaml`, cùng cấu hình huấn luyện, không đồ thị |

### Thay đổi code

`load_concept_edges_for` (`src/mmxai/eval/graph_rerank.py`) trước đây bắt cạnh PMI phải được đếm trên **chính** `split_file` của thí nghiệm. Với cạnh VG, nó sẽ báo lỗi. Cách sửa:

- Thêm trường tuỳ chọn `concept_edges_split_file: str | None = None` vào `GraphRerankConfig`, `PcstRerankConfig` và `PcstSageConfig`.
- Nếu trường này được đặt, `split_sha256` trong meta của cạnh được so với nó; nếu không, so với `split_file` như cũ.
- Config VG **không sửa**. Run cũ vẫn tái lập được.

### Config mới

| File | Bản gốc VG | Khác bản gốc ở |
|---|---|---|
| `configs/clip_pcst_sage_flickr30k.yaml` | `configs/clip_pcst_sage_vg.yaml` | `experiment`, `dataset`, `split_file`, `features_dir`, `graph_dir`, `cache_dir`, `model_dir`; thêm `concept_edges_split_file: splits/visual_genome/split_seed0.json`. `concept_edges_dir` **giữ nguyên** (cạnh VG) |
| `configs/clip_nograph_flickr30k_lr1e4.yaml` | `configs/clip_nograph_vg_lr1e4.yaml` | `experiment`, `dataset`, `split_file`, `features_dir`, `graph_dir`, `model_dir` |

Test `tests/test_flickr30k_pcst_configs.py` kiểm hai điều: mọi khoá khác giữ đúng giá trị của bản VG, và các đường dẫn dữ liệu trỏ tới `flickr30k`.

### Chống rò rỉ (ADR-0009, PDF §6)

- Cạnh PMI chỉ đếm trên **train VG**. Không dùng ảnh, caption hay split nào của Flickr30k.
- Flickr30k vẫn dùng split riêng của ADR-0050; `alpha` và epoch chỉ chọn trên val Flickr30k; test chạy **một lần**.
- VG và Flickr30k là hai tập khác nhau, nên không có ảnh test Flickr30k nào ảnh hưởng tới cạnh.

## Các phương án đã cân nhắc

| Phương án | Ưu | Nhược | Lý do không chọn |
|---|---|---|---|
| Dùng lại cạnh PMI của VG | Không thêm cách dựng cạnh; cùng hướng với ADR-0050 (chép từ vựng VG); không rò rỉ | Từ vựng VG có thể thiếu khái niệm hay gặp trong Flickr30k | **Được chọn** |
| Đếm PMI trên caption train Flickr30k | Cạnh sát dữ liệu Flickr30k | Là cách dựng cạnh mới, cần thêm code, khác cách làm trên VG nên khó so sánh | Thêm phạm vi và khác thiết kế gốc |
| Đếm PMI từ cạnh CLIP top-16 | Không cần annotation | Cạnh vẫn suy từ CLIP, đúng điểm yếu đã thấy trên VG (ADR-0048) | Không mang thông tin ngoài CLIP |

## Hệ quả

- Tốt: thay đổi code nhỏ (một trường cấu hình, một chỗ kiểm tra); run VG không bị ảnh hưởng.
- Giới hạn: từ vựng VG có thể thiếu khái niệm hay gặp ở Flickr30k. Giới hạn này đã ghi trong ADR-0050.
- Ngoài phạm vi: G-Retriever bậc 1, ablation `no_tree` và `shuffled` trên Flickr30k; bộ truy vấn trực tiếp/gián tiếp trên Flickr30k.
- Chạy thật do người dùng chạy trên Colab theo `docs/guides/huan-luyen-va-xep-hang-lai.md` mục 11. Chưa có kết quả nào.
- **Trùng số ADR-0050:** nhánh Flickr30k và PR #14 (ConceptNet) cùng dùng số 0050. Một trong hai phải đổi số trước khi cả hai vào `main`. Nhóm quyết định; ADR này không đổi số.
- Việc tiếp theo: nhóm duyệt ADR này, rồi mới chạy test một lần mỗi seed.

## Nguồn tham khảo

- `docs/superpowers/specs/2026-11-17-g-retriever-flickr30k-design.md`.
- ADR-0048 (cạnh PMI), ADR-0050 (Flickr30k), ADR-0051 (G-Retriever).
- Đề bài `Project03_Multimodal_XAI.pdf`, §5 và §6.
