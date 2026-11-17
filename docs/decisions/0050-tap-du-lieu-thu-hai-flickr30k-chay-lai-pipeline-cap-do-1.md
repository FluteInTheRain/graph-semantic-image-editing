# ADR-0050: Tập dữ liệu thứ hai Flickr30k: chạy lại pipeline cấp độ 1, chỉ đổi dữ liệu

- **Trạng thái:** Đề xuất
- **Ngày:** 2026-11-17

## Bối cảnh

Cấp độ 2 (PDF §5) yêu cầu đánh giá trên **tập dữ liệu thứ hai**. Embedding CLIP của Flickr30k (bản Karpathy) đã có sẵn trong `artifacts/features/flickr30k/ViT-B-32__laion2b_s34b_b79k/`:
- 31.014 ảnh, mỗi ảnh 5 caption;
- trích bằng cùng mô hình CLIP như Visual Genome (ADR-0041).

Nhóm muốn câu trả lời đơn giản nhất cho câu hỏi "kết quả cấp độ 1 có lặp lại trên dữ liệu khác không?". Vì vậy pipeline cấp độ 1 được chạy lại **y nguyên**, chỉ đổi dữ liệu.

Có hai chỗ không thể chạy y nguyên:

1. **Từ vựng khái niệm.** Trên VG, đỉnh khái niệm là các synset xuất hiện trong ít nhất 5 ảnh train, đếm từ `objects.json` (ADR-0045). Flickr30k chỉ có caption, không có annotation đối tượng.
2. **Mô hình đồng xuất hiện** (ADR-0048) đếm PMI trên annotation đối tượng, nên cũng không có trên Flickr30k.

## Quyết định

**1. Split.** Dùng đúng quy trình ADR-0042:
- `scripts/make_splits.py` với `configs/splits_flickr30k.yaml`: 5.000 val, 5.000 test, phần còn lại là train; seed 0; ảnh có cosine ≥ 0,99 vào cùng tập.
- Kết quả: train 21.014 / val 5.000 / test 5.000, có 2 cặp ảnh trùng.
- File: `splits/flickr30k/split_seed0.json`.
- Trường `split` trong file Karpathy vẫn **không** được đọc.

**2. Đồ thị.** Đỉnh khái niệm (khoá synset, tên, vector văn bản CLIP) được **chép** từ đồ thị chính của VG `contains_clip_top16_v1`: 3.880 khái niệm.
- Mỗi ảnh Flickr30k nối tới 16 khái niệm gần nhất theo cosine CLIP. Cách này giống hệt ADR-0046, cùng `k = 16`.
- Code: `src/mmxai/graph/vocabulary_transfer.py`, `scripts/build_graph_from_vocabulary.py`. Config: `configs/graph_flickr30k_clip_topk.yaml`.
- Kết quả: 31.014 ảnh, 496.224 cạnh.

**3. Mô hình.** Chạy 4 experiment: bản sao config VG, chỉ đổi tên experiment và đường dẫn dữ liệu/đồ thị/mô hình. `tests/test_flickr30k_configs.py` kiểm điều này.

| Flickr30k | Bản gốc VG |
|---|---|
| `clip_flickr30k` | `clip_vg` |
| `clip_nograph_flickr30k` | `clip_nograph_vg` (lr 1e-3, dòng chính trong `main.csv`) |
| `clip_graph_flickr30k_lr1e4` | `clip_graph_vg_lr1e4` |
| `clip_graph_flickr30k_lr1e4_shuffled` | `clip_graph_vg_lr1e4_shuffled` (đồ thị `configs/graph_flickr30k_clip_topk_shuffled.yaml`) |

- Không chỉnh lại siêu tham số nào trên Flickr30k, kể cả `lr` và `k`.
- `alpha` và epoch vẫn được chọn trên **val Flickr30k** như ADR-0047, vì đó là một phần của quy trình chứ không phải siêu tham số cố định.
- Test chạy **một lần** cho mỗi seed 0, 1, 2. Báo cáo mọi số, dù kết quả ra sao.

**4. Giải thích cơ bản của cấp độ 1.** Chạy `scripts/explain.py` trên val như VG: 5 thành công + 5 thất bại, Occlusion và kiểm tra xoá cạnh, trên `clip_graph_flickr30k_lr1e4` seed 0.
- Code đã được sửa để đọc được id câu của Flickr30k (vị trí caption trong ảnh, `caption_index`).
- Câu truy vấn được tra theo cặp (ảnh, vị trí caption), đọc từ `dataset_flickr30k.json`.

**5. Không chạy** mô hình đồng xuất hiện và giao thức v2 (ADR-0049) trên Flickr30k:
- mô hình đồng xuất hiện cần annotation đối tượng;
- nhóm truy vấn theo vị từ trong ADR-0049 được chốt trên câu VG.

## Hệ quả

- Rò rỉ (PDF §6): từ vựng chỉ đếm trên **train VG**, nên không chứa thông tin của Flickr30k. Cạnh của mỗi ảnh chỉ phụ thuộc vector của chính ảnh đó, như trên VG. Không có thống kê nào đếm trên test Flickr30k.
- Đồ thị Flickr30k dùng từ vựng của VG. Một số khái niệm hay gặp trong Flickr30k (ví dụ hoạt động của người) có thể không có trong từ vựng. Đây là giới hạn cần ghi khi báo cáo.
- Train Flickr30k (21k ảnh, khoảng 105k caption) nhỏ hơn VG nhiều, nên mô hình đồ thị có ít dữ liệu huấn luyện hơn.
- Cách này chưa trả lời câu hỏi "phương pháp 2024+" hay "≥ 20 truy vấn" của cấp độ 2; đó là các việc riêng.
