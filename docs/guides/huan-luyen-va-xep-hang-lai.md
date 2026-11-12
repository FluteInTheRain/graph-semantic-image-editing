# Hướng dẫn: huấn luyện GraphSAGE và xếp hạng lại kết quả CLIP

> **Trạng thái 2026-11-12:** bản đầu, đã chạy thật trên laptop CPU, cả val lẫn test (số liệu ở mục 6). Thiết kế: [ADR-0046](../decisions/0046-gnn-mot-lop-huan-luyen-cho-truy-van-anh-voi-canh-clip-top-k.md) (mô hình) và [ADR-0047](../decisions/0047-quy-trinh-huan-luyen-chon-alpha-va-chay-test-cho-clip-graphsage.md) (quy trình), cả hai đang **Đề xuất**.

## 1. Ý tưởng trong 1 phút

CLIP tìm 100 ảnh gần câu truy vấn nhất. Ta **xếp lại** 100 ảnh đó bằng điểm kết hợp:

```
final = alpha · cos(q, x_i) + (1 − alpha) · graph_score(q, i)
```

`graph_score` đến từ một **GraphSAGE một lớp**. Mỗi ảnh "nghe" 16 khái niệm gần nó nhất (cạnh `contains` từ CLIP top-k):

```
 dog.n.01 ──┐
 sofa.n.01 ─┼──► [ảnh i]      h_i = W_self·x_i + b + trung bình_c(W_nb·x_c)
 lamp.n.02 ─┘                 graph_score(q, i) = cos(q, h_i)
```

- `x_i`, `x_c`, `q` là vector CLIP **đóng băng**. Chỉ `W_self`, `W_nb`, `b` được học (khoảng 525 nghìn số).
- Khởi tạo `W_self = I`, `W_nb = 0`, `b = 0`: lúc chưa học, `graph_score` **bằng đúng** cosine của CLIP.
- Code: `src/mmxai/models/sage.py` (`ConceptToImageSage`).

## 2. Ba bước, ba tập dữ liệu

| Bước | Dùng tập | Làm gì | Code |
|---|---|---|---|
| Huấn luyện | **train** | Mỗi region description của ảnh train: ảnh đúng + 15 ảnh sai lấy từ top 100 của CLIP (chỉ trong ảnh train). Loss InfoNCE | `src/mmxai/models/train.py` |
| Chọn | **val** | Sau mỗi epoch, đo MRR@100 của `graph_score` một mình; giữ epoch tốt nhất. Rồi thử `alpha` từ 0.0 tới 1.0, chọn MRR@100 cao nhất (hoà thì `alpha` lớn hơn) | `fit`, `sweep_alpha`, `choose_alpha` |
| Báo cáo | **test** | Script riêng nạp mô hình và `alpha` đã chọn, chấm **một lần**, không chọn gì | `evaluate_test` |

Tại sao tách như vậy (PDF §6): nếu test được dùng để chọn epoch hay `alpha`, con số báo cáo sẽ đẹp giả tạo.

**Đối chứng không đồ thị** (`configs/clip_nograph_vg.yaml`): huấn luyện y hệt, nhưng `W_nb` cố định bằng 0. Nếu đối chứng này tốt ngang CLIP + Graph, thì cải thiện đến từ một phép chiếu tuyến tính học trên Visual Genome, **không** phải từ đồ thị.

## 3. Chuẩn bị

| Cần gì | Lấy ở đâu |
|---|---|
| Thư viện | `uv sync --group ml --group graph` |
| Embedding CLIP của VG | `docs/guides/chuan-bi-embedding-clip.md` §2 → `artifacts/features/visual_genome/ViT-B-32__laion2b_s34b_b79k/` |
| Đồ thị CLIP top-16 | `uv run python scripts/build_graph.py --config configs/graph_vg_clip_topk.yaml` (vài phút, CPU được) → `artifacts/graphs/visual_genome/contains_clip_top16_v1/` |
| Máy | CPU là đủ. Laptop i5-1135G7 (WSL, 4 luồng) đo được khoảng 220 GFLOP/s; cần khoảng 4–5 GB RAM |

## 4. Chạy

**Quan trọng:** commit code trước khi chạy, và **không sửa file đã track** trong lúc chạy. Mỗi run ghi `git_dirty`, và `scripts/summarize_runs.py` bỏ run có code chưa commit.

```bash
# 1. Train + chọn trên val (chạy bao nhiêu lần cũng được khi phát triển; xoá artifacts/models/... để chạy lại)
uv run python scripts/train_graph_rerank.py --config configs/clip_graph_vg.yaml --seeds 0 1 2 --threads 4
uv run python scripts/train_graph_rerank.py --config configs/clip_nograph_vg.yaml --seeds 0 1 2 --threads 4

# 2. Test: MỘT lần mỗi seed, sau khi đã chốt config
for s in 0 1 2; do
  uv run python scripts/evaluate_graph_rerank.py --config configs/clip_graph_vg.yaml --seed $s
  uv run python scripts/evaluate_graph_rerank.py --config configs/clip_nograph_vg.yaml --seed $s
done

# 3. Bảng tổng hợp
uv run python scripts/summarize_runs.py
```

Kết quả ra đâu:

| File | Nội dung |
|---|---|
| `artifacts/models/visual_genome/<experiment>/seed<k>/model.pt` | Trọng số tốt nhất (không commit) |
| `artifacts/models/visual_genome/<experiment>/seed<k>/selection.json` | `alpha`, epoch tốt nhất, lịch sử val, kết quả từng `alpha` |
| `results/runs/<thời gian>_<experiment>_seed<k>/` | `config.yaml`, `metrics.json` (split, `alpha`, hash split, hash mô hình), `env.json` |
| `results/tables/main.csv` | Mean ± std qua 3 seed: số duy nhất được đưa vào báo cáo |

## 5. Các điều bắt buộc và test chứng minh

| Điều bắt buộc | Test |
|---|---|
| Chưa học thì điểm đồ thị = CLIP | `test_untrained_model_scores_exactly_like_clip` |
| `alpha = 1` cho đúng metric của CLIP thuần | `test_alpha_one_gives_exactly_the_plain_clip_metrics` |
| Huấn luyện không thấy ảnh val/test | `test_training_never_sees_val_or_test_images` |
| Mỗi split chấm trên đồ thị con của nó, chỉ cạnh khái niệm → ảnh | `test_split_graph_holds_only_concept_to_image_edges_of_that_split` |
| Shortcut huấn luyện trùng với `SAGEConv` | `test_training_shortcut_gives_the_same_vectors_as_sageconv` |
| Xoá một cạnh chỉ đổi điểm của đúng ảnh đó (cần cho `explain/`) | `test_removing_an_edge_changes_only_its_image` |
| Test dùng `alpha` đã chọn trên val, chỉ chạy một lần | `test_test_uses_the_saved_alpha_and_runs_only_once` |
| Ảnh có cùng câu không bị coi là ảnh sai | `test_image_with_the_same_sentence_is_not_a_negative` |
| Cùng seed cho cùng trọng số | `test_same_seed_gives_the_same_trained_weights` |

## 6. Kết quả lần chạy 2026-11-12

Commit `8ac8e63`, laptop CPU 4 luồng. Mỗi seed: chuẩn bị dữ liệu khoảng 8 phút (một lần cho cả 3 seed), mỗi epoch khoảng 25 giây. Ở cả 6 lần huấn luyện, val tốt nhất sau **epoch 1** rồi giảm dần, nên early stopping dừng ở epoch 3.

**Test** (mỗi seed chạy đúng một lần; nguồn: `results/tables/main.csv`, cùng `split_sha256` với baseline):

| Mô hình | R@1 | R@5 | R@10 | MRR@100 | `alpha` chọn trên val (seed 0/1/2) |
|---|---|---|---|---|---|
| CLIP thuần (`clip_vg`) | 0,0584 | 0,1422 | 0,2018 | 0,1076 | – |
| Đối chứng không đồ thị (`clip_nograph_vg`) | 0,0689 ± 0,0006 | 0,1741 ± 0,0056 | 0,2457 ± 0,0043 | 0,1266 ± 0,0006 | 0,3 / 0,4 / 0,2 |
| **CLIP + Graph** (`clip_graph_vg`) | 0,0692 ± 0,0015 | 0,1750 ± 0,0056 | 0,2495 ± 0,0002 | 0,1276 ± 0,0015 | 0,2 / 0,4 / 0,3 |

**Val** (dùng để chọn, không báo cáo như kết quả chính): CLIP 0,1061; đối chứng 0,1262 ± 0,0017; CLIP + Graph 0,1277 ± 0,0022 (MRR@100).

**Đọc kết quả thế nào:**

- Xếp hạng lại giúp rõ: MRR@100 tăng khoảng **+0,020** (gần +19% tương đối) so với CLIP thuần.
- Nhưng **gần như toàn bộ** phần tăng đó đối chứng không đồ thị cũng có. Đồ thị chỉ thêm khoảng +0,001 MRR@100 và +0,004 R@10, cỡ một độ lệch chuẩn. Phần lớn cải thiện đến từ phép chiếu tuyến tính `W_self` học trên câu của Visual Genome, **không** phải từ cấu trúc đồ thị.
- Điều này khớp với hạn chế đã nêu trong ADR-0046: cạnh và vector khái niệm đều lấy từ CLIP, nên đồ thị gần như không mang thông tin mới. Kết quả này phải báo cáo trung thực, kể cả khi nó không đẹp.
- Std nhỏ vì khởi tạo cố định (`W_self = I`, `W_nb = 0`); seed chỉ đổi thứ tự batch và ảnh sai được lấy mẫu.

**Lần chạy đầu bị dừng:** bản đầu huấn luyện trên mọi câu train và học được cách đảo ngược CLIP (val tụt từ 0,106 xuống 0,038). Lý do và cách sửa: ADR-0047, mục "Lần chạy đầu".

**Việc tiếp theo gợi ý** (đều chọn trên val, test mới phải đặt tên experiment mới):

1. Ablation bắt buộc của cấp độ 1: xáo cạnh giữ bậc, khái niệm ngẫu nhiên, cạnh annotation (cận trên, ADR-0045).
2. Thử `lr` nhỏ hơn (val đạt đỉnh ngay epoch 1) và `k` khác (8, 32).
3. Nguồn cạnh mang thông tin ngoài CLIP (ví dụ ConceptNet giữa các khái niệm), nếu nhóm muốn đồ thị đóng góp nhiều hơn.

**Lưu ý tái lập:** đồ thị `contains_clip_top16_v1` được dựng lúc cây làm việc có một thay đổi chưa commit (`git_dirty: true` trong `meta.json`). Thay đổi đó chỉ thêm `split_subgraph`, hàm mà việc dựng đồ thị không gọi, nên nội dung đồ thị không bị ảnh hưởng. Dựng lại từ commit sạch thì xoá thư mục rồi chạy lại `build_graph.py`.

## 7. Mô hình hai lớp với cạnh đồng xuất hiện (ADR-0048)

Mục 6 cho thấy đồ thị một lớp gần như không giúp, vì mọi thứ trong nó đều lấy từ CLIP. ADR-0048 thêm thông tin CLIP không có: hai khái niệm thường được **người gán nhãn** cùng trong một ảnh train (đo bằng PMI).

```
 leash.n.01 ──► dog.n.01 ──┐
 collar.n.01 ─┘            ├──► [ảnh i]      lớp 1: z_c = W_cself·x_c + trung bình_c'(W_co·x_c')
               sofa.n.01 ──┘                 lớp 2: h_i = W_self·x_i + b + trung bình_c(W_nb·z_c)
```

- Cạnh `contains` không đổi (CLIP top-16). Chỉ thêm cạnh khái niệm → khái niệm, đếm trên **train**.
- Code: `src/mmxai/graph/cooccurrence.py` (dựng cạnh), `CooccurrenceSage` trong `src/mmxai/models/sage.py`.
- Huấn luyện, chọn `alpha`, chạy test: y hệt mục 2–4, chỉ đổi config.

```bash
uv run python scripts/build_concept_edges.py --config configs/concept_edges_vg_cooccurrence.yaml
uv run python scripts/train_graph_rerank.py --config configs/clip_cooc_graph_vg.yaml --seeds 0 1 2 --threads 4
```

So sánh cần đọc: `clip_cooc_graph_vg − clip_graph_vg` = đóng góp của cạnh đồng xuất hiện.

## 8. Câu hỏi hay gặp khi vấn đáp

- **Vì sao chỉ một lớp?** Đơn giản, giải thích được bằng công thức, và điểm của một ảnh chỉ phụ thuộc cạnh của chính nó (ảnh test không ảnh hưởng lẫn nhau). Đổi sang hai lớp cần ADR mới.
- **Vì sao hard negative lấy từ top 100 của CLIP?** Đó đúng là những ảnh mà bước xếp hạng lại phải đẩy xuống.
- **Shortcut huấn luyện có "gian lận" không?** Không. Với một lớp, mean và không có hàm kích hoạt, trung bình vector khái niệm của mỗi ảnh không đổi khi học, nên tính trước một lần cho kết quả giống hệt. Có test chứng minh. Lúc chấm val/test và lúc giải thích vẫn dùng `SAGEConv` thật.
- **Đồ thị có thêm thông tin gì ngoài CLIP?** Ít: cạnh và vector khái niệm đều từ CLIP. Đó là lý do phải có đối chứng không đồ thị (ADR-0046, mục Hệ quả).
