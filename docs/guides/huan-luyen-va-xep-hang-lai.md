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

### Chạy lại trên CPU (cho thành viên khác)

Không cần GPU: mô hình chỉ khoảng 0,5–1 triệu tham số. Các bước, theo thứ tự:

```bash
uv sync --group ml --group graph
nproc                                   # số luồng CPU; dùng cho --threads
# Cần sẵn: artifacts/features/... (chuan-bi-embedding-clip.md §2), data/visual_genome/objects.json (xay-do-thi.md §2)
uv run python scripts/build_graph.py --config configs/graph_vg_clip_topk.yaml
uv run python scripts/build_concept_edges.py --config configs/concept_edges_vg_cooccurrence.yaml   # chỉ cho mô hình đồng xuất hiện

# Cấu hình chính (results/tables/README.md); --threads = kết quả của nproc
uv run python scripts/train_graph_rerank.py --config configs/clip_graph_vg_lr1e4.yaml --seeds 0 1 2 --threads 4
```

| Bước | Thời gian đo được (laptop i5-1135G7, 4 luồng, WSL) | RAM |
|---|---|---|
| `build_graph.py` | vài phút | chưa đo |
| `build_concept_edges.py` | khoảng 1 phút (đọc `objects.json`) | chưa đo |
| `train_graph_rerank.py`, phần chuẩn bị (một lần cho cả 3 seed) | khoảng 8 phút (tìm hard negative) | khoảng 4–5 GB |
| Mỗi epoch: một lớp / đồng xuất hiện / đối chứng | khoảng 25 / 42 / 19 giây | |
| Cả 3 seed với `lr = 1e-4` (dừng ở epoch 7–10) | khoảng 20–30 phút | |
| `evaluate_graph_rerank.py`, mỗi seed | khoảng 15 giây | |

Lưu ý:

- Cảnh báo `CUDA initialization ... driver is too old` khi chạy là **vô hại**: code chạy trên CPU.
- **macOS:** chương trình dừng với lỗi `OMP: Error #15 ... libomp.dylib already initialized` (hoặc `Fatal Python error: Aborted` trong FAISS).
  Nguyên nhân: wheel `faiss-cpu` và `torch` mỗi bên mang một bản `libomp.dylib`. Cách sửa trên máy mình: trỏ bản
  của faiss sang bản của torch. Đây là sửa **môi trường cục bộ** trong `.venv`, không phải sửa repo; trên macOS,
  `uv run pytest` chỉ chạy hết khi đã sửa. Phải làm lại sau mỗi lần `uv sync` cài lại faiss. Không dùng
  `KMP_DUPLICATE_LIB_OK=TRUE`: cách đó làm test bị treo.

  ```bash
  # sao lưu bản của faiss, rồi trỏ nó sang bản của torch
  cp .venv/lib/python3.11/site-packages/faiss/.dylibs/libomp.dylib /tmp/faiss_libomp.dylib.bak
  ln -sf "$PWD/.venv/lib/python3.11/site-packages/torch/lib/libomp.dylib" \
         .venv/lib/python3.11/site-packages/faiss/.dylibs/libomp.dylib
  # hoàn tác: cài lại faiss
  uv sync --group ml --group graph --reinstall-package faiss-cpu
  ```
- Thiếu RAM: đóng bớt chương trình. Không giảm `batch_size` trong config đã commit; muốn đổi thì tạo config với tên experiment mới.
- Số val có thể lệch rất nhỏ so với bảng ở mục 6–7 nếu khác phiên bản thư viện hoặc số luồng.
- **Không chạy lại test** cho experiment đã có trong `results/tables/main.csv`. Script chỉ chặn khi tìm thấy run test trong `results/runs/` **trên chính máy đó** (thư mục này gitignored), nên trên máy khác nó không chặn được. Muốn kiểm tái lập, chỉ chạy train + val rồi so với `selection.json` / số val ở mục 7.

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

**Kết quả val 2026-11-12** (commit `203f97f`; 20.639 cạnh, 1.370 khái niệm không có cạnh; **chưa chạy test**):

| Mô hình (val MRR@100, 3 seed) | Điểm kết hợp | Điểm đồ thị một mình (`alpha = 0`) |
|---|---|---|
| CLIP thuần | 0,1061 | – |
| Đối chứng không đồ thị | 0,1262 ± 0,0017 | 0,1240 ± 0,0021 |
| CLIP + Graph một lớp | 0,1277 ± 0,0022 | 0,1243 ± 0,0004 |
| CLIP + Graph đồng xuất hiện (`clip_cooc_graph_vg`) | 0,1279 ± 0,0024 | 0,1237 ± 0,0006 |

Đọc: cạnh đồng xuất hiện **không giúp** trên val (chênh +0,0002, nhỏ hơn nhiều so với độ lệch chuẩn). Cả 3 seed lại tốt nhất ở epoch 1, rồi val giảm trong khi loss train vẫn giảm: mô hình học quá khớp rất nhanh. `alpha` chọn được dao động mạnh (0,4 / 0,1 / 0,3). Nguồn: `artifacts/models/visual_genome/clip_cooc_graph_vg/seed*/selection.json` và `results/runs/20261112-140*_clip_cooc_graph_vg_seed*/`.

**Thử `lr = 1e-4`** (commit `ed5873c`, configs `*_lr1e4.yaml`, cả ba mô hình; **chỉ val, chưa chạy test**):

| Mô hình (val MRR@100, 3 seed) | `lr` | Điểm kết hợp | Điểm đồ thị một mình | Epoch tốt nhất | `alpha` |
|---|---|---|---|---|---|
| Đối chứng không đồ thị | 1e-3 | 0,1262 ± 0,0017 | 0,1240 ± 0,0021 | 1 / 1 / 1 | 0,3 / 0,4 / 0,2 |
| Đối chứng không đồ thị | 1e-4 | 0,1246 ± 0,0003 | 0,1241 ± 0,0002 | 7 / 7 / 8 | 0,2 / 0,2 / 0,1 |
| CLIP + Graph một lớp | 1e-3 | 0,1277 ± 0,0022 | 0,1243 ± 0,0004 | 1 / 1 / 1 | 0,2 / 0,4 / 0,3 |
| CLIP + Graph một lớp | 1e-4 | 0,1281 ± 0,0008 | 0,1276 ± 0,0016 | 6 / 8 / 6 | 0,0 / 0,0 / 0,1 |
| CLIP + Graph đồng xuất hiện | 1e-3 | 0,1279 ± 0,0024 | 0,1237 ± 0,0006 | 1 / 1 / 1 | 0,4 / 0,1 / 0,3 |
| CLIP + Graph đồng xuất hiện | 1e-4 | 0,1282 ± 0,0010 | 0,1278 ± 0,0005 | 5 / 5 / 6 | 0,0 / 0,1 / 0,0 |

Đọc:

- Với `lr = 1e-4`, huấn luyện ổn định hơn: val tăng dần trong 5–8 epoch, std nhỏ hơn.
- Ở cùng `lr = 1e-4`, hai mô hình có đồ thị hơn đối chứng khoảng **+0,0035** (vài lần std). Nhưng nếu lấy `lr` tốt nhất trên val của **từng** mô hình (đối chứng: 1e-3, 0,1262), chênh lệch chỉ còn khoảng +0,002, cỡ một std của đối chứng.
- Cạnh đồng xuất hiện vẫn **không thêm gì** so với đồ thị một lớp (+0,0001).
- `alpha` gần 0: điểm cuối gần như chỉ là điểm đồ thị.
- Đường val dao động giữa các epoch (ví dụ 0,1244 → 0,1285), nên chọn epoch trên val rồi báo cáo chính val thì hơi lạc quan. Con số chính vẫn phải là test.

Nguồn: `artifacts/models/visual_genome/*_lr1e4/seed*/selection.json`, `results/runs/*_lr1e4_seed*/`.

**Quy tắc chọn trước khi chạy test** (qphulong chốt 2026-11-12, trước khi xem số test): mỗi mô hình dùng `lr` có điểm kết hợp val cao nhất. Đối chứng: `clip_nograph_vg` (1e-3, đã test). Một lớp: `clip_graph_vg_lr1e4`. Đồng xuất hiện: `clip_cooc_graph_vg_lr1e4`. Các bản khác không chạy test.

**Test** (mỗi seed đúng một lần, commit `6c97c4c`; nguồn `results/tables/main.csv`):

| Mô hình | R@1 | R@5 | R@10 | MRR@100 |
|---|---|---|---|---|
| CLIP thuần | 0,0584 | 0,1422 | 0,2018 | 0,1076 |
| Đối chứng không đồ thị (`clip_nograph_vg`) | 0,0689 ± 0,0006 | 0,1741 ± 0,0056 | 0,2457 ± 0,0043 | 0,1266 ± 0,0006 |
| CLIP + Graph một lớp (`clip_graph_vg_lr1e4`) | 0,0705 ± 0,0014 | 0,1765 ± 0,0037 | 0,2492 ± 0,0015 | **0,1284** ± 0,0006 |
| CLIP + Graph đồng xuất hiện (`clip_cooc_graph_vg_lr1e4`) | 0,0709 ± 0,0009 | 0,1792 ± 0,0012 | 0,2469 ± 0,0006 | 0,1280 ± 0,0006 |

MRR@100 test theo từng seed (từ `results/runs/*/metrics.json`):

| Seed | Đối chứng (1e-3) | Một lớp (1e-3) | Một lớp (1e-4) | Đồng xuất hiện (1e-4) | Một lớp 1e-4 − đối chứng |
|---|---|---|---|---|---|
| 0 | 0,1259 | 0,1258 | 0,1279 | 0,1273 | +0,0020 |
| 1 | 0,1269 | 0,1286 | 0,1283 | 0,1284 | +0,0014 |
| 2 | 0,1271 | 0,1283 | 0,1291 | 0,1282 | +0,0020 |

Đọc (đã sửa theo review `experiment-reviewer` và `leakage-auditor`):

- **Đồ thị đóng góp nhỏ, nhất quán ở cả 3 seed khi `lr = 1e-4`** (+0,0014 đến +0,0020), nhưng **không ổn định theo `lr`**: ở `lr = 1e-3`, mô hình một lớp chỉ hơn đối chứng +0,0010 và còn thua ở seed 0. Std của hiệu khoảng 0,0009, nên +0,0018 chỉ cỡ 2 std. Với 3 seed và khởi tạo cố định (std đánh giá thấp dao động thật), đây là **dấu hiệu**, chưa phải bằng chứng chắc. Cần thêm ablation xáo cạnh/đồ thị ngẫu nhiên (đã làm 2026-11-13: xem "Ablation: đồ thị xáo cạnh giữ bậc" ở cuối mục này).
- Phần lớn cải thiện so với CLIP (+0,019) vẫn đến từ phép chiếu học được, không phải đồ thị.
- `alpha` chọn được gần 0 ở `lr = 1e-4`: điểm cuối gần như chỉ còn điểm đồ thị. Vì vậy so sánh chính là với **đối chứng**, không phải với CLIP thuần.
- Cạnh đồng xuất hiện: **không có bằng chứng cải thiện** (−0,0004 MRR@100, chênh R@5/R@10 cũng nằm trong nhiễu). Chưa chứng minh được là nó kém hơn.

**Giới hạn về quy trình (phải ghi trong báo cáo):** ý định thử `lr = 1e-4` có **sau khi** đã xem test của `clip_graph_vg` và `clip_nograph_vg` ở `lr = 1e-3` (test lúc 11:06, config `lr = 1e-4` commit `ed5873c` lúc 14:10). Lý do thử là quan sát trên val (val tốt nhất ngay epoch 1), và quy tắc chọn được commit trước khi chạy test mới (`6c97c4c`), nhưng họ mô hình một lớp vì vậy đã được xem test hai lần. Trên val, hai `lr` của mô hình một lớp chênh rất ít (0,1281 so với 0,1277, std 0,0022). Giữ **cả hai** dòng test trong `results/tables/main.csv`, không bỏ dòng nào. Bảng nào là cấu hình chính: xem `results/tables/README.md`.

### Ablation: đồ thị xáo cạnh giữ bậc (2026-11-13)

**Câu hỏi:** mức hơn đối chứng (+0,0018) có đến từ việc *ảnh nào nối với khái niệm nào*, hay chỉ vì mô hình có thêm
một nhánh láng giềng, nối với khái niệm nào cũng được?

**Cách làm:**
- Giữ nguyên mô hình chính `clip_graph_vg_lr1e4` và mọi siêu tham số.
- Chỉ thay đồ thị bằng bản **xáo cạnh `contains` trong từng split**. Mỗi ảnh vẫn có 16 cạnh, mỗi khái niệm vẫn có
  đúng số cạnh như cũ trong split đó, nhưng ảnh nối với khái niệm ngẫu nhiên. Đây là ablation ADR-0046 §6.
- Code: `src/mmxai/graph/shuffle.py` (`shuffle_contains_edges`). Config: `configs/graph_vg_clip_topk_shuffled.yaml`,
  `configs/clip_graph_vg_lr1e4_shuffled.yaml`.
- Quy tắc báo cáo được commit **trước khi chạy** (`ea7068f`, `results/tables/README.md`): chạy test đúng một lần
  mỗi seed, báo cáo dù kết quả ra sao.

```bash
uv run python scripts/shuffle_graph.py --config configs/graph_vg_clip_topk_shuffled.yaml
uv run python scripts/train_graph_rerank.py --config configs/clip_graph_vg_lr1e4_shuffled.yaml --seeds 0 1 2 --threads 8
for s in 0 1 2; do uv run python scripts/evaluate_graph_rerank.py --config configs/clip_graph_vg_lr1e4_shuffled.yaml --seed $s; done
```

Thời gian trên MacBook (8 luồng): xáo đồ thị vài giây; huấn luyện 3 seed khoảng 5 phút; test khoảng 15 giây mỗi seed.

**Kết quả test** (`results/tables/main.csv`; MRR@100 theo seed lấy từ `results/runs/*/metrics.json`):

| Seed | Đồ thị thật (`clip_graph_vg_lr1e4`) | Đồ thị xáo (`_shuffled`) | Đối chứng không đồ thị (`clip_nograph_vg`) | Thật − xáo |
|---|---|---|---|---|
| 0 | 0,1279 | 0,12681 | 0,1259 | +0,00109 |
| 1 | 0,1283 | 0,12735 | 0,1269 | +0,00095 |
| 2 | 0,1291 | 0,12736 | 0,1271 | +0,00174 |
| mean ± std | 0,1284 ± 0,0006 | 0,12718 ± 0,00031 | 0,1266 ± 0,0006 | **+0,00126 ± 0,00042** |

Mean, std và hiệu tính trên số chưa làm tròn của bản xáo (`metrics.json`). Số của mô hình thật và đối chứng chỉ có
4 chữ số (mục 7 ở trên).

Trên val, đồ thị xáo được 0,1240 (3 seed gần như bằng nhau). Con số này **thấp hơn** cả đối chứng không đồ thị cùng
`lr = 1e-4` (0,1246) và thấp hơn đồ thị thật (0,1281). `alpha` chọn được là 0,1 / 0,2 / 0,1, tức mô hình tin CLIP
hơn so với khi dùng đồ thị thật (0 / 0 / 0,1).

**Đọc** (theo review `experiment-reviewer`, 2026-11-13):

- Đồ thị thật hơn đồ thị xáo ở **cả 3 seed** (+0,00095 đến +0,00174 MRR@100). Kết quả **phù hợp** với giả thuyết
  rằng một phần nhỏ cải thiện đến từ quan hệ ảnh – khái niệm thật. Không được viết là "chứng minh": chỉ có 3 seed,
  1 bản xáo, và hiệu ứng tuyệt đối khoảng 1% MRR. t ghép cặp = 5,16 (p ≈ 0,04, df = 2) chỉ là số mô tả: nó không tính
  dao động do cách xáo hay do mẫu truy vấn.
- So với đối chứng `lr = 1e-3` (cấu hình chính của đối chứng), xáo cạnh làm mất khoảng 70% phần chênh +0,0018. Vì
  đối chứng khác `lr`, 70% chỉ là ước lượng thô. Trên val, so với đối chứng **cùng** `lr = 1e-4` (0,1246), đồ thị xáo
  (0,1240) còn thấp hơn.
- R@1: +0,0019 (0,0705 so với 0,0686), lớn hơn std của mô hình chính (0,0014) nhưng sát nhiễu. R@5 (+0,0006) và
  R@10 (+0,0003) chênh nhỏ hơn std, không kết luận được.
- Không được viết "đồ thị xáo tốt hơn không đồ thị": chênh chỉ +0,00054, và hai mô hình dùng `lr` khác nhau.
- Bất thường cần ghi trong báo cáo: trên val, đồ thị xáo mất **toàn bộ** phần chênh (+0,0040 so với đồ thị thật),
  còn trên test chỉ mất +0,0013. Tỉ lệ cạnh xáo trùng cạnh gốc gần như bằng nhau ở val và test (1,78% và 1,89%), nên
  điều đó không giải thích được. Với 1 bản xáo thì không tách được dao động do cách xáo.

**Giới hạn:**
- Chỉ một `shuffle_seed` (0), như quy tắc đã chốt. FluteInTheRain quyết định (2026-11-13) **không** thêm bản xáo
  seed 1, 2, dù `experiment-reviewer` có đề xuất. Báo cáo phải ghi rõ giới hạn này.
- Toàn bộ số theo từng run, tham số, suy luận và review của lần chạy này: `results/records/20261113_cap-do-1/README.md`.
- Ba split dùng chung một bộ sinh số ngẫu nhiên, xáo theo thứ tự `test`, `train`, `val`. Vì vậy cách xáo của train
  phụ thuộc vào *số cạnh* của test. Phần này không mang nhãn hay nội dung test, và không cạnh nào vượt split
  (`leakage-auditor`: không phải rò rỉ). Không sửa code, vì đồ thị đã chốt trước khi chạy.
- Dòng mô hình chính trong `main.csv` chạy trên máy khác (commit `6c97c4c`). Dòng ablation chạy trên MacBook
  (commit `ea7068f`). Mô hình chính huấn luyện lại trên MacBook cho kết quả val khớp tới 4 chữ số (0,1281 ± 0,0008,
  epoch 6/8/6, `alpha` 0/0/0,1), nhưng chưa đối chiếu `model_sha256` của run test cũ.

## 8. Câu hỏi hay gặp khi vấn đáp

- **Vì sao chỉ một lớp?** Đơn giản, giải thích được bằng công thức, và điểm của một ảnh chỉ phụ thuộc cạnh của chính nó (ảnh test không ảnh hưởng lẫn nhau). Đổi sang hai lớp cần ADR mới.
- **Vì sao hard negative lấy từ top 100 của CLIP?** Đó đúng là những ảnh mà bước xếp hạng lại phải đẩy xuống.
- **Shortcut huấn luyện có "gian lận" không?** Không. Với một lớp, mean và không có hàm kích hoạt, trung bình vector khái niệm của mỗi ảnh không đổi khi học, nên tính trước một lần cho kết quả giống hệt. Có test chứng minh. Lúc chấm val/test và lúc giải thích vẫn dùng `SAGEConv` thật.
- **Đồ thị có thêm thông tin gì ngoài CLIP?** Ít: cạnh và vector khái niệm đều từ CLIP. Đó là lý do phải có đối chứng không đồ thị (ADR-0046, mục Hệ quả).
