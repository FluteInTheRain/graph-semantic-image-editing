# ADR-0048: Thêm cạnh đồng xuất hiện giữa khái niệm, đếm trên annotation của train, và GraphSAGE hai lớp

- **Trạng thái:** Đề xuất
- **Ngày:** 2026-11-12

## Bối cảnh

Lần chạy 2026-11-12 (ADR-0047, `results/tables/main.csv`) cho thấy:

| Mô hình (test, 3 seed) | MRR@100 |
|---|---|
| CLIP thuần | 0,1076 |
| Đối chứng không đồ thị (`W_nb = 0`) | 0,1266 ± 0,0006 |
| CLIP + Graph (một lớp, ADR-0046) | 0,1276 ± 0,0015 |

Gần như toàn bộ phần tăng đến từ phép chiếu tuyến tính `W_self`; đồ thị chỉ thêm khoảng +0,001, cỡ một độ lệch chuẩn. Lý do đã nêu trước trong ADR-0046: cạnh `contains` và vector khái niệm **đều lấy từ CLIP**, nên trung bình khái niệm của một ảnh chỉ là một hàm khác của chính vector ảnh. Đồ thị không mang thông tin mà CLIP chưa có.

Muốn đồ thị đóng góp, nó phải mang **thông tin ngoài CLIP**. Ba hướng đã được xem xét (xem bảng phương án). qphulong chọn hướng 1 ngày 2026-11-12: đồng xuất hiện của khái niệm trong annotation Visual Genome.

Khái niệm cho người mới:

- **Đồng xuất hiện (co-occurrence):** hai khái niệm "đi cùng nhau" nếu chúng thường được gán nhãn trong cùng một ảnh, ví dụ *dog* và *leash*.
- **PMI (pointwise mutual information):** đo hai khái niệm đi cùng nhau nhiều hơn ngẫu nhiên bao nhiêu:

  ```
  PMI(a, b) = log( n_ab · N / (n_a · n_b) )
  ```

  `n_ab` là số ảnh train có cả `a` lẫn `b`, `n_a` là số ảnh train có `a`, `N` là số ảnh train. PMI > 0 nghĩa là đi cùng nhau nhiều hơn ngẫu nhiên. Đếm thô thì khái niệm nào cũng nối với *man*, *window* (có trong hầu hết ảnh); PMI ưu tiên các cặp đặc trưng.
- **GraphSAGE hai lớp:** lớp 1 cho mỗi khái niệm "nghe" các khái niệm đi cùng nó; lớp 2 (đúng mô hình ADR-0046) cho mỗi ảnh nghe các khái niệm của nó, nay đã được làm giàu. Một ảnh có *dog* (từ CLIP top-k) vì vậy nhận được thêm thông tin về *leash*, *collar*, dù CLIP không chọn chúng cho ảnh đó.

Đo thử trên annotation train (2026-11-12): 3.880 khái niệm, 152.580 cặp xuất hiện cùng nhau trong ≥ 10 ảnh train. Ví dụ hàng xóm PMI cao nhất: *dog* → *dog tag*, *leash*, *lab*, *tongue*; *surfboard* → *surfer*, *swell*, *foam*; *cake* → *fondant*, *birthday*, *wedding cake*.

## Quyết định

### 1. Cạnh khái niệm → khái niệm (`cooccurs_with`)

| Thành phần | Quy định | Code |
|---|---|---|
| Nguồn | Annotation đối tượng (`objects.json`) của **chỉ ảnh train**. Ánh xạ tên → synset dùng cùng từ điển đếm trên train như ADR-0045 | `run_build_concept_edges` |
| Đỉnh | Đúng 3.880 khái niệm của đồ thị `contains` (cùng thứ tự hàng; kiểm khi nạp) | `load_concept_edges_for` |
| Điều kiện một cặp | `n_ab ≥ 10` và PMI > 0 | `top_pmi_edges` |
| Số cạnh | Mỗi khái niệm `a` giữ tối đa `top_m = 10` bạn `b` có PMI cao nhất, cạnh có hướng `b → a` | `top_pmi_edges` |
| Lưu | `artifacts/graphs/visual_genome/concept_cooccurrence_pmi10_top10_v1/` (`concept_edges.npz` + `meta.json`), không commit | `save_concept_edges` |
| Config | `configs/concept_edges_vg_cooccurrence.yaml` | `scripts/build_concept_edges.py` |

Cạnh `contains` của mọi split **không đổi**: vẫn CLIP top-16 (ADR-0046). Ảnh val/test không dùng annotation.

### 2. Mô hình `CooccurrenceSage`

```
lớp 1 (khái niệm → khái niệm): z_c = W_cself · x_c + mean_{c' ∈ co(c)} W_co · x_c'
lớp 2 (khái niệm → ảnh):       h_i = W_self · x_i + b + mean_{c ∈ N(i)} W_nb · z_c
graph_score(q, i) = cos(q, h_i)
```

- Hai `SAGEConv` của PyG (`aggr="mean"`), không hàm kích hoạt. Lớp 1 không có bias (bias của lớp 2 là đủ).
- Khởi tạo `W_cself = I`, `W_co = 0` cộng khởi tạo ADR-0046: lúc chưa học, `z_c = x_c` nên mô hình **bằng đúng CLIP**.
- Một khái niệm không có cạnh đồng xuất hiện có `z_c = W_cself · x_c`.
- `num_layers = 2` cho `explain/`. Explainer có thể xoá cạnh của cả hai loại.

### 3. Huấn luyện và chọn: giữ nguyên ADR-0047

Cùng dữ liệu, cùng loss, cùng early stopping, cùng lưới `alpha`, cùng 3 seed, cùng hai script. Shortcut huấn luyện vẫn chính xác vì mô hình tuyến tính: tính trước hai trung bình cho mỗi ảnh (`neighbor_mean`, `second_hop_mean`), rồi

```
h_i = W_nb · (W_cself · neighbor_mean_i + W_co · second_hop_mean_i) + b + W_self · x_i
```

Có test chứng minh shortcut trùng với hai `SAGEConv` (`test_cooccurrence_shortcut_gives_the_same_vectors_as_sageconv`).

### 4. Thí nghiệm

`configs/clip_cooc_graph_vg.yaml` (experiment `clip_cooc_graph_vg`) **chỉ khác** `configs/clip_graph_vg.yaml` ở `experiment`, `model_dir` và `concept_edges_dir` (có test kiểm). Hiệu `clip_cooc_graph_vg − clip_graph_vg` là đóng góp của cạnh đồng xuất hiện; hiệu với `clip_nograph_vg` là đóng góp của toàn bộ đồ thị.

Quyết định bởi: qphulong, 2026-11-12 (chọn hướng 1: đồng xuất hiện trên annotation train, mô hình hai lớp). Các giá trị `n_ab ≥ 10`, `top_m = 10`, dùng PMI, cách khởi tạo do Claude Code đề xuất, **nhóm chưa duyệt**; chỉ chỉnh trên val, mỗi lần chỉnh là một tên experiment mới.

## Các phương án đã cân nhắc

| Phương án | Ưu | Nhược | Lý do không chọn |
|---|---|---|---|
| Đồng xuất hiện trên annotation train, PMI | Thông tin ngoài CLIP (nhãn người gán); dữ liệu có sẵn; không thêm thư viện | Chỉ đếm cặp trong cùng ảnh, không có loại quan hệ | **Được chọn** |
| Quan hệ WordNet (hypernym: *dog* → *animal*) | Khái niệm đã là synset WordNet | Cần thêm `nltk` + dữ liệu WordNet (dependency mới, cần ADR); quan hệ phân loại ít giúp câu mô tả vùng ảnh | Tốn thêm dependency, lợi ích không rõ hơn |
| ConceptNet | Quan hệ phong phú (*UsedFor*, *AtLocation*) | Tải dữ liệu lớn, phải ánh xạ sang synset, ghi nguồn và phiên bản | Nhiều việc nhất, để sau |
| Đếm thô thay vì PMI | Đơn giản hơn | Mọi khái niệm nối với *man*, *window*; cạnh gần như giống nhau cho mọi khái niệm | Không phân biệt được |
| Cạnh ảnh – ảnh (ảnh có chung khái niệm) | Đường đi ảnh – khái niệm – ảnh | Ảnh val/test ảnh hưởng lẫn nhau; ảnh test phải đọc ảnh khác | Phức tạp, dễ rò rỉ |
| Thêm hàm kích hoạt giữa hai lớp | Biểu diễn mạnh hơn | Mất shortcut chính xác (phải chạy `SAGEConv` trên 1,6 triệu cạnh mỗi bước, quá RAM laptop); khó vấn đáp | Giữ tuyến tính cho bản đầu |
| Đổi cạnh `contains` của train sang annotation | Cạnh sạch hơn lúc học | Lệch phân bố với val/test (ADR-0046) | Không đổi |

## Hệ quả

- **Rò rỉ dữ liệu (PDF §6, ADR-0009):**
  - Cạnh đồng xuất hiện là thống kê, đếm **chỉ** trên ảnh train: `run_build_concept_edges` chỉ dựng scene graph cho id train; `count_cooccurrence` nhận `train_ids` bắt buộc. Test: `test_statistical_edges_ignore_test_images`, `test_build_links_train_pairs_only_and_saves_them` (ảnh test có *cat* + *sofa* không tạo cạnh).
  - Ảnh val/test vẫn chỉ có cạnh CLIP top-k của chính nó, không đọc annotation.
  - Đồ thị khái niệm cố định, dùng chung cho mọi split, nên điểm của một ảnh vẫn chỉ phụ thuộc vào vector của nó, cạnh `contains` của nó và đồ thị khái niệm; ảnh val/test không ảnh hưởng lẫn nhau. Chấm trên đồ thị con của từng split vẫn chính xác.
  - `n_ab`, `top_m` và mọi siêu tham số chọn trên val; `queries/` không dùng.
- **Giải thích (ADR-0016):** giải thích giờ có đường đi hai bước thật trong cơ chế tính điểm: *ảnh ← dog ← leash*. Xoá cạnh `leash → dog` làm đổi điểm của đúng các ảnh có *dog* (test: `test_removing_a_concept_edge_changes_only_images_with_that_concept`). `explain/` cần đọc `num_layers = 2` và cho phép can thiệp cả hai loại cạnh.
- **Hạn chế:**
  - Ảnh chỉ nhận thông tin đồng xuất hiện qua khái niệm mà CLIP đã chọn cho nó; khái niệm CLIP chọn sai vẫn kéo theo bạn của nó.
  - Mô hình tuyến tính: `W_nb · W_co` là tích hai ma trận, nhiều tham số hơn mức cần (khoảng 1 triệu thay vì 0,5 triệu), nhưng vẫn rất nhỏ.
  - Nếu cạnh đồng xuất hiện không giúp trên val, kết quả âm vẫn phải báo cáo.
- **Việc tiếp theo:** dựng cạnh; huấn luyện 3 seed và chọn trên val; nếu nhóm chốt cấu hình, chạy test một lần. Ablation: xáo cạnh khái niệm giữ bậc (kiểm phần tăng đến từ đúng cặp khái niệm, không chỉ từ thêm tham số).

## Nguồn tham khảo

- Đề bài, §4.2, §6, §7.
- ADR-0009, ADR-0016, ADR-0045, ADR-0046, ADR-0047.
- `results/tables/main.csv` (kết quả 2026-11-12) và `docs/guides/huan-luyen-va-xep-hang-lai.md` mục 6.
- Mã nguồn `SAGEConv` của PyTorch Geometric 2.8.1 đã cài: tham số `bias` áp cho `lin_l`; `lin_r` không có bias; gọi với `x` dạng tuple `(x_nguồn, x_đích)` cho đồ thị hai phía.

## Kết quả và giới hạn quy trình (bổ sung 2026-11-12, khi ADR còn Đề xuất)

- Test (`results/tables/main.csv`): `clip_cooc_graph_vg_lr1e4` 0,1280 ± 0,0006 so với `clip_graph_vg_lr1e4` 0,1284 ± 0,0006 (MRR@100). **Không có bằng chứng** cạnh đồng xuất hiện giúp thêm.
- `lr = 1e-4` được thử sau khi đã xem test ở `lr = 1e-3` của mô hình một lớp và đối chứng. Quy tắc chọn theo val ("mỗi mô hình dùng `lr` tốt nhất trên val") được commit trước khi chạy test mới (`6c97c4c`). Cả hai dòng test của mô hình một lớp được giữ. Chi tiết: `results/tables/README.md`, `docs/guides/huan-luyen-va-xep-hang-lai.md` mục 7.
- Đề nghị nhóm: từ nay, mỗi biến thể mới của một họ mô hình đã chạy test cần một quy tắc chọn đã commit **trước** khi chạy test (ghi vào ADR-0047 khi duyệt).
