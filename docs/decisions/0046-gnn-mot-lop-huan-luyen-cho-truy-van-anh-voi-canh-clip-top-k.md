# ADR-0046: Huấn luyện GNN một lớp, tách được đóng góp, trực tiếp cho truy vấn văn bản → ảnh, với cạnh `contains` lấy từ CLIP top-k

- **Trạng thái:** Đề xuất
- **Ngày:** 2026-11-10

## Bối cảnh

Cấp độ 1 (PDF §4.2, §5) cần một GNN (GCN/GAT/GraphSAGE) tính **điểm đồ thị** cho 100 ứng viên mà CLIP trả về, rồi xếp hạng lại:

```
final = alpha * cos(query, image) + (1 - alpha) * graph_score(query, image)
```

ADR-0045 đã dựng đồ thị ảnh – khái niệm, nhưng chưa quy định GNN **học bài toán gì** để `graph_score` có ích. Review PR #6 và phần Hệ quả của ADR-0045 nêu hai vấn đề rò rỉ phải giải quyết trước khi huấn luyện:

- **A1.** `to_pyg_data` gộp ảnh train, val, test vào một đồ thị, không đánh dấu split. Huấn luyện trên đó là GNN "nhìn thấy" ảnh test (PDF §6).
- **A2.** Cạnh `contains` của ảnh test lấy từ annotation, mà đáp án của truy vấn (ADR-0044) cũng lấy từ **cùng annotation**. Điểm đồ thị khi đó là đọc đáp án.

Khái niệm cho người mới:

- **Học tương phản (contrastive, InfoNCE):** cho một câu, mô hình phải cho ảnh đúng điểm cao hơn các ảnh sai. Loss là cross-entropy trên danh sách {ảnh đúng, các ảnh sai}.
- **Hard negative:** ảnh sai nhưng "trông giống" đúng. Ở đây là các ảnh sai mà CLIP xếp cao. Đây chính là lỗi mà bước xếp hạng lại cần sửa.
- **Inductive:** mô hình học trên một đồ thị (chỉ train), rồi áp dụng cho đỉnh mới (val/test) mà không học lại.
- **Tách được đóng góp (decomposable):** điểm là **tổng** của các phần, mỗi phần ứng với một cạnh. Giải thích là chính các phần đó, nên trung thực theo cấu tạo (ADR-0016).

Dữ liệu huấn luyện đã có sẵn, không cần gán nhãn thêm: mỗi ảnh **train** có 5 region description đã mã hoá bằng CLIP (ADR-0041), tức khoảng 500 nghìn cặp (câu, ảnh). Kiểm tra mô hình có học được hay không bằng giao thức ADR-0043 trên **val**; test chạy một lần ở cuối.

## Quyết định

### 1. Cạnh `contains` lấy từ CLIP top-k, cho mọi split

| Thành phần | Quy định |
|---|---|
| Đỉnh khái niệm | Giữ từ vựng của ADR-0045 (synset gặp ở ≥ 5 ảnh **train**, 3.880 khái niệm), vector là CLIP text của `"a photo of a {lemma}"` |
| Cạnh của mỗi ảnh | `k` khái niệm có `cos(vector ảnh CLIP, vector khái niệm)` cao nhất. Áp dụng **như nhau** cho train, val, test |
| `k` | Đề xuất `k = 16`, gần bậc trung bình của đồ thị annotation (1.719.645 cạnh / 108.077 ảnh ≈ 15,9), để ablation so sánh cùng số cạnh. Chỉ chỉnh trên **val** |
| Cạnh từ annotation | **Không** dùng cho mô hình chính. Chỉ dùng làm ablation "cận trên" trên val/test, ghi rõ trong bảng là có dùng annotation |

Lý do: cạnh của mỗi ảnh chỉ phụ thuộc vào vector CLIP của chính ảnh đó, không dùng nhãn, không đếm thống kê trên test. Vì vậy hệ thống chạy được với một ảnh mới chưa có annotation, và không đọc đáp án (giải quyết A2). Train và test có cùng một loại cạnh, nên không bị lệch phân bố.

### 2. Mô hình: một lớp GraphSAGE (mean), chiều khái niệm → ảnh

Mỗi ảnh `i` có `k` khái niệm hàng xóm `N(i)`. Vector mới của ảnh:

```
h_i = W_self · x_i + b + (1/k) · Σ_{c ∈ N(i)} W_nb · x_c
graph_score(q, i) = cos(q, h_i)
```

- `x_i`: vector CLIP của ảnh; `x_c`: vector CLIP của khái niệm; `q`: vector CLIP của câu truy vấn. Cả ba **đóng băng**, chỉ `W_self`, `W_nb`, `b` được học.
- Đây đúng là `SAGEConv` của PyG với `aggr="mean"`, đồ thị hai phía (khái niệm → ảnh), **không** có hàm kích hoạt.
- Khởi tạo `W_self = I`, `W_nb = 0`, `b = 0`: lúc đầu `graph_score` bằng đúng cosine của CLIP. Mô hình xuất phát từ baseline, và chỉ lệch khỏi nó khi việc đó giảm loss.
- Chỉ một lớp, một chiều. Vì vậy điểm của một ảnh chỉ phụ thuộc vào chính nó và các khái niệm của nó; ảnh val/test không ảnh hưởng tới nhau hay tới ảnh train.

### 3. Huấn luyện

| Thành phần | Quy định |
|---|---|
| Đồ thị lúc huấn luyện | Chỉ đỉnh ảnh **train** + đỉnh khái niệm, chỉ cạnh của ảnh train (giải quyết A1) |
| Mẫu | Mỗi region description `t` của một ảnh train |
| Ứng viên của `t` | Ảnh gốc của `t` (đúng) + `n_hard` ảnh sai lấy mẫu từ top 100 CLIP của `t`, **chỉ tìm trong kho ảnh train**. Ảnh có cùng câu (cosine văn bản ≥ 0,9999, ADR-0043) không được làm ảnh sai |
| Loss | Cross-entropy trên `cos(t, h)/τ` của các ứng viên (InfoNCE, listwise). Học đúng việc xếp hạng lại top 100 của CLIP |
| Đề xuất ban đầu | `n_hard = 15`, `τ = 0.05`, Adam `lr = 1e-3`, số epoch chọn bằng early stopping trên val |
| Chọn siêu tham số | `alpha`, `k`, `n_hard`, `τ`, `lr`, số epoch: **chỉ trên val** theo giao thức ADR-0043. Không dùng `queries/` |
| Seed | `[0, 1, 2]`, báo cáo mean ± std (ADR-0011) |

### 4. Suy luận và xếp hạng lại

1. CLIP + FAISS lấy 100 ứng viên (cùng hàm lấy ứng viên với baseline ADR-0043).
2. Gắn đỉnh ảnh val/test vào đồ thị bằng cạnh CLIP top-k của chính chúng; tính `h_i` bằng trọng số đã học, **không** học lại.
3. `final = alpha · cos(q, x_i) + (1 − alpha) · cos(q, h_i)`. Hai điểm đều là cosine nên cùng thang đo.

### 5. Giải thích (ADR-0016)

Vì `h_i` là tổng, `graph_score` tách chính xác thành:

```
graph_score(q, i) = [q · (W_self x_i + b)] / (‖q‖‖h_i‖)            # phần của chính ảnh
                  + Σ_{c ∈ N(i)} [q · W_nb x_c / k] / (‖q‖‖h_i‖)    # phần của từng cạnh i–c
```

Scorer trả về điểm kèm danh sách `(khái niệm c, đóng góp)`. `explain/` chỉ đọc danh sách này. Giải thích hiển thị là "ảnh được đẩy lên nhờ cạnh tới *dog*, *fire hydrant*". Fidelity đo bằng can thiệp thật: xoá các cạnh có đóng góp lớn nhất, tính lại `h_i` và thứ hạng, so với xoá cùng số cạnh ngẫu nhiên.

### 6. So sánh bắt buộc

Cùng split, cùng 100 ứng viên, 3 seed, báo cáo tách direct/indirect trên `queries/`:

| Mô hình | Trả lời câu hỏi |
|---|---|
| CLIP thuần | Baseline (đã có, `results/tables/main.csv`) |
| **Đối chứng không đồ thị**: chỉ `W_self`, bỏ `W_nb`, huấn luyện y hệt | Phần tăng có đến từ đồ thị không, hay chỉ từ một phép chiếu tuyến tính học trên VG? |
| Không huấn luyện: `W_self = I`, `W_nb = I` | Đồ thị có giúp ngay cả khi không học không? |
| Xáo cạnh giữ bậc, khái niệm ngẫu nhiên | Ablation đồ thị (PDF §7) |
| Cạnh annotation trên val/test (cận trên, có ghi chú) | Cạnh tốt hơn giúp được bao nhiêu; ADR-0045 yêu cầu |

Quyết định bởi: qphulong, 2026-11-10 (huấn luyện trực tiếp cho truy vấn ảnh; cạnh CLIP top-k cho mọi split; GNN một lớp tách được đóng góp). Các giá trị còn lại (`k = 16`, `n_hard = 15`, `τ = 0.05`, `lr = 1e-3`, cách khởi tạo) do Claude Code đề xuất, **nhóm chưa duyệt**.

## Các phương án đã cân nhắc

| Phương án | Ưu | Nhược | Lý do không chọn |
|---|---|---|---|
| Huấn luyện trực tiếp cho truy vấn ảnh (contrastive, hard negative từ CLIP top-k) | Mục tiêu chính là việc xếp hạng lại, nên cải thiện thể hiện thẳng ở Recall/MRR; dữ liệu train có sẵn | Phải tự thiết kế để giải thích được | **Được chọn** |
| Link prediction trên cạnh `contains` (che cạnh rồi đoán lại) | Giải thích theo đường đi tự nhiên | Mục tiêu học không phải là truy vấn, cải thiện có thể nhỏ; vẫn cần cùng benchmark ADR-0043 để chứng minh | Mục tiêu học lệch với việc cần làm |
| Cạnh annotation cho mọi split (đồ thị ADR-0045 nguyên trạng) | Cạnh sạch nhất, không phải làm thêm | Ảnh test có cạnh lấy từ đáp án (A2), kết quả lạc quan theo cấu tạo | Chỉ giữ làm ablation cận trên |
| Annotation cho train, CLIP top-k cho val/test | Huấn luyện trên cạnh sạch | Học trên một loại cạnh, kiểm tra trên loại khác (lệch phân bố) | Kết quả khó diễn giải |
| Ảnh val/test không có cạnh | Không rò rỉ | Điểm đồ thị của ảnh test chỉ còn là phép chiếu của vector CLIP; đồ thị không đóng góp gì lúc kiểm tra | Mất ý nghĩa của đồ thị |
| GraphSAGE/GAT hai lớp + attribution (masking, gradient) | Biểu diễn mạnh hơn, ảnh "nghe" được ảnh khác qua khái niệm chung | Đóng góp chỉ là xấp xỉ, phải chứng minh trung thực bằng fidelity; ảnh test ảnh hưởng lẫn nhau | Để làm mở rộng sau, so sánh với bản một lớp |
| Dùng attention của GAT làm giải thích | Có sẵn | Attention không chắc phản ánh tầm quan trọng (ADR-0016) | Bị ADR-0016 loại |
| Loss chỉ dùng ảnh sai trong batch (in-batch negatives) | Đơn giản, nhanh | Ảnh sai ngẫu nhiên dễ phân biệt, không giống tình huống xếp hạng lại top 100 | Hard negative từ CLIP sát với bài toán hơn; có thể thêm in-batch sau nếu val cho thấy có ích |

## Hệ quả

- **Rò rỉ dữ liệu (PDF §6, ADR-0009):**
  - Huấn luyện chỉ thấy ảnh train; hard negative chỉ tìm trong kho ảnh train. Cần test chứng minh `edge_index` lúc huấn luyện không chứa đỉnh val/test.
  - Cạnh CLIP top-k của một ảnh chỉ dùng vector của chính ảnh đó và CLIP đóng băng, không dùng nhãn hay thống kê trên test. Từ vựng khái niệm vẫn đếm trên train (ADR-0045).
  - Mọi siêu tham số chọn trên val. `queries/` chỉ để đánh giá.
- **Giải thích:** trung thực theo cấu tạo, vì giải thích là chính các số hạng của điểm. Cần test chứng minh tổng các đóng góp bằng `graph_score`, và xoá cạnh có đóng góp dương làm điểm giảm.
- **Hạn chế cần nói rõ khi báo cáo và vấn đáp:**
  - Cạnh và vector khái niệm đều đến từ CLIP, nên đồ thị không mang thông tin "mới" ngoài CLIP; nó chỉ tổ chức lại thông tin đó theo khái niệm. Lợi ích có thể nhỏ. Đối chứng không đồ thị cho biết đồ thị có thật sự giúp hay không, và kết quả âm cũng phải báo cáo.
  - Một lớp nên không có đường đi nhiều bước (ảnh – khái niệm – ảnh). Truy vấn gián tiếp chỉ được hỗ trợ qua vector khái niệm; kết quả nhóm indirect có thể không tốt hơn direct.
- **Ảnh hưởng tới code của ADR-0045 (PR #6):** builder cần tuỳ chọn nguồn cạnh `edge_source: clip_topk | annotations` ghi trong config và `meta.json`, và lưu split của từng đỉnh ảnh. Hai việc này trùng với điểm A1, A2 của review.
- **Việc tiếp theo, mỗi việc một PR nhỏ có test:**
  1. Hàm dựng cạnh CLIP top-k + lưu split của từng ảnh.
  2. Hàm lấy đồ thị con chỉ gồm ảnh train.
  3. Mô hình một lớp + hàm tách đóng góp.
  4. Vòng huấn luyện (hard negative từ CLIP top-k trên train), early stopping trên val.
  5. Xếp hạng lại + chọn `alpha` trên val.
  6. Fidelity và các ablation ở mục 6.
- Nếu sau này cần hai lớp, viết ADR mới, giữ bản một lớp làm đối chứng.

## Nguồn tham khảo

- Đề bài, §2, §4.2, §5, §6, §7.
- ADR-0009, ADR-0011, ADR-0016, ADR-0041, ADR-0043, ADR-0044, ADR-0045.
- Review PR #6 (điểm A1, A2): https://github.com/FluteInTheRain/graph-semantic-image-editing/pull/6#issuecomment-6730701603
- Tài liệu PyG về `SAGEConv` trên đồ thị hai phía (`in_channels` dạng tuple) và tham số `aggr`: https://github.com/pyg-team/pytorch_geometric/blob/master/docs/source/cheatsheet/gnn_cheatsheet.rst
