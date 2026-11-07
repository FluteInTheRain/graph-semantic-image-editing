# Đề xuất phương pháp cho module xAI (Nhiệm vụ 3)

- **Trạng thái:** Bản nháp nghiên cứu, **chưa phải quyết định**. Sẽ chốt bằng ADR sau khi trả lời các câu hỏi ở mục 8.
- **Ngày:** 2026-11-07
- **Soạn bởi:** Claude Code (AI hỗ trợ). Thành viên phải tự đọc lại các bài báo trước khi trích dẫn.
- **Phạm vi:** chỉ module xAI (`src/mmxai/explain/`). Module Graph (dựng đồ thị, GNN, scorer trong `models/`) do phần khác của nhóm làm. Tài liệu này **không đề xuất hay sửa thiết kế của module Graph**.
- **Tuân theo ADR-0016** (đã chấp nhận): scorer của module Graph trả về điểm kèm bản ghi đóng góp; `explain/` chỉ đọc bản ghi đó, không tự tính hay tự tìm giải thích.

Nguyên tắc: **áp dụng phương pháp đã có**, không phát minh mới. Với mỗi phương pháp phải trả lời được: **làm gì, vì sao chọn, input, output, luồng xử lý**.

> 📣 **Cần báo nhóm Graph:** xem [mục 7](#7-lưu-ý-gửi-nhóm-graph).

## 1. Đề bài yêu cầu gì ở phần xAI

Nguồn: `Project03_Multimodal_XAI.pdf`, ADR-0016, `.claude/rules/explanations.md`.

**Quy tắc xuyên suốt (PDF §2):** giải thích phải đến từ **chính cơ chế xếp hạng**. Không được sinh ra một đường đi "nghe hợp lý" sau khi đã có kết quả nếu đường đi đó không ảnh hưởng tới điểm.

| Cấp độ | Yêu cầu xAI |
|---|---|
| 1 (trần 7.0) | Hiển thị quan hệ/đường đi **đã dùng khi xếp hạng** cho một số truy vấn; phân tích 5 truy vấn thành công + 5 truy vấn thất bại |
| 3 (trần 10) | Đánh giá **định lượng** fidelity, validity, sparsity trên **≥ 10 truy vấn** |
| 4 (điểm cộng) | Tùy chọn: so sánh giải thích giữa các mô hình; khảo sát người dùng có đối chứng |

Demo (PDF §8) phải hiển thị được quan hệ/đường đi giải thích kèm top-k kết quả. Đề chấp nhận giải thích dạng **quan hệ, đường đi hoặc đồ thị con** (PDF §1).

## 2. Phân công theo ADR-0016

| Việc | Module chịu trách nhiệm | Căn cứ |
|---|---|---|
| Tính điểm đồ thị **và** bản ghi đóng góp (cạnh/path nào góp bao nhiêu) | **Graph** (`models/`) | ADR-0016 mục 1 |
| Truyền nguyên bản ghi đóng góp tới kết quả cuối, `final = α·cosine + (1−α)·graph` | Rerank (`rerank/`) | ADR-0016 mục 2 |
| Đọc bản ghi, trình bày giải thích; can thiệp lên đồ thị; đo fidelity/validity/sparsity; so sánh với explainer post-hoc | **xAI** (`explain/`) | ADR-0016 mục 3–6 |

Module xAI **không quyết định** đóng góp được tính thế nào. Việc nghiên cứu của module xAI tập trung vào **cách trình bày và cách kiểm chứng** giải thích.

## 3. Bốn phần việc của module xAI

| # | Phần việc | Phương pháp có sẵn để áp dụng | Căn cứ ADR-0016 |
|---|---|---|---|
| 1 | Đọc và trình bày giải thích | Lấy top-m cạnh/path theo đóng góp từ bản ghi của scorer; hiển thị dạng đường đi hoặc đồ thị con | Mục 3 |
| 2 | Hàm can thiệp | Xoá hoặc làm nhiễu cạnh trên `edge_index` rồi gọi lại scorer. Dùng chung cho fidelity và ablation đồ thị | Mục 4, 6 |
| 3 | **Đánh giá giải thích** (phần nghiên cứu chính) | Fid+/Fid−, characterization (GraphFramEx, LoG 2022); robust fidelity (Zheng et al., ICLR 2024); sparsity; validity; đối chứng xoá ngẫu nhiên; kiểm định Wilcoxon | Mục 4, 5 |
| 4 | So sánh với explainer post-hoc (**chỉ làm baseline**) | Occlusion, Integrated Gradients/Saliency, GNNExplainer; attention nếu mô hình là GAT | Bảng phương án: GNNExplainer "dùng làm phương án so sánh" |

Chi tiết từng phần ở mục 4.

## 4. Chi tiết từng phần

### 4.1. Phần 1 – Đọc và trình bày giải thích

- **Làm gì:** chọn các cạnh/path có đóng góp lớn nhất từ bản ghi của scorer, đổi id thành tên node và tên quan hệ, xuất ra dạng đọc được.
- **Vì sao:** ADR-0016 cấm `explain/` tự tìm đường đi (ví dụ `networkx.shortest_path`). Lấy đúng các số hạng scorer đã dùng là cách duy nhất bảo đảm giải thích đến từ cơ chế xếp hạng (PDF §2).
- **Input:** kết quả rerank cho (q, i) gồm `cosine`, `graph`, `final`, `contributions`; ánh xạ id → tên node/quan hệ.
- **Output:** đồ thị con giải thích gồm top-m cạnh/path kèm đóng góp, dùng cho demo và phân tích 5 thành công + 5 thất bại.
- **Luồng:**
  ```
  contributions của (q, i)
   → kiểm: tổng đóng góp ≈ graph(q, i)
   → sắp xếp theo đóng góp, lấy top-m      (m chọn trên val, ghi vào config)
   → đổi id → tên
   → xuất bảng / hình đồ thị con
  ```

### 4.2. Phần 2 – Hàm can thiệp

- **Làm gì:** từ đồ thị gốc tạo một đồ thị đã xoá (hoặc làm nhiễu) một tập cạnh cho trước, rồi gọi lại scorer để lấy điểm mới.
- **Vì sao:** fidelity theo đề là "bỏ hoặc làm nhiễu cạnh thì điểm/thứ hạng phải đổi", tức là phải **can thiệp thật**. ADR-0016 mục 6 yêu cầu ablation (bỏ cạnh, xáo cạnh, đồ thị ngẫu nhiên) dùng chung hàm này, để hai kết quả so sánh được với nhau.
- **Input:** `edge_index` (và loại cạnh), tập cạnh cần xoá/nhiễu, scorer (gọi như hộp đen).
- **Output:** điểm và thứ hạng mới của các ứng viên.
- **Luồng:**
  ```
  edge_index, tập cạnh E
   → tạo edge_index' = edge_index ∖ E        (không sửa đồ thị gốc)
   → gọi scorer(q, ứng viên, edge_index')
   → tính lại final với α đã chốt → thứ hạng mới
  ```
- **Lưu ý:** khi xoá cạnh, chỉ trừ đi số hạng đóng góp là **không đủ**. Với GCN, chuẩn hoá theo bậc làm trọng số các cạnh còn lại đổi theo, nên phải gọi lại scorer.

### 4.3. Phần 3 – Đánh giá giải thích (phần nghiên cứu chính)

- **Làm gì:** đo xem giải thích có trung thực, hợp lệ và gọn không, trên ≥ 10 truy vấn × 3 seed.
- **Vì sao chọn các thước đo dưới đây:**
  - Đề không định nghĩa chi tiết validity và sparsity, nên nhóm lấy định nghĩa từ các khung đánh giá đã công bố thay vì tự chế.
  - GraphFramEx tách hai khía cạnh: giải thích **cần thiết** (bỏ đi thì điểm sụp, Fid+) và **đủ** (chỉ giữ nó thì điểm vẫn còn, Fid−).
  - Robust Fidelity xử lý vấn đề xoá cạnh tạo ra đồ thị lệch khỏi phân phối lúc huấn luyện (OOD).
- **Input:** bản ghi đóng góp, hàm can thiệp (phần 2), danh sách truy vấn đánh giá.
- **Output:** bảng metric theo truy vấn, có tách nhóm trực tiếp/gián tiếp, kèm đối chứng ngẫu nhiên và p-value.

Ký hiệu: `i*` là ảnh đang giải thích, `E_x` là tập cạnh giải thích, `S(q,i*)` là đồ thị con k-hop quanh `i*`, `g` là điểm đồ thị.

| Metric | Công thức / cách đo | Ý nghĩa |
|---|---|---|
| Fid+ (điểm) | `[g(q,i*) − g_{G∖E_x}(q,i*)] / g(q,i*)` | Bỏ cạnh giải thích thì điểm giảm bao nhiêu |
| Fid+ (thứ hạng) | `rank_{G∖E_x}(i*) − rank_G(i*)`, tính trên `final` với α đã chốt | Bỏ cạnh giải thích thì ảnh tụt bao nhiêu hạng |
| Fid− | `[g(q,i*) − g_{E_x}(q,i*)] / g(q,i*)` | Chỉ giữ cạnh giải thích thì còn bao nhiêu điểm (≈ 0 là tốt) |
| Characterization | `Fid+·(1−Fid−) / (0.5·(1−Fid−) + 0.5·Fid+)`, sau khi clip Fid về [0,1] | Gộp Fid+ và Fid− thành một số |
| Robust Fid+ | Như Fid+, nhưng mỗi cạnh của `E_x` chỉ bị xoá với xác suất 0.9 (tức được giữ với xác suất 0.1), lấy trung bình nhiều lần | Giảm ảnh hưởng của OOD |
| Sparsity | `1 − |E_x| / |E(S(q,i*))|` | Giải thích càng gọn càng tốt |
| Validity | Tỉ lệ cạnh giải thích có thật trong đồ thị train **và** nằm trong `S(q,i*)`, nối được tới `i*`. `networkx` chỉ được dùng ở bước kiểm này (ADR-0016) | Không có cạnh "ảo" |

**Đối chứng:** xoá cùng `|E_x|` cạnh chọn ngẫu nhiên **trong `S(q,i*)`** (không lấy từ toàn đồ thị), lặp R ≥ 20 lần. Kiểm định bằng Wilcoxon signed-rank ghép cặp theo truy vấn.

- **Luồng:**
  ```
  với mỗi truy vấn q (≥ 10) và mỗi seed (3):
    với ảnh i* cần giải thích:
      E_x ← top-m từ phần 1
      Fid+, Fid−, robust Fid+  ← dùng hàm can thiệp (phần 2)
      đối chứng: R lần xoá |E_x| cạnh ngẫu nhiên trong S(q,i*)
      sparsity, validity
  → gộp: mean ± std (ddof=1), Wilcoxon so với đối chứng, tách trực tiếp/gián tiếp
  ```

**Bẫy cần ghi trong báo cáo:**
1. **OOD:** xoá cạnh tạo ra đồ thị khác lúc huấn luyện, nhất là với GCN (chuẩn hoá theo bậc) và GAT (softmax attention dồn sang cạnh khác). Vì vậy báo cáo cả Fid+ thường lẫn robust Fid+.
2. **Nếu scorer tuyến tính thì Fid+ cao là hiển nhiên.** Khi đó phần đáng phân tích là Fid−, và những path có đóng góp lớn nhưng xoá đi không làm tụt hạng.

### 4.4. Phần 4 – So sánh với explainer post-hoc (baseline)

- **Làm gì:** chạy các explainer post-hoc trên **cùng mô hình**, lấy cùng số cạnh m, rồi chấm bằng cùng metric ở phần 3.
- **Vì sao:** để trả lời bằng số câu hỏi "giải thích lấy từ scorer có trung thực hơn cách tính sau hay không". Bảng phương án của ADR-0016 cho phép dùng các phương pháp này làm phương án so sánh, không làm phương pháp chính.

| Baseline | Cách làm | Ghi chú |
|---|---|---|
| Ngẫu nhiên | Chọn m cạnh ngẫu nhiên trong `S(q,i*)` | Mốc thấp nhất |
| Occlusion | Xoá từng cạnh, đo điểm giảm, lấy top-m | Can thiệp thật; dùng lại hàm can thiệp ở phần 2 |
| Integrated Gradients / Saliency (ICML 2017) | Gradient theo mask trên cạnh | PyG `CaptumExplainer`; cần kiểm API qua context7 |
| GNNExplainer (NeurIPS 2019) | Tối ưu một mask cạnh riêng cho từng mẫu | Có sẵn trong `torch_geometric.explain` |
| Attention của GAT | Đọc trọng số attention | Chỉ dùng nếu mô hình là GAT |

## 5. Câu hỏi nghiên cứu và giả thuyết

| RQ | Câu hỏi | Giả thuyết (bác bỏ được) |
|---|---|---|
| RQ1 | Giải thích có thật sự ảnh hưởng tới xếp hạng không? | Fid+ (thứ hạng) của `E_x` lớn hơn xoá ngẫu nhiên cùng kích thước (Wilcoxon, p < 0.05) |
| RQ2 | Giải thích lấy từ scorer có trung thực hơn explainer post-hoc không? | Ở cùng sparsity (top-m cạnh), characterization của giải thích từ scorer ≥ Occlusion, IG và GNNExplainer |
| RQ3 | Đóng góp theo bản ghi có khớp với can thiệp thật không? | Thứ tự cạnh theo đóng góp tương quan Spearman ≥ 0.5 với Occlusion |
| RQ4 | Truy vấn gián tiếp có khó giải thích hơn không? | Với truy vấn gián tiếp, khoảng cách giữa Fid+ thường và robust Fid+ lớn hơn truy vấn trực tiếp |

Nếu mô hình là GAT, thêm RQ5: attention có khớp với Occlusion không (Jain & Wallace 2019 dự đoán là không). RQ2 đáp ứng mục mở rộng "so sánh giải thích" của đề (PDF §7).

## 6. Interface module xAI cần nhận

**Interface** ở đây là hợp đồng giữa hai module: module Graph đưa cho module xAI dữ liệu gì, ở dạng nào, và gọi hàm nào. Không phải giao diện người dùng (UI). Đây chỉ là **yêu cầu interface**, không phải thiết kế mô hình.

| Cần nhận | Từ module | Vì sao xAI cần |
|---|---|---|
| Kết quả mỗi ứng viên: `cosine`, `graph`, `final`, `contributions` | Graph → Rerank | Phần 1; ADR-0016 (mục Hệ quả) đã nêu kiểu dữ liệu này |
| `contributions`: danh sách cạnh/path (id node, loại quan hệ) kèm phần điểm đóng góp | Graph | Phần 1, 3 |
| Scorer gọi lại được trên một `edge_index` đã sửa | Graph | Phần 2, 3. Cần **mô hình chạy lại được**, không chỉ embedding đã cache |
| `x`, `edge_index`, loại cạnh, ánh xạ id ↔ tên | Graph | Hiển thị, validity, đối chứng ngẫu nhiên |
| `α` đã chốt trên val | Rerank | Tính thứ hạng sau can thiệp |

Trong lúc module Graph chưa có code, module xAI viết và test trên một **scorer giả cỡ nhỏ trong `tests/`**, có trả về bản ghi đóng góp. Scorer giả này không nằm trong `models/`.

## 7. Lưu ý gửi nhóm Graph

> **Nếu nhóm Graph dùng GCN/GAT/GraphSAGE thông thường, có hàm kích hoạt phi tuyến, scorer không tự phân rã được thành đóng góp. Khi đó họ cần một cách để đáp ứng ADR-0016.**

Giải thích thêm:

- **Vấn đề:** ADR-0016 (đã chấp nhận) yêu cầu scorer trả về điểm **kèm** bản ghi đóng góp, và đóng góp đó phải là cái thật sự tạo ra điểm. Với GNN có hàm kích hoạt phi tuyến (ReLU…) giữa các lớp, điểm cuối không phải là tổng các phần riêng của từng cạnh/path. Muốn có bản ghi đóng góp thì phải có thêm một cơ chế, và cơ chế đó cần được chọn **ngay từ thiết kế scorer**.
- **Ai quyết:** nhóm Graph, qua một ADR của họ. Module xAI chỉ đưa **gợi ý tham khảo** (mục 7.1); quyết định cuối thuộc nhóm Graph. Mục 9.3 có thêm tài liệu về các mô hình có giải thích theo đường đi.
- **Ảnh hưởng nếu không xử lý:** module xAI không có gì để đọc. Giải thích sẽ phải tính bằng explainer post-hoc, tức là trái ADR-0016 và trái yêu cầu PDF §2.
- **Hiện trạng (2026-11-07):** chưa có ADR nào chọn GNN. `configs/clip_graph_vg.yaml` ghi `gnn: gcn`, nhưng `configs/README.md` nói đây là ví dụ mẫu chưa dùng bởi code. `src/mmxai/models/` chưa có code.
- **Cần nhóm Graph trả lời:**
  1. Dùng GNN nào, và bao giờ có ADR?
  2. Đáp ứng ADR-0016 bằng cách nào?
  3. Có đồng ý interface ở mục 6 không?

### 7.1. Gợi ý cách xử lý (để nhóm Graph tham khảo)

Có ba hướng, xếp theo mức độ khớp với ADR-0016. Cả ba đều dùng phương pháp đã công bố, không phải phát minh mới.

#### Hướng A (khuyến nghị): dùng GCN tuyến tính – SGC hoặc LightGCN

- **Làm gì:** bỏ hàm kích hoạt phi tuyến giữa các lớp GCN. SGC (Wu et al., ICML 2019) tính `H = S^K · X · Θ`, trong đó `S` là ma trận kề đã chuẩn hoá và `K` là số hop. LightGCN (He et al., SIGIR 2020) cũng lan truyền tuyến tính, rồi lấy tổng embedding của các lớp.
- **Vì sao điểm phân rã được:** giả sử điểm đồ thị là tích vô hướng giữa truy vấn đã chiếu `qW` và embedding ảnh `h_i`. Khi đó:
  ```
  g(q, i) = ⟨qW, h_i⟩ = Σ_j (S^K)_ij · ⟨qW, x_j Θ⟩
                         └──── đóng góp của node j vào điểm của ảnh i ────┘
  (S^K)_ij = tổng trên các walk dài K từ i tới j, mỗi walk là tích trọng số các cạnh
           → tách tiếp được thành đóng góp của từng walk (đường đi)
  ```
  Tổng các đóng góp **bằng đúng** điểm, nên bản ghi đóng góp là chính các số hạng trong tổng. Không cần ước lượng.
- **Input / output cho scorer:** input là `x`, `edge_index`, vector truy vấn; output là `g(q, i)` kèm danh sách (walk hoặc node, đóng góp), đúng interface ở mục 6.
- **Ưu:** vẫn thuộc họ GCN. Faithful by construction, đúng ADR-0016. Code ngắn, dễ giải thích từng dòng khi vấn đáp. Test được bằng một assert: tổng đóng góp bằng điểm.
- **Nhược và rủi ro:**
  - Mô hình có thể yếu hơn GCN có ReLU. Có thể đo mức chênh bằng cách chạy thêm GCN có ReLU làm biến thể so sánh.
  - Số walk tăng nhanh theo `K`. Nên giữ `K = 2` hoặc chỉ ghi lại top walk.
  - **Cần hỏi giảng viên** SGC/LightGCN có được tính là "GCN" cho yêu cầu cấp độ 1 không.

#### Hướng B: giữ GNN phi tuyến, nhưng làm bước tính điểm cuối thành tổng theo cạnh

- **Làm gì:** GNN (GCN/GAT/SAGE, có ReLU) chỉ dùng để tạo embedding node. Điểm đồ thị được tính bằng một bước đọc ra dạng tổng, ví dụ `g(q, i) = Σ_{e ∈ S(i)} s(q, e)`, trong đó `s(q, e)` là điểm của cạnh `e` tính từ embedding hai đầu cạnh và truy vấn.
- **Ưu:** giữ được GNN thông thường. Tổng các số hạng vẫn bằng đúng điểm.
- **Nhược:** đóng góp chỉ chính xác **ở bước đọc ra**. Mỗi `s(q, e)` vẫn phụ thuộc embedding đã trộn thông tin từ cả vùng lân cận, nên xoá một cạnh có thể làm đổi số hạng của cạnh khác. Báo cáo phải nói rõ giới hạn này, và fidelity sẽ cho thấy mức ảnh hưởng.

#### Hướng C (không khuyến nghị): giữ GNN phi tuyến, tính đóng góp bằng attribution

- **Làm gì:** scorer tự gọi Integrated Gradients (ICML 2017) trên mask cạnh, lấy kết quả làm bản ghi đóng góp. IG có tính completeness: tổng attribution bằng `g(đủ cạnh) − g(không cạnh)`.
- **Vì sao không khuyến nghị:** đây là phương pháp tính **sau khi có điểm**. ADR-0016 xếp nhóm phương pháp này là "chỉ dùng để so sánh". Muốn dùng làm chính phải viết **ADR mới thay thế phần đó của ADR-0016**, và phải chứng minh độ trung thực bằng số fidelity. Hướng này cũng làm mất baseline IG ở phần 4 của module xAI.

#### Tóm tắt

| Hướng | Đóng góp chính xác? | Đúng ADR-0016? | Giữ GNN phi tuyến? | Chi phí |
|---|---|---|---|---|
| **A. SGC / LightGCN** | Có, toàn bộ | Có | Không | Thấp |
| B. Tính điểm cuối dạng tổng theo cạnh | Chỉ ở bước đọc ra | Có, kèm ghi chú giới hạn | Có | Trung bình |
| C. IG làm bản ghi đóng góp | Xấp xỉ | Không, cần ADR mới | Có | Cao (20–50 lần lan truyền ngược mỗi ứng viên) |

## 8. Câu hỏi còn mở (phía xAI)

1. **m** (số cạnh/path trong một giải thích): chọn trên **val**, không chọn trên bộ truy vấn đánh giá.
2. **Dạng hiển thị:** đường đi hay đồ thị con. Phụ thuộc vào việc bản ghi đóng góp tính theo cạnh hay theo path (mục 7).
3. **Số lần lặp R** của đối chứng và xác suất giữ cạnh của robust fidelity: chốt trước, ghi vào config.
4. **Truy vấn đánh giá:** lấy ≥ 10 truy vấn từ bộ benchmark (nhánh `feat/vg-query-benchmark`, ADR-0044), có cả nhóm trực tiếp lẫn gián tiếp.

## 9. Tóm tắt khảo sát tài liệu

Subagent `paper-researcher` khảo sát qua web vì MCP `arxiv` không kết nối được. Chỗ đánh dấu **[chưa xác minh]** là subagent chưa đọc được nguồn gốc.

### 9.1. Explainer cho GNN (baseline của phần 4)

- **GNNExplainer** (NeurIPS 2019): học mask cạnh cho từng mẫu; có trong `torch_geometric.explain`.
- **PGExplainer** (NeurIPS 2020): mask được tham số hoá bằng mạng nhỏ, dùng lại được cho mẫu mới.
- **SubgraphX** (ICML 2021): tìm đồ thị con bằng MCTS + Shapley; chậm; code trong DIG (GPL-3.0).
- **Integrated Gradients** (ICML 2017): có tính completeness (tổng attribution = chênh lệch điểm); dùng trong PyG qua `CaptumExplainer`.
- **Attention làm giải thích:** Jain & Wallace (NAACL 2019) cho rằng không phải giải thích; Wiegreffe & Pinter (EMNLP 2019) phản biện, đề nghị kiểm bằng can thiệp. Với GNN: GAtt (2024, venue [chưa xác minh]), FGAI (TMLR 2025).

### 9.2. Đánh giá giải thích (phần 3)

- **GraphFramEx** (LoG 2022): giải thích sufficient (Fid− ≈ 0) và necessary (Fid+ ≈ 1); characterization score; khuyến nghị so sánh các giải thích có cùng kích thước.
- **Robust Fidelity** (ICLR 2024): chỉ xoá ngẫu nhiên một phần cạnh giải thích để giảm OOD; không phạt kích thước nên vẫn cần báo cáo sparsity.
- **GInX-Eval** (NeurIPS 2023 workshop): xoá cạnh rồi huấn luyện lại; nhiều explainer không tốt hơn chọn ngẫu nhiên.
- **GraphXAI** (*Scientific Data* 2023): các thước đo GEA, GEF/unfaithfulness, stability.
- PyG có `fidelity` và `characterization_score`, nhưng viết cho **phân loại**. Bài của nhóm là **xếp hạng**, nên cần bản liên tục như ở mục 4.3.

### 9.3. Mô hình có giải thích theo đường đi (thuộc phạm vi module Graph, chỉ để tham khảo)

- NBFNet (NeurIPS 2021), ULTRA (ICLR 2024): giải thích theo path là **xấp xỉ bằng gradient** (tác giả NBFNet ghi "better approximation is left as future work").
- Path Ranking Algorithm (Lao & Cohen 2010 [chưa đọc lại]): điểm là tổ hợp tuyến tính theo loại path, nên phân rã được chính xác.
- KGAT (KDD 2019): attention trên cạnh KG, nhưng điểm đi qua lớp phi tuyến. KPRN (AAAI 2019): trọng số theo path qua pooling. PathCon (KDD 2021).
- Truy vấn cross-modal có giải thích, 2022–2026: không tìm thấy bài làm đúng chuỗi "CLIP top-k → xếp hạng lại bằng scene graph → giải thích faithful". Các bài gần nhất (chưa đọc toàn văn): Kim & Ko (*Sensors* 2025), QCRS (*Applied Sciences* 2026), *Scientific Reports* 2026, Wang et al. (WACV 2020).

## Tài liệu tham khảo

- GraphFramEx: https://arxiv.org/abs/2206.09677 · https://github.com/GraphFramEx/graphframex
- Robust Fidelity: https://arxiv.org/abs/2310.01820
- GInX-Eval: https://arxiv.org/abs/2309.16223
- GraphXAI: https://arxiv.org/abs/2208.09339 · https://github.com/mims-harvard/GraphXAI
- Integrated Gradients: https://arxiv.org/abs/1703.01365
- GNNExplainer: https://arxiv.org/abs/1903.03894
- PGExplainer: https://arxiv.org/abs/2011.04573
- SubgraphX: https://arxiv.org/abs/2102.05152 · https://github.com/divelab/DIG (GPL-3.0)
- Jain & Wallace: https://arxiv.org/abs/1902.10186
- Wiegreffe & Pinter: https://arxiv.org/abs/1908.04626
- GAtt: https://arxiv.org/abs/2406.04612
- FGAI: https://arxiv.org/abs/2410.06950
- SGC (Wu et al., ICML 2019): https://arxiv.org/abs/1902.07153 (do AI đề xuất, không nằm trong bản khảo sát; cần tự đọc lại)
- LightGCN (He et al., SIGIR 2020): https://arxiv.org/abs/2002.02126 (do AI đề xuất, không nằm trong bản khảo sát; cần tự đọc lại)
- NBFNet: https://arxiv.org/abs/2106.06935
- ULTRA: https://arxiv.org/abs/2310.04562
- KGAT: https://arxiv.org/abs/1905.07854
- KPRN: https://arxiv.org/abs/1811.04540
- Kim & Ko 2025: https://pmc.ncbi.nlm.nih.gov/articles/PMC12158163/
- QCRS 2026: https://doi.org/10.3390/app16104947
- Sci. Rep. 2026: https://www.nature.com/articles/s41598-026-51127-4
- Wang et al. WACV 2020: https://arxiv.org/abs/1910.05134
