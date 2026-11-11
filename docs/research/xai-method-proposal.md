# Đề xuất phương pháp cho module xAI (Nhiệm vụ 3)

- **Trạng thái:** Bản nghiên cứu làm cơ sở cho ADR-0016 (bản sửa đổi 2026-11-07).
- **Ngày:** 2026-11-07
- **Soạn bởi:** Claude Code (AI hỗ trợ). Thành viên phải tự đọc lại các bài báo trước khi trích dẫn.
- **Phạm vi:** chỉ module xAI (`src/mmxai/explain/`). Module Graph (dựng đồ thị, GNN, scorer trong `models/`) do phần khác của nhóm làm. Mô hình đồ thị được coi là **hộp đen**: nhóm Graph chọn GNN nào cũng được.
- **Nguồn sự thật là đề bài** (`Project03_Multimodal_XAI.pdf`). ADR-0016 đã được sửa theo hướng của tài liệu này (mục 7).

Nguyên tắc: **áp dụng phương pháp đã có**, không phát minh mới. Với mỗi phương pháp phải trả lời được: **làm gì, vì sao chọn, input, output, luồng xử lý**.

## 1. Đề bài yêu cầu gì ở phần xAI

**Quy tắc xuyên suốt (PDF §2):** giải thích phải đến từ **chính cơ chế xếp hạng**. Không được sinh ra một đường đi "nghe hợp lý" sau khi đã có kết quả nếu đường đi đó không ảnh hưởng tới dự đoán.

| Cấp độ | Yêu cầu xAI |
|---|---|
| 1 (trần 7.0) | Hiển thị quan hệ/đường đi **đã dùng khi xếp hạng** cho một số truy vấn; phân tích 5 truy vấn thành công + 5 truy vấn thất bại |
| 3 (trần 10) | Đánh giá **định lượng** fidelity, validity, sparsity trên **≥ 10 truy vấn** |
| 4 (điểm cộng) | Tùy chọn: so sánh giải thích giữa các mô hình; khảo sát người dùng có đối chứng |

Demo (PDF §8) phải hiển thị được quan hệ/đường đi giải thích kèm top-k kết quả. Đề chấp nhận giải thích dạng **quan hệ, đường đi hoặc đồ thị con** (PDF §1).

## 2. "Giải thích đến từ chính cơ chế xếp hạng" nghĩa là gì

### 2.1. Đề bài cấm cái gì

Điều bị cấm là giải thích **tách rời khỏi mô hình**. Ví dụ: xếp hạng xong mới tìm đường ngắn nhất từ `football` tới ảnh trong đồ thị. Đường đi đó nghe hợp lý, nhưng mô hình có thể chưa từng dùng nó. Cách kiểm là **fidelity** (PDF §5): bỏ hoặc làm nhiễu đường đi đó thì điểm/thứ hạng phải đổi.

Đề **không** yêu cầu mô hình có dạng đặc biệt. Một giải thích hợp lệ khi:
1. nó được tính **trên đúng mô hình đã xếp hạng**, và
2. có **số fidelity** chứng minh nó thật sự ảnh hưởng tới điểm/thứ hạng.

### 2.2. Vì sao với GNN phi tuyến thì "đóng góp của một cạnh" không duy nhất

GNN thông thường có hàm kích hoạt phi tuyến (ReLU…), nên các cạnh tương tác với nhau. Ví dụ một ảnh có hai cạnh, mask `m1`, `m2` (1 = có cạnh):

| Scorer | Đủ 2 cạnh | Xoá e1 | Xoá e2 | Occlusion nói | IG / Shapley nói |
|---|---|---|---|---|---|
| Tuyến tính: `g = 0.6·m1 + 0.4·m2` | 1.0 | 0.4 | 0.6 | e1 = 0.6, e2 = 0.4 | e1 = 0.6, e2 = 0.4 |
| Phi tuyến: `g = ReLU(m1 + m2 − 1)` | 1.0 | 0 | 0 | e1 = 1, e2 = 1 (tổng 2 ≠ 1) | e1 = 0.5, e2 = 0.5 |

- **Tuyến tính:** mọi phương pháp cho cùng một đáp án.
- **Phi tuyến:** hai cạnh hoạt động như phép "VÀ"; mỗi phương pháp chia điểm một kiểu.

Điều này **không làm giải thích sai đề**. Nó chỉ có nghĩa là nhóm phải **chọn một explainer và bảo vệ lựa chọn đó bằng số liệu**. Vì vậy phần nghiên cứu chính của module xAI là so sánh các explainer bằng fidelity (mục 4.4).

### 2.3. Hướng của module xAI

Module xAI **tự tính đóng góp của cạnh trên chính mô hình đồ thị đã huấn luyện**, bằng các explainer đã công bố. Explainer chính được chọn **trên val** dựa vào fidelity. Cách này:
- đúng đề (PDF §2, §5);
- không ràng buộc thiết kế của module Graph;
- biến câu hỏi "vì sao chọn explainer này" thành một câu trả lời có số liệu.

## 3. Bốn phần việc của module xAI

| # | Phần việc | Phương pháp có sẵn để áp dụng | Căn cứ trong đề |
|---|---|---|---|
| 1 | Tính đóng góp của cạnh và trình bày giải thích | Occlusion, Integrated Gradients, GNNExplainer (attention nếu mô hình là GAT); lấy top-m cạnh, hiển thị dạng đường đi hoặc đồ thị con | §1, §2, §5 cấp độ 1 |
| 2 | Hàm can thiệp | Xoá hoặc làm nhiễu cạnh trên `edge_index` rồi gọi lại scorer. Dùng chung cho fidelity và ablation đồ thị | §5, §7 |
| 3 | **Đánh giá giải thích** | Fid+/Fid−, characterization (GraphFramEx, LoG 2022); robust fidelity (Zheng et al., ICLR 2024); sparsity; validity; đối chứng xoá ngẫu nhiên; kiểm định Wilcoxon | §5 cấp độ 3, §7 |
| 4 | **So sánh các explainer, chọn explainer chính trên val** | Chấm mọi explainer bằng cùng metric ở phần 3, cùng kích thước m | §7 (mở rộng: so sánh giải thích) |

## 4. Chi tiết từng phần

### 4.1. Phần 1 – Tính đóng góp và trình bày giải thích

- **Làm gì:** với mỗi cặp (truy vấn q, ảnh ứng viên i), gán cho mỗi cạnh trong đồ thị con k-hop quanh i một điểm đóng góp, tính trên chính mô hình. Lấy top-m cạnh, đổi id thành tên node và tên quan hệ, xuất ra dạng đọc được.
- **Vì sao:** đề cấm giải thích tách rời khỏi mô hình, nên `explain/` không được chọn đường đi bằng thuật toán riêng như `networkx.shortest_path`. Các explainer dưới đây đều hỏi **chính mô hình**: xoá cạnh thì điểm đổi bao nhiêu, hoặc gradient của điểm theo cạnh.
- **Các explainer ứng viên:**

| Explainer | Cách làm | Ưu | Nhược |
|---|---|---|---|
| **Occlusion** | Xoá từng cạnh, chấm lại, đo điểm giảm | Can thiệp thật, dễ giải thích khi vấn đáp, không cần gradient | Bỏ sót tương tác giữa các cạnh (mục 2.2); mỗi cạnh tốn một lần chấm lại |
| **Integrated Gradients** (ICML 2017) | Cho mask trên cạnh chạy từ 0 tới 1, cộng dồn gradient | Tính completeness: tổng đóng góp = `g(đủ cạnh) − g(không cạnh)` | Cần mô hình khả vi; khoảng 20–50 lần lan truyền ngược |
| **GNNExplainer** (NeurIPS 2019) | Tối ưu một mask cạnh riêng cho từng mẫu | Phổ biến, có sẵn trong PyG | Có siêu tham số; GInX-Eval (2023) cho thấy thường không hơn chọn ngẫu nhiên |
| Attention (chỉ khi mô hình là GAT) | Đọc trọng số attention | Rẻ | Attention không chắc phản ánh tầm quan trọng (Jain & Wallace 2019) |

- **Input:** mô hình đồ thị đã huấn luyện; `x`, `edge_index`, loại cạnh; truy vấn q; ảnh i; ánh xạ id → tên.
- **Output:** danh sách `(cạnh, quan hệ, đóng góp)` cho mỗi cặp (q, i); đồ thị con giải thích gồm top-m cạnh, dùng cho demo và phân tích 5 thành công + 5 thất bại.
- **Luồng:**
  ```
  (q, i) từ kết quả rerank
   → lấy đồ thị con k-hop S(q,i) quanh ảnh i      (k = số lớp GNN: cạnh xa hơn không ảnh hưởng điểm)
   → explainer chạy trên mô hình → đóng góp của từng cạnh trong S
   → sắp xếp, lấy top-m                            (m chọn trên val, ghi vào config)
   → đổi id → tên → xuất bảng / hình đồ thị con
  ```

### 4.2. Phần 2 – Hàm can thiệp

- **Làm gì:** từ đồ thị gốc tạo một đồ thị đã xoá (hoặc làm nhiễu) một tập cạnh cho trước, rồi gọi lại scorer để lấy điểm mới.
- **Vì sao:** fidelity theo đề là "bỏ hoặc làm nhiễu cạnh thì điểm/thứ hạng phải đổi", tức là phải **can thiệp thật**. Ablation đồ thị (PDF §7: bỏ cạnh, xáo cạnh, đồ thị ngẫu nhiên) và Occlusion cũng cần đúng thao tác này. Dùng chung một hàm thì các con số so sánh được với nhau.
- **Input:** `edge_index` (và loại cạnh), tập cạnh cần xoá/nhiễu, scorer (gọi như hộp đen).
- **Output:** điểm và thứ hạng mới của các ứng viên.
- **Luồng:**
  ```
  edge_index, tập cạnh E
   → tạo edge_index' = edge_index ∖ E        (không sửa đồ thị gốc)
   → gọi scorer(q, ứng viên, edge_index')
   → tính lại final với α đã chốt → thứ hạng mới
  ```

### 4.3. Phần 3 – Đánh giá giải thích

- **Làm gì:** đo xem giải thích có trung thực, hợp lệ và gọn không, trên ≥ 10 truy vấn × 3 seed.
- **Vì sao chọn các thước đo dưới đây:**
  - Đề không định nghĩa chi tiết validity và sparsity, nên nhóm lấy định nghĩa từ các khung đánh giá đã công bố thay vì tự chế.
  - GraphFramEx tách hai khía cạnh: giải thích **cần thiết** (bỏ đi thì điểm sụp, Fid+) và **đủ** (chỉ giữ nó thì điểm vẫn còn, Fid−).
  - Robust Fidelity xử lý vấn đề xoá cạnh tạo ra đồ thị lệch khỏi phân phối lúc huấn luyện (OOD).
- **Input:** giải thích từ phần 1, hàm can thiệp (phần 2), danh sách truy vấn đánh giá.
- **Output:** bảng metric theo truy vấn, có tách nhóm trực tiếp/gián tiếp, kèm đối chứng ngẫu nhiên và p-value.

Ký hiệu: `i*` là ảnh đang giải thích, `E_x` là tập cạnh giải thích, `S(q,i*)` là đồ thị con k-hop quanh `i*`, `g` là điểm đồ thị.

| Metric | Công thức / cách đo | Ý nghĩa |
|---|---|---|
| Fid+ (điểm) | `g(q,i*) − g_{G∖E_x}(q,i*)` | Bỏ cạnh giải thích thì điểm giảm bao nhiêu (lớn là tốt) |
| Fid+ (thứ hạng) | `rank_{G∖E_x}(i*) − rank_G(i*)`, tính trên `final` với α đã chốt | Bỏ cạnh giải thích thì ảnh tụt bao nhiêu hạng |
| Fid− | `g(q,i*) − g_{E_x}(q,i*)` | Chỉ giữ cạnh giải thích thì điểm giảm bao nhiêu (≈ 0 là tốt) |
| Characterization | `Fid+·(1−Fid−) / (0.5·(1−Fid−) + 0.5·Fid+)`, tính trên **Fid chuẩn hoá** (xem dưới bảng) | Gộp Fid+ và Fid− thành một số |
| Robust Fid+ / Fid− | Fid+: mỗi quan hệ của `E_x` chỉ bị xoá với xác suất `robust_remove_prob` (mặc định 0.1). Fid−: giữ `E_x` và thêm lại mỗi quan hệ khác với xác suất `robust_add_back_prob` (mặc định 0.9). Lấy trung bình nhiều lần. Xác suất 1 và 0 cho lại Fid thường | Đồ thị sau can thiệp vẫn gần đồ thị gốc, giảm ảnh hưởng của OOD |
| Sparsity | `1 − |E_x| / |E(S(q,i*))|` | Giải thích càng gọn càng tốt |
| Validity | (i) mọi cạnh của `E_x` có thật trong đồ thị train và nằm trong `S(q,i*)`; (ii) `E_x` nối được tới `i*` thành ít nhất một đường đi. `networkx` chỉ dùng ở bước kiểm này | Không có cạnh "ảo", giải thích đọc được thành đường đi |

Fid+ và Fid− dùng **hiệu chưa chia**, đúng định nghĩa gốc của Zheng et al. (ICLR 2024). Không chia cho `g(q,i*)` vì điểm đồ thị có thể bằng 0 hoặc âm (ví dụ điểm kiểu cosine). Code: `src/mmxai/explain/fidelity.py`.

**Fid chuẩn hoá (cho characterization):** chia cho **toàn bộ phần điểm mà cạnh đồ thị tạo ra**, `g(đủ cạnh) − g(không cạnh nào)`, rồi kẹp về [0, 1]. Ví dụ Fid+ chuẩn hoá = 0.5 nghĩa là bỏ giải thích làm mất một nửa phần điểm do đồ thị tạo ra. Mẫu số này khác 0 kể cả khi điểm âm. Nếu cạnh không làm đổi điểm (mẫu số = 0) thì không có gì để giải thích, và metric để trống (`None`).

**Lưu ý về robust fidelity:** bản trước của tài liệu ghi ngược ("xoá với xác suất 0.9"). Code dùng cách hiểu xoá **ít** (0.1) và thêm lại **nhiều** (0.9), để đồ thị sau can thiệp gần đồ thị gốc, đúng mục đích chống OOD của bài báo. **Cần đọc lại bài gốc để xác nhận** trước khi báo cáo; hai giá trị nằm trong `configs/explain_vg.yaml`.

**Đối chứng:** xoá cùng `|E_x|` cạnh chọn ngẫu nhiên **trong `S(q,i*)`** (không lấy từ toàn đồ thị, vì như vậy gần như không chạm cạnh nào liên quan), lặp R ≥ 20 lần. Kiểm định bằng Wilcoxon signed-rank ghép cặp theo truy vấn.

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
2. **Occlusion có lợi thế sẵn ở Fid+:** Occlusion chọn cạnh đúng theo kiểu "xoá đi thì điểm giảm", gần giống cách Fid+ đo. Vì vậy khi so sánh explainer phải xem cả Fid− và characterization, không chỉ Fid+.

### 4.4. Phần 4 – So sánh các explainer, chọn explainer chính

- **Làm gì:** chạy mọi explainer ở mục 4.1 trên cùng mô hình, cùng truy vấn, cùng kích thước m; chấm bằng metric ở phần 3, kèm baseline ngẫu nhiên. Explainer có characterization cao nhất **trên val** được dùng làm explainer chính cho demo và bảng test.
- **Vì sao:** với GNN phi tuyến, không có "đóng góp đúng duy nhất" (mục 2.2), nên lựa chọn phải dựa trên số đo. Chọn trên val để tuân thủ PDF §6 (mọi lựa chọn làm trên validation, test chỉ chạy một lần).
- **Input:** đầu ra của phần 1 cho từng explainer; metric của phần 3.
- **Output:** bảng so sánh explainer (mean ± std trên 3 seed), explainer chính đã chốt.
- **Luồng:**
  ```
  val:  mỗi explainer → top-m → metric phần 3 → bảng so sánh → chọn explainer chính, chốt m
  test: chạy một lần với explainer và m đã chốt → bảng báo cáo
  ```

## 5. Câu hỏi nghiên cứu và giả thuyết

| RQ | Câu hỏi | Giả thuyết (bác bỏ được) |
|---|---|---|
| RQ1 | Giải thích có thật sự ảnh hưởng tới xếp hạng không? | Fid+ (thứ hạng) của `E_x` lớn hơn xoá ngẫu nhiên cùng kích thước (Wilcoxon, p < 0.05) |
| RQ2 | Explainer nào trung thực nhất trên mô hình của nhóm? | Ở cùng sparsity, có ít nhất một explainer có characterization cao hơn hẳn baseline ngẫu nhiên; thứ tự giữa Occlusion, IG và GNNExplainer được báo cáo kèm p-value |
| RQ3 | Các explainer có đồng ý với nhau không? | Thứ tự cạnh theo IG và theo Occlusion tương quan Spearman ≥ 0.5. Nếu thấp, đó là bằng chứng các cạnh tương tác mạnh (mục 2.2) |
| RQ4 | Truy vấn gián tiếp có khó giải thích hơn không? | Với truy vấn gián tiếp, khoảng cách giữa Fid+ thường và robust Fid+ lớn hơn truy vấn trực tiếp |

Nếu mô hình là GAT, thêm RQ5: attention có khớp với Occlusion không (Jain & Wallace 2019 dự đoán là không). RQ2 đáp ứng mục mở rộng "so sánh giải thích" của đề (PDF §7).

## 6. Interface module xAI cần nhận

**Interface** ở đây là hợp đồng giữa hai module: module Graph đưa cho module xAI dữ liệu gì, ở dạng nào, và gọi hàm nào. Không phải giao diện người dùng (UI). Đây chỉ là **yêu cầu interface**; nhóm Graph chọn GNN nào cũng được.

| Cần nhận | Từ module | Vì sao xAI cần |
|---|---|---|
| Mô hình đồ thị đã huấn luyện, **chạy lại được** trên một `edge_index` đã sửa, ở chế độ `eval()` (cùng đồ thị cho cùng điểm) | Graph | Phần 1, 2, 3. Cần mô hình, không chỉ embedding đã cache. Mô hình còn ngẫu nhiên (dropout) thì mức điểm giảm chỉ là nhiễu; `occlusion` báo lỗi nếu gọi hai lần ra hai điểm khác nhau |
| Hàm tính điểm đồ thị `g(q, i)` từ embedding | Graph / Rerank | Explainer và fidelity đều phải gọi đúng hàm đã dùng khi xếp hạng |
| `x`, `edge_index`, loại cạnh, ánh xạ id ↔ tên node/quan hệ | Graph | Hiển thị, validity, đối chứng ngẫu nhiên |
| Số lớp GNN (để biết k-hop) | Graph | Giới hạn đồ thị con cần giải thích. `edges_reaching_node` lấy thêm 1 bước vì GCN chuẩn hoá theo bậc của node gửi; `occlusion` báo lỗi nếu cạnh ngoài vùng này vẫn làm đổi điểm (số lớp khai báo sai) |
| Kết quả rerank: `cosine`, `graph`, `final`, thứ hạng; `α` đã chốt trên val | Rerank | Tính thứ hạng sau can thiệp |

IG và GNNExplainer được cài trực tiếp trong `src/mmxai/explain/` (không dùng bản của PyG), nên không phụ thuộc cách nhóm Graph xây mô hình. Hai explainer này cần một hàm **`score_and_gradient(weights)`**: nhận trọng số 0..1 cho từng cạnh, trả về điểm đồ thị của ảnh và gradient theo từng trọng số. Phía Graph tạo hàm này bằng autograd của torch, đại ý:

```python
def score_and_gradient(weights):
    w = torch.tensor(weights, requires_grad=True)
    score = graph_score(model, x, edge_index, edge_weight=w, image=i)  # mô hình ở eval()
    score.backward()
    return score.item(), w.grad.numpy()
```

Mô hình cần nhận trọng số cạnh (ví dụ `edge_weight` của `GCNConv`). Occlusion chỉ cần `score_image(mask)` (bật/tắt cạnh), không cần gradient.

Trong lúc module Graph chưa có code, module xAI viết và test trên một **mô hình giả cỡ nhỏ trong `tests/`**. Mô hình giả này không nằm trong `models/`.

## 7. Quan hệ với ADR-0016

Bản đầu của ADR-0016 (2026-11-01) chọn cách hiểu **chặt hơn đề**: scorer tự trả về bản ghi đóng góp, `explain/` chỉ đọc. Ngày 2026-11-07, nhóm **sửa trực tiếp ADR-0016** theo hướng của tài liệu này (xem mục "Lịch sử sửa đổi" trong ADR):

| Mục ADR-0016 | Thay đổi |
|---|---|
| 1. Module Graph | Chọn GNN tuỳ ý; chỉ cần mô hình chạy lại được và hàm tính điểm. Bỏ yêu cầu trả bản ghi đóng góp |
| 2. `final = α·cosine + (1−α)·graph` | Giữ |
| 3. `explain/` | Tính đóng góp trên chính mô hình bằng explainer đã công bố, chọn trên val bằng fidelity; **vẫn cấm** chọn giải thích bằng thuật toán tìm đường riêng |
| 4. Fidelity bằng can thiệp thật, so với xoá ngẫu nhiên | Giữ |
| 5. ≥ 10 truy vấn, đối chứng ngẫu nhiên, ≥ 3 seed | Giữ |
| 6. Ablation dùng chung hàm can thiệp | Giữ |

`CLAUDE.md` và `.claude/rules/explanations.md` đã được cập nhật cho khớp.

## 7b. Code đã cài (cấp độ 1 và 3)

| File | Hàm | Dùng cho |
|---|---|---|
| `src/mmxai/explain/intervention.py` | `mark_remaining_edges` | Xoá cạnh rồi chấm lại; dùng chung cho Occlusion, fidelity, ablation |
| `src/mmxai/explain/subgraph.py` | `edges_reaching_node`, `relations_in_reach` | Các cạnh/quan hệ có thể làm đổi điểm của ảnh |
| `src/mmxai/explain/occlusion.py` | `occlusion` | Explainer 1: xoá từng quan hệ (cấp độ 1) |
| `src/mmxai/explain/integrated_gradients.py` | `integrated_gradients` | Explainer 2: IG, kèm sai lệch completeness |
| `src/mmxai/explain/gnn_explainer.py` | `gnn_explainer` | Explainer 3: mask học được |
| `src/mmxai/explain/presentation.py` | `build_explanation`, `format_explanation` | Trình bày giải thích bằng tên (cấp độ 1) |
| `src/mmxai/explain/fidelity.py` | `fidelity_plus/minus`, `rank_drop`, `evaluate_fidelity`, `random_control`, `robust_fidelity`, `normalized_fidelity`, `characterization`, `evaluate_explanation` | Metric cấp độ 3 cho một giải thích |
| `src/mmxai/explain/quality.py` | `sparsity`, `validity` | Độ gọn, tính hợp lệ |
| `src/mmxai/explain/statistics.py` | `spearman`, `wilcoxon_greater`, `mean_and_std`, `choose_explainer` | So sánh explainer, kiểm định, chọn explainer trên val |
| `src/mmxai/explain/config.py`, `configs/explain_vg.yaml` | `load_explain_config` | Mọi siêu tham số của module xAI |

**Còn thiếu (bị chặn bởi module Graph):** script chạy trên mô hình thật cho ≥ 10 truy vấn × 3 seed và ghi `results/runs/` (ADR-0010), bảng tổng hợp, phân tích 5 thành công + 5 thất bại. Attention của GAT chỉ thêm nếu nhóm Graph chọn GAT.

## 8. Câu hỏi còn mở (phía xAI)

1. **m** (số cạnh trong một giải thích) và **explainer chính**: chọn trên **val** (mục 4.4).
2. **Số lần lặp R** của đối chứng, **số bước T** của IG, xác suất của robust fidelity, siêu tham số GNNExplainer: đã đưa vào `configs/explain_vg.yaml` với giá trị mặc định từ bài báo/PyG; chỉnh (nếu cần) chỉ trên val.
3. **Truy vấn đánh giá:** lấy ≥ 10 truy vấn từ bộ benchmark (nhánh `feat/vg-query-benchmark`, ADR-0044), có cả nhóm trực tiếp lẫn gián tiếp.
4. **Xoá cạnh hai chiều hay một chiều:** PyG lưu một quan hệ vô hướng thành hai cạnh có hướng. Cần hỏi nhóm Graph đồ thị của họ có hướng hay không.

## 9. Tóm tắt khảo sát tài liệu

Subagent `paper-researcher` khảo sát qua web vì MCP `arxiv` không kết nối được. Chỗ đánh dấu **[chưa xác minh]** là subagent chưa đọc được nguồn gốc.

### 9.1. Explainer cho GNN (phần 1, 4)

- **GNNExplainer** (NeurIPS 2019): học mask cạnh cho từng mẫu; có trong `torch_geometric.explain`.
- **PGExplainer** (NeurIPS 2020): mask được tham số hoá bằng mạng nhỏ, dùng lại được cho mẫu mới.
- **SubgraphX** (ICML 2021): tìm đồ thị con bằng MCTS + Shapley; chậm; code trong DIG (GPL-3.0).
- **Integrated Gradients** (ICML 2017): có tính completeness; dùng trong PyG qua `CaptumExplainer`.
- **Attention làm giải thích:** Jain & Wallace (NAACL 2019) cho rằng không phải giải thích; Wiegreffe & Pinter (EMNLP 2019) phản biện, đề nghị kiểm bằng can thiệp. Với GNN: GAtt (2024, venue [chưa xác minh]), FGAI (TMLR 2025).

### 9.2. Đánh giá giải thích (phần 3)

- **GraphFramEx** (LoG 2022): giải thích sufficient (Fid− ≈ 0) và necessary (Fid+ ≈ 1); characterization score; khuyến nghị so sánh các giải thích có cùng kích thước.
- **Robust Fidelity** (ICLR 2024): chỉ xoá ngẫu nhiên một phần cạnh giải thích để giảm OOD; không phạt kích thước nên vẫn cần báo cáo sparsity.
- **GInX-Eval** (NeurIPS 2023 workshop): xoá cạnh rồi huấn luyện lại; nhiều explainer không tốt hơn chọn ngẫu nhiên.
- **GraphXAI** (*Scientific Data* 2023): các thước đo GEA, GEF/unfaithfulness, stability.
- PyG có `fidelity` và `characterization_score`, nhưng viết cho **phân loại**. Bài của nhóm là **xếp hạng**, nên cần bản liên tục như ở mục 4.3.

### 9.3. Bối cảnh

- Mô hình có giải thích theo đường đi như NBFNet (NeurIPS 2021) và ULTRA (ICLR 2024) cũng tính giải thích bằng gradient, tức là cùng loại với IG.
- Truy vấn cross-modal có giải thích, 2022–2026: không tìm thấy bài làm đúng chuỗi "CLIP top-k → xếp hạng lại bằng scene graph → giải thích faithful". Các bài gần nhất (chưa đọc toàn văn): Kim & Ko (*Sensors* 2025), QCRS (*Applied Sciences* 2026), *Scientific Reports* 2026, Wang et al. (WACV 2020).

## Tài liệu tham khảo

- Integrated Gradients: https://arxiv.org/abs/1703.01365
- GNNExplainer: https://arxiv.org/abs/1903.03894
- PGExplainer: https://arxiv.org/abs/2011.04573
- SubgraphX: https://arxiv.org/abs/2102.05152 · https://github.com/divelab/DIG (GPL-3.0)
- GraphFramEx: https://arxiv.org/abs/2206.09677 · https://github.com/GraphFramEx/graphframex
- Robust Fidelity: https://arxiv.org/abs/2310.01820
- GInX-Eval: https://arxiv.org/abs/2309.16223
- GraphXAI: https://arxiv.org/abs/2208.09339 · https://github.com/mims-harvard/GraphXAI
- Jain & Wallace: https://arxiv.org/abs/1902.10186
- Wiegreffe & Pinter: https://arxiv.org/abs/1908.04626
- GAtt: https://arxiv.org/abs/2406.04612
- FGAI: https://arxiv.org/abs/2410.06950
- NBFNet: https://arxiv.org/abs/2106.06935
- ULTRA: https://arxiv.org/abs/2310.04562
- Kim & Ko 2025: https://pmc.ncbi.nlm.nih.gov/articles/PMC12158163/
- QCRS 2026: https://doi.org/10.3390/app16104947
- Sci. Rep. 2026: https://www.nature.com/articles/s41598-026-51127-4
- Wang et al. WACV 2020: https://arxiv.org/abs/1910.05134
