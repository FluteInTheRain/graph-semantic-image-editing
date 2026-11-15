# ADR-0051: Tái lập phần truy xuất đồ thị con của G-Retriever (PCST) để xếp hạng lại ảnh

- **Trạng thái:** Đề xuất
- **Ngày:** 2026-11-15

## Bối cảnh

Cấp độ 2 (PDF §5) yêu cầu "triển khai hoặc tái lập **1 phương pháp mới từ năm 2024 trở lại đây** về multimodal KG, graph-enhanced retrieval hoặc explainable graph retrieval, đánh giá trên **cùng pipeline và cách chia** của cấp độ 1".

Kết quả cấp độ 1 trên Visual Genome (`results/tables/main.csv`, test, 3 seed, MRR@100) cho thấy đồ thị hiện **gần như không đóng góp**:

| Mô hình | MRR@100 |
|---|---|
| CLIP thuần `clip_vg` | 0,1076 |
| Đối chứng không đồ thị `clip_nograph_vg` | 0,1266 ± 0,0006 |
| CLIP + GraphSAGE `clip_graph_vg_lr1e4` | 0,1284 ± 0,0006 |
| Xáo cạnh `clip_graph_vg_lr1e4_shuffled` | 0,1272 ± 0,0003 |

Lý do chính: `h_i` của GraphSAGE (ADR-0046, ADR-0048) **giống nhau với mọi truy vấn**. Đồ thị không biết người dùng hỏi gì.

Khái niệm cho người mới:

- **G-Retriever** (He và cộng sự, NeurIPS 2024, https://arxiv.org/abs/2402.07630): trả lời câu hỏi trên đồ thị có chữ. Các bước: (1) chấm độ liên quan của đỉnh với câu hỏi; (2) **chọn một đồ thị con liên thông** bằng PCST; (3) GNN mã hoá đồ thị con; (4) LLM sinh câu trả lời. Bài báo cho rằng cách này giảm "bịa", và đồ thị con cho thấy phần tri thức đã dùng.
- **PCST** (Prize-Collecting Steiner Tree): mỗi đỉnh có **giải thưởng**, mỗi cạnh có **phí**. Thuật toán tìm cây liên thông có tổng giải thưởng trừ tổng phí là lớn nhất. Nếu hai đỉnh có thưởng chỉ nối được qua một đỉnh không có thưởng, đỉnh đó vẫn được giữ làm **cầu nối**. Đây là cơ chế hợp với truy vấn gián tiếp kiểu *child – plays – football*.

Nghiên cứu chọn phương pháp (subagent `paper-researcher`, 2026-11-15) chấm G-Retriever cao nhất trong 9 ứng viên: 33/40. Các ứng viên gồm NativE, AdaMF-MAT, Mixed-Curvature MMKGC, ULTRA, CORA, Structure-CLIP, GOAt, G-RAG.

## Quyết định

Tái lập **bước (1)–(3)** của G-Retriever, **bỏ bước (4) LLM**, để tính điểm đồ thị **phụ thuộc truy vấn** khi xếp hạng lại 100 ứng viên của CLIP.

### 1. Đồ thị con của một truy vấn

| Thành phần | Quy định |
|---|---|
| Đỉnh | 3.880 khái niệm của ADR-0045 (synset VG ở ≥ 5 ảnh train, vector CLIP text đóng băng) |
| Cạnh (bản chính) | Cạnh đồng xuất hiện PMI của ADR-0048 (đếm trên **train**, top 10 mỗi khái niệm), xem như **vô hướng**, bỏ cạnh trùng. **Không** dùng ConceptNet ở bản chính, để tránh "Kênh rò 2" của ADR-0050 |
| Giải thưởng | Theo bài gốc: `k` khái niệm có `cos(q, x_c)` cao nhất nhận giải thưởng `k, k−1, …, 1` theo hạng; các khái niệm khác nhận 0. `q` là vector CLIP text của truy vấn |
| Phí cạnh | Một hằng số `C` cho mọi cạnh, theo bài gốc. PMI chỉ quyết định cạnh nào tồn tại. Cạnh của nhóm không có chữ, nên **không có giải thưởng cho cạnh**. Đây là chỗ khác bài gốc, phải ghi trong báo cáo |
| Thuật toán | `pcst-fast` (Goemans–Williamson kèm pruning, một cây). Tham số gọi hàm cụ thể ghi trong kế hoạch cài đặt sau khi kiểm tài liệu thư viện |
| Kết quả | `S_q`: tập khái niệm của cây, gồm khái niệm có thưởng và khái niệm cầu nối |

### 2. Điểm đồ thị của ảnh

`N(i)` là khái niệm của ảnh `i`: CLIP top-16 của chính ảnh đó, **như nhau cho mọi split, không dùng annotation** (ADR-0046).

| Bậc | Công thức | Ghi chú |
|---|---|---|
| **Bậc 1** (không huấn luyện) | `g(q, i) = \|N(i) ∩ S_q\| / \|S_q\|` | Mọi đỉnh trong `S_q` có trọng số như nhau, **kể cả cầu nối**. Nếu chỉ cộng giải thưởng thì cầu nối (thưởng 0) không góp gì, mất cơ chế gián tiếp. Giải thích chính là `N(i) ∩ S_q` |
| **Bậc 2** (GNN trên đồ thị con) | `h_i(q) = W_self·x_i + b + mean_{c ∈ N(i) ∩ S_q} W_nb·x_c`, `g = cos(q, h_i(q))` | GraphSAGE một lớp của ADR-0046, nhưng chỉ gộp hàng xóm nằm trong `S_q`. Không có hàng xóm trong `S_q` thì bỏ hạng hàng xóm. Khởi tạo bằng CLIP, huấn luyện InfoNCE như ADR-0047, `S_q` của mỗi câu train cache sẵn. **Chỉ làm khi bậc 1 đã chạy được** |

Kết hợp điểm giữ như ADR-0047: `final = alpha·cos(q, x_i) + (1−alpha)·g(q, i)`, trên cùng 100 ứng viên do `retrieve_candidates` trả về.

### 3. Chọn tham số, quy tắc dừng, đánh giá

| Việc | Quy định |
|---|---|
| Chọn | Chỉ trên **val**, theo giao thức ADR-0043 và ADR-0047, chỉ số MRR@100. Bậc 1: `k ∈ {5, 10, 20}`, `C` trong lưới nhỏ ghi ở config, rồi `alpha` (hoà thì chọn `alpha` lớn hơn). Bậc 2: thêm các tham số huấn luyện của ADR-0047 |
| Test | Chạy **một lần** với cấu hình đã chốt, bằng script riêng không chọn gì, giống ADR-0047 |
| Seed | Bậc 1 tất định (std = 0, ghi rõ). Bậc 2 chạy `[0, 1, 2]`, báo cáo mean ± std |
| Quy tắc dừng (ghi trước) | Nếu cả bậc 1 lẫn bậc 2 **không hơn `clip_nograph` trên val**, vẫn chạy test một lần và báo cáo trung thực kết quả âm. Chỉ chuyển sang AdaMF-MAT nếu còn thời gian. Không chỉnh lại tham số sau khi thấy kết quả test |
| Đánh giá | (a) Giao thức chính trên VG test (region description, 5.000 ảnh); (b) bộ truy vấn trực tiếp/gián tiếp v2 (ADR-0050) khi đã được kiểm duyệt; (c) Flickr30k để sau, dùng cùng code khi có đồ thị khái niệm của Flickr |

### 4. Ablation

1. **Không dựng cây**: `S_q` chỉ gồm top-k khái niệm có thưởng, không có cầu nối. Ablation này kiểm phần đóng góp của PCST.
2. **Xáo cạnh PMI**: xáo cạnh khái niệm–khái niệm, giữ bậc của từng đỉnh. Hàm shuffle hiện có chỉ xáo cạnh `contains`, nên cần thêm hàm mới.
3. **Biến thể ConceptNet**: thêm cạnh ConceptNet (đã lọc theo ADR-0050) giữa các khái niệm VG. Kết quả ghi là kịch bản **"KG đã biết"**.
4. Trên bộ truy vấn v2: **loại đúng các cạnh trong `kg_evidence`** của từng truy vấn (ADR-0050, Kênh rò 2).

### 5. Giải thích (ADR-0016)

- Bậc 1: giải thích là `N(i) ∩ S_q`, kèm các cạnh của cây nối chúng. Fidelity kiểm được trực tiếp: bỏ một khái niệm khỏi `N(i) ∩ S_q` thì `g` giảm đúng `1/|S_q|`.
- Bậc 2: dùng explainer có sẵn trong `explain/` (occlusion) trên chính mô hình.

### 6. Dependency

Thêm `pcst-fast` vào nhóm `graph`. Thư viện có wheel cho Windows `cp311` và Linux, license **GNU GPL**. Đồ án học thuật không phân phối thương mại, nhưng phải ghi license trong README hoặc báo cáo. Thêm dependency là quyết định của nhóm (CLAUDE.md).

Quyết định bởi: Thuat Nguyen (brainstorm với Claude Code), 2026-11-15. Các giá trị lưới cụ thể là đề xuất, chốt trên val. **Nhóm chưa duyệt.**

## Các phương án đã cân nhắc

| Phương án | Ưu | Nhược | Lý do không chọn |
|---|---|---|---|
| G-Retriever, phần truy xuất đồ thị con (PCST) + GNN | Điểm phụ thuộc truy vấn; cầu nối hợp truy vấn gián tiếp; đồ thị con là giải thích; code gốc MIT | Bỏ phần LLM; cạnh không có chữ nên không có giải thưởng cho cạnh; thêm dependency GPL | **Được chọn** |
| AdaMF-MAT (LREC-COLING 2024) | Nằm trong danh sách đề gợi ý; không cần dependency mới | Bài toán hoàn thiện KG; các "modality" đều từ CLIP nên mức tăng có thể nằm trong nhiễu; repo không có license | Giữ làm phương án dự phòng (quy tắc dừng) |
| NativE (SIGIR 2024) | Đề gợi ý | Môi trường cũ, không có license, không làm truy vấn ảnh | Khó tái lập trong thời gian môn học |
| Mixed-Curvature MMKGC (AAAI 2025) | Đề gợi ý | Không tìm thấy code | Không tái lập được |
| ULTRA / UltraQuery | MIT, zero-shot | Kernel CUDA tự biên dịch, quy mô đồ thị ở cận trên, khó vấn đáp | Chi phí cao |
| PCST tự viết (tham lam) | Không thêm dependency, đọc được từng dòng | Chỉ là xấp xỉ, khác thuật toán bài gốc | Muốn bám sát bài gốc |
| `networkx.steiner_tree` | Có sẵn | Steiner thường, không có giải thưởng | Lệch bài gốc nhiều nhất |
| Đồ thị khái niệm chỉ từ ConceptNet | Đúng tinh thần KG của đề | Trùng nguồn với câu gián tiếp v2, thành kịch bản "KG đã biết" | Giữ làm ablation 3 |
| Bậc 1 cộng giải thưởng | Gần cách chấm đỉnh của bài gốc | Cầu nối có thưởng 0 nên không góp gì | Chấm theo tỉ lệ chạm `S_q` |

## Hệ quả

- **Dễ hơn:** dùng lại gần hết pipeline: ứng viên CLIP, cạnh `contains` CLIP top-16, cạnh PMI, giao thức chọn trên val, script test tách riêng, `explain/`. Bậc 1 không cần GPU.
- **Rò rỉ dữ liệu:**
  - Cạnh PMI chỉ đếm trên train.
  - Cạnh `contains` không dùng annotation ở split nào.
  - PCST chỉ dùng văn bản truy vấn.
  - Mọi tham số chọn trên val; test chạy một lần.
  - Bộ truy vấn v2 chỉ dùng để đánh giá.
  - Bản chính không dùng ConceptNet nên tránh Kênh rò 2 của ADR-0050.
- **Khó hơn:**
  - Thêm dependency GPL.
  - Bậc 2 phải chạy PCST cho mọi câu train và cache kết quả. Thời gian chưa đo, phải đo trước khi chạy hết.
  - Báo cáo phải nói rõ phần nào là tái lập, phần nào là điều chỉnh của nhóm: bỏ LLM, không có giải thưởng cho cạnh, cách chấm bậc 1.
- **Việc tiếp theo:**
  - Kế hoạch 1: ADR này, `pcst.py`, bậc 1, chọn trên val, ablation 1–2.
  - Kế hoạch 2: bậc 2, ablation 3–4, đánh giá trên bộ truy vấn v2.

## Nguồn tham khảo

- Đề bài, §2, §5, §6, §7, §10.
- He et al., "G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering", NeurIPS 2024. https://arxiv.org/abs/2402.07630, code https://github.com/XiaoxinHe/G-Retriever (MIT). Chi tiết cách đặt giải thưởng và phí lấy từ báo cáo nghiên cứu ngày 2026-11-15; **cần đối chiếu lại với mục "Retrieval" và "Subgraph Construction" của bài gốc** trước khi cài đặt.
- `pcst-fast`: https://pypi.org/project/pcst-fast/ (GPL).
- ADR-0009, ADR-0016, ADR-0043, ADR-0045, ADR-0046, ADR-0047, ADR-0048, ADR-0050.
