# ADR-0016: Giải thích phải tính trên chính mô hình xếp hạng và được kiểm bằng fidelity

- **Trạng thái:** Đã chấp nhận
- **Ngày:** 2026-11-01
- **Sửa đổi:** 2026-11-07 – bỏ yêu cầu scorer tự trả bản ghi đóng góp (xem mục "Lịch sử sửa đổi")

## Bối cảnh

Đề (§2) có yêu cầu xuyên suốt: "giải thích phải đến từ chính cơ chế xếp hạng, không được sinh một đường đi hợp lý sau khi đã có kết quả mà đường đi đó không ảnh hưởng tới dự đoán". Cấp độ 3 đo định lượng:

| Thước đo | Ý nghĩa đơn giản |
|---|---|
| **Fidelity** (trung thực) | Bỏ/làm nhiễu các cạnh trong giải thích → điểm/thứ hạng phải giảm rõ |
| **Validity** (hợp lệ) | Đường đi có thật trong đồ thị và nối được truy vấn với ứng viên |
| **Sparsity** (gọn) | Giải thích dùng ít cạnh so với toàn bộ đồ thị con |

Đề **không** yêu cầu mô hình có dạng đặc biệt. Điều bị cấm là giải thích **tách rời khỏi mô hình** (ví dụ tìm đường ngắn nhất sau khi xếp hạng). Một giải thích hợp lệ khi nó được tính **trên đúng mô hình đã xếp hạng** và có **số fidelity** chứng minh nó ảnh hưởng tới điểm/thứ hạng.

Với GNN có hàm kích hoạt phi tuyến (GCN/GAT/GraphSAGE thông thường), "đóng góp của một cạnh" không duy nhất: các explainer khác nhau chia điểm theo các cách khác nhau. Vì vậy nhóm phải chọn explainer bằng số liệu, không chọn cảm tính.

## Quyết định

1. Module Graph (`src/mmxai/models/`) được chọn GNN tuỳ ý. Nó chỉ cần cung cấp một mô hình **chạy lại được** trên `edge_index` đã sửa, cùng hàm tính điểm đồ thị đã dùng khi xếp hạng. Không bắt buộc trả về bản ghi đóng góp.
2. `src/mmxai/rerank/` tính `final = alpha·cosine + (1−alpha)·graph`.
3. `src/mmxai/explain/` **tính đóng góp của cạnh trên chính mô hình đã xếp hạng** bằng explainer đã công bố (Occlusion, Integrated Gradients, GNNExplainer; attention nếu mô hình là GAT). Explainer chính và kích thước giải thích được **chọn trên val** bằng fidelity. Cấm chọn giải thích bằng thuật toán tìm đường riêng (ví dụ `networkx.shortest_path`); `networkx` chỉ được dùng để kiểm tra validity.
4. Fidelity đo bằng **can thiệp thật**: xoá các cạnh giải thích khỏi đồ thị → chạy lại scorer → so điểm/thứ hạng; so với xoá cùng số cạnh ngẫu nhiên (đối chứng).
5. Mọi bảng explain báo cáo trên ≥ 10 truy vấn, kèm đối chứng ngẫu nhiên, ≥ 3 seed.
6. Ablation đồ thị (bỏ cạnh / xáo cạnh / đồ thị ngẫu nhiên, §7) dùng chung hàm can thiệp với fidelity.

## Các phương án đã cân nhắc

| Phương án | Ưu | Nhược | Lý do không chọn |
|---|---|---|---|
| Explainer tính trên chính mô hình, chọn trên val bằng fidelity | Đúng §2; module Graph chọn GNN tuỳ ý; lựa chọn explainer có số liệu bảo vệ | Đóng góp không duy nhất với GNN phi tuyến; tốn thêm thời gian chạy | **Được chọn** (từ 2026-11-07) |
| Scorer tự trả kèm đóng góp (faithful by construction) | Đóng góp chính xác, không cần chọn explainer | Buộc module Graph dùng scorer phân rã được (ví dụ GCN tuyến tính); chặt hơn đề | Lựa chọn ban đầu (2026-11-01), bỏ vì ràng buộc module Graph mà đề không yêu cầu |
| Post-hoc: tìm đường đi ngắn nhất sau khi xếp hạng | Dễ làm | Đường đi có thể không ảnh hưởng điểm → vi phạm đề | Bị đề cấm |
| Chỉ dùng một explainer cố định (ví dụ GNNExplainer) | Đơn giản | Không có số liệu để bảo vệ lựa chọn | Thay bằng so sánh nhiều explainer trên val |
| Attention weights của GAT làm giải thích | Có sẵn | Attention không chắc phản ánh tầm quan trọng | Chỉ là một ứng viên, phải qua so sánh fidelity |

## Hệ quả

- Interface giữa `models/` và `explain/` (mô hình chạy lại được, hàm tính điểm, ánh xạ id ↔ tên) phải được thống nhất sớm. Chi tiết: `docs/research/xai-method-proposal.md` mục 6.
- Có thể trả lời vấn đáp "vì sao giải thích này trung thực" bằng số liệu fidelity, không chỉ bằng lời.

## Nguồn tham khảo

- Jacovi & Goldberg, "Towards Faithfully Interpretable NLP Systems", ACL (2020): https://arxiv.org/abs/2004.03685
- Ying et al., "GNNExplainer", NeurIPS (2019): https://arxiv.org/abs/1903.03894
- Sundararajan et al., "Axiomatic Attribution for Deep Networks" (Integrated Gradients), ICML (2017): https://arxiv.org/abs/1703.01365
- Amara et al., "GraphFramEx: Towards Systematic Evaluation of Explainability Methods for GNNs" (2022): https://arxiv.org/abs/2206.09677
- Đề bài, §2, §5 (cấp độ 3), §7.

## Lịch sử sửa đổi

- **2026-11-07:** bản đầu yêu cầu scorer tự trả về bản ghi đóng góp và `explain/` chỉ đọc. Yêu cầu này chặt hơn đề (PDF §2 chỉ cấm giải thích tách rời khỏi mô hình) và buộc module Graph dùng scorer phân rã được. Sửa mục 1–3 của Quyết định, bảng phương án và Hệ quả; giữ nguyên mục 4–6. Nhóm sửa trực tiếp ADR này thay vì viết ADR mới. Phân tích: `docs/research/xai-method-proposal.md`.
