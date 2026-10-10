# ADR-0016: Giải thích phải sinh ra từ chính graph scorer (faithful by construction)

- **Trạng thái:** Đã chấp nhận
- **Ngày:** 2026-11-01

## Bối cảnh

Đề (§2) có yêu cầu xuyên suốt: "giải thích phải đến từ chính cơ chế xếp hạng, không được sinh một đường đi hợp lý sau khi đã có kết quả mà đường đi đó không ảnh hưởng tới dự đoán". Cấp độ 3 đo định lượng:

| Thước đo | Ý nghĩa đơn giản |
|---|---|
| **Fidelity** (trung thực) | Bỏ/làm nhiễu các cạnh trong giải thích → điểm/thứ hạng phải giảm rõ |
| **Validity** (hợp lệ) | Đường đi có thật trong đồ thị và nối được truy vấn với ứng viên |
| **Sparsity** (gọn) | Giải thích dùng ít cạnh so với toàn bộ đồ thị con |

Đây là quyết định kiến trúc vì nó ràng buộc **giao diện giữa `models/`, `rerank/` và `explain/`**: nếu scorer chỉ trả về một con số, `explain/` buộc phải đoán sau → vi phạm đề.

## Quyết định

1. Graph scorer (`src/mmxai/models/`) trả về **điểm kèm bản ghi đóng góp**: danh sách cạnh/đường đi đã dùng và phần điểm mỗi cái đóng góp (ví dụ đóng góp theo đường đi khi điểm là tổng trên các path, hoặc attribution theo cạnh trên đồ thị con k-hop của ứng viên).
2. `src/mmxai/rerank/` truyền nguyên bản ghi đóng góp tới kết quả cuối; `final = alpha·cosine + (1−alpha)·graph`.
3. `src/mmxai/explain/` **chỉ đọc** bản ghi đóng góp để hiển thị; không tự tìm đường đi mới. Cấm dùng `networkx.shortest_path` để "chọn" giải thích; chỉ được dùng để kiểm tra validity.
4. Fidelity đo bằng **can thiệp thật**: xoá các cạnh giải thích khỏi đồ thị → chạy lại scorer → so điểm/thứ hạng; so với xoá cùng số cạnh ngẫu nhiên (đối chứng).
5. Mọi bảng explain báo cáo trên ≥ 10 truy vấn, kèm đối chứng ngẫu nhiên, ≥ 3 seed.
6. Ablation đồ thị (bỏ cạnh / xáo cạnh / đồ thị ngẫu nhiên, §7) dùng chung hàm can thiệp với fidelity.

## Các phương án đã cân nhắc

| Phương án | Ưu | Nhược | Lý do không chọn |
|---|---|---|---|
| Scorer trả kèm đóng góp (faithful by construction) | Đúng yêu cầu §2, đo fidelity trực tiếp | Giới hạn cách thiết kế scorer | **Được chọn** |
| Post-hoc: tìm đường đi ngắn nhất sau khi xếp hạng | Dễ làm | Đường đi có thể không ảnh hưởng điểm → vi phạm đề | Bị đề cấm |
| GNNExplainer/mask học riêng | Phổ biến | Tối ưu hoá riêng, vẫn là xấp xỉ; tốn thời gian | Dùng làm phương án so sánh (mở rộng), không làm chính |
| Attention weights của GAT làm giải thích | Có sẵn | Attention không chắc phản ánh tầm quan trọng | Chỉ dùng kèm kiểm tra fidelity |

## Hệ quả

- Kiểu dữ liệu kết quả (dataclass chứa `cosine`, `graph`, `final`, `contributions`) phải được thống nhất sớm, trước khi chia việc `models/` và `explain/`.
- Có thể trả lời vấn đáp "vì sao giải thích này trung thực" bằng số liệu fidelity, không chỉ bằng lời.

## Nguồn tham khảo

- Jacovi & Goldberg, "Towards Faithfully Interpretable NLP Systems", ACL (2020): https://arxiv.org/abs/2004.03685
- Ying et al., "GNNExplainer", NeurIPS (2019): https://arxiv.org/abs/1903.03894
- Amara et al., "GraphFramEx: Towards Systematic Evaluation of Explainability Methods for GNNs" (2022): https://arxiv.org/abs/2206.09677
- Đề bài, §2, §5 (cấp độ 3), §7.
