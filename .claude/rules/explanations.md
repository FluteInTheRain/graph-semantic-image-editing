---
paths:
  - "src/mmxai/explain/**"
  - "src/mmxai/rerank/**"
  - "src/mmxai/models/**"
  - "scripts/explain.py"
  - "tests/**/*explain*"
  - "tests/**/*fidelity*"
---

# Quy tắc cho code giải thích (xAI)

Giải thích **trung thực** = các cạnh/đường đi được hiển thị là thứ đã thực sự tạo ra điểm đồ thị.

- Đóng góp của cạnh được tính **trên chính mô hình đã xếp hạng** bằng explainer đã công bố (Occlusion,
  Integrated Gradients, GNNExplainer; attention nếu là GAT). Không chọn giải thích bằng một heuristic
  riêng (ví dụ "shortest path tìm sau khi đã xếp hạng" là sai nếu path đó không dùng khi tính điểm).
- Với GNN phi tuyến, đóng góp của một cạnh không duy nhất, nên explainer chính và kích thước giải thích
  được **chọn trên val** bằng fidelity, không chọn trên test hay bộ truy vấn đánh giá.
- **Fidelity**: xoá (hoặc làm nhiễu) các cạnh trong giải thích → tính lại điểm/thứ hạng → điểm phải
  giảm / thứ hạng phải tụt. So với baseline xoá cùng số cạnh **ngẫu nhiên**.
- **Validity**: path trong giải thích phải tồn tại trong đồ thị đã dùng khi chấm (không có cạnh
  "ảo"), và nối đúng truy vấn với ứng viên.
- **Sparsity**: báo cáo kích thước giải thích (số cạnh/node) – giải thích ngắn mà vẫn fidelity cao là tốt.
- Đồ thị dùng để giải thích phải là đồ thị xây từ train (xem checklist rò rỉ trong `CLAUDE.md`).
- Mỗi hàm explain cần test trên đồ thị nhỏ tự tạo: xoá cạnh giải thích làm điểm đổi; xoá cạnh
  không liên quan thì điểm (gần như) không đổi.
- Đánh giá định lượng trên ≥ 10 truy vấn (cấp độ 3); phân tích 5 truy vấn thành công + 5 thất bại.
- Quyết định gốc: ADR-0016 (explainer tính trên chính mô hình, chọn trên val bằng fidelity; cấm
  `networkx.shortest_path` để chọn giải thích, chỉ dùng để kiểm validity; fidelity và ablation dùng chung
  hàm can thiệp). Phân tích chi tiết: `docs/research/xai-method-proposal.md`.
