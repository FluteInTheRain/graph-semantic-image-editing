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

- Hàm tính điểm đồ thị nên trả về cả điểm lẫn thành phần đóng góp (ví dụ: danh sách path + đóng góp
  của từng path, hoặc mask/attention trên cạnh). Giải thích được đọc ra từ đó, không tính lại bằng
  một heuristic riêng (ví dụ "shortest path tìm sau khi đã xếp hạng" là sai nếu path đó không dùng khi tính điểm).
- **Fidelity**: xoá (hoặc làm nhiễu) các cạnh trong giải thích → tính lại điểm/thứ hạng → điểm phải
  giảm / thứ hạng phải tụt. So với baseline xoá cùng số cạnh **ngẫu nhiên**.
- **Validity**: path trong giải thích phải tồn tại trong đồ thị đã dùng khi chấm (không có cạnh
  "ảo"), và nối đúng truy vấn với ứng viên.
- **Sparsity**: báo cáo kích thước giải thích (số cạnh/node) – giải thích ngắn mà vẫn fidelity cao là tốt.
- Đồ thị dùng để giải thích phải là đồ thị xây từ train (xem checklist rò rỉ trong `CLAUDE.md`).
- Mỗi hàm explain cần test trên đồ thị nhỏ tự tạo: xoá cạnh giải thích làm điểm đổi; xoá cạnh
  không liên quan thì điểm (gần như) không đổi.
- Đánh giá định lượng trên ≥ 10 truy vấn (cấp độ 3); phân tích 5 truy vấn thành công + 5 thất bại.
- Quyết định gốc: ADR-0016 (scorer trả bản ghi đóng góp; `explain/` chỉ đọc; cấm `networkx.shortest_path`
  để chọn giải thích, chỉ dùng để kiểm validity; fidelity và ablation dùng chung hàm can thiệp).
- Tham khảo cơ chế: GNNExplainer (mask cạnh), attention của GAT, đóng góp theo path. Đổi cơ chế → ADR mới.
