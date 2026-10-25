---
paths:
  - "tests/**"
---

# Quy tắc viết test

- Chạy bằng `uv run pytest`. Test phải nhanh (< vài giây mỗi file), không cần GPU, không tải mạng,
  không cần `data/`. Dùng dữ liệu tổng hợp nhỏ (vài ảnh giả, đồ thị 5–10 node) tạo ngay trong test.
- Mỗi test kiểm **một** ý, tên nói rõ ý đó (`test_statistical_edges_ignore_test_images`).
- Ưu tiên test bất biến quan trọng của đề: không rò rỉ split, metric đúng trên ví dụ tính tay
  (Recall@K, MRR), `alpha=1` cho đúng thứ hạng CLIP thuần, fidelity của giải thích.
- Cố định seed trong test có ngẫu nhiên. Không sửa test cho "xanh" khi code sai – sửa code.
