---
paths:
  - "docs/**"
  - "reports/**"
  - "**/README.md"
---

# Quy tắc viết tài liệu

- Viết **tiếng Việt**; giữ nguyên tiếng Anh cho thuật ngữ kỹ thuật, tên thư viện, tên hàm/biến,
  đường dẫn, lệnh. Không viết song ngữ từng dòng.
- Viết cho người mới: giải thích khái niệm bằng lời đơn giản trước, rồi mới tới chi tiết. Ưu tiên
  bảng, sơ đồ, ví dụ; câu ngắn.
- Khi nhắc tới code, ghi đường dẫn (và tên hàm) để người đọc mở ra đối chiếu.
- ADR theo template trong `docs/decisions/0001-ghi-lai-quyet-dinh-kien-truc-bang-adr.md`; ADR đã
  chấp nhận không sửa nội dung, muốn đổi thì viết ADR mới thay thế.
- Số liệu trong báo cáo phải truy được về file trong `results/` và config đã chạy.
