---
name: new-adr
description: Tạo ADR (Architecture Decision Record) mới trong docs/decisions/ theo template ADR-0001 – tự đánh số tiếp theo, đặt tên kebab-case, cập nhật mục lục docs/decisions/README.md nếu có. Dùng khi nhóm chốt một quyết định quan trọng (dữ liệu, cách dựng cạnh, mô hình, metric, dependency, đổi split, quy trình).
argument-hint: "[tiêu đề quyết định]"
allowed-tools: Read, Glob, Grep, Bash(ls *)
---

# Tạo ADR mới

Quyết định cần ghi: **$ARGUMENTS** (nếu trống, hỏi người dùng quyết định là gì).

## Bước 1 – Đọc template và tránh trùng

1. Đọc `docs/decisions/0001-ghi-lai-quyet-dinh-kien-truc-bang-adr.md`: template nằm trong mục "Mẫu (template)".
2. Liệt kê `docs/decisions/` (`ls docs/decisions`). Nếu đã có ADR cùng chủ đề: hỏi người dùng có muốn
   viết ADR **thay thế** không (ADR đã chấp nhận không được sửa nội dung, chỉ đổi dòng Trạng thái
   thành `Bị thay thế bởi ADR-NNNN`).

## Bước 2 – Đánh số và đặt tên

- Số = số lớn nhất đang có + 1, 4 chữ số (vd. có `0026-...` thì dùng `0027`). Không dùng lại số cũ.
  Ngoại lệ: dải 0001–0019 dành cho cấu trúc repo/môi trường, 0020–0039 cho quy trình công cụ AI
  (quy ước trong ADR-0001); nếu quyết định thuộc một dải còn trống thì lấy số trống tiếp theo trong dải đó.
- Tên file: `docs/decisions/NNNN-tieu-de-kebab.md` – tiếng Việt không dấu, chữ thường, nối bằng `-`.

## Bước 3 – Viết nội dung (tiếng Việt, thuật ngữ/code giữ tiếng Anh)

Đủ các mục theo template, đúng thứ tự:

- `# ADR-NNNN: <Tiêu đề ngắn, dạng mệnh lệnh>`
- `- **Trạng thái:** Đề xuất` (chỉ ghi `Đã chấp nhận` khi người dùng xác nhận nhóm đã đồng ý)
- `- **Ngày:** <hôm nay, YYYY-MM-DD>`
- `## Bối cảnh` – vấn đề, ràng buộc (dẫn mục PDF nếu có), giải thích khái niệm cho người mới.
- `## Quyết định` – cụ thể đến mức người khác làm theo được.
- `## Các phương án đã cân nhắc` – bảng `| Phương án | Ưu | Nhược | Lý do không chọn |`, ghi
  **Được chọn** cho phương án đã chọn. Ít nhất 2 phương án.
- `## Hệ quả` – cái gì dễ hơn, khó hơn, việc cần làm tiếp.
- `## Nguồn tham khảo` – chỉ URL/tài liệu đã thực sự đọc, và mục PDF liên quan.

Nếu quyết định ảnh hưởng tới rò rỉ dữ liệu hay tính trung thực của giải thích, nói rõ trong Hệ quả.

## Bước 4 – Mục lục

Nếu `docs/decisions/README.md` tồn tại: thêm một dòng cho ADR mới theo đúng định dạng các dòng đang
có. Nếu không tồn tại: **không tạo**, chỉ nhắc người dùng (mục lục do trưởng nhóm duy trì).

## Bước 5 – Báo lại

In đường dẫn file đã tạo, tóm tắt quyết định một câu, và nhắc: ADR nên vào repo qua Pull Request để cả nhóm review.
