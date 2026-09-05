# ADR-0001: Ghi lại quyết định kiến trúc bằng ADR

- **Trạng thái:** Đã chấp nhận
- **Ngày:** 2026-11-01

## Bối cảnh

**ADR là gì?** ADR (Architecture Decision Record) là một file Markdown ngắn ghi lại **một** quyết định quan trọng: chọn gì, vì sao, đã cân nhắc những gì khác.

Vì sao nhóm cần:
- Đề (§9) cho phép giảng viên hỏi **bất kỳ thành viên nào** về dữ liệu, mô hình, mã nguồn, rò rỉ dữ liệu. Mỗi người phải giải thích được vì sao repo được tổ chức như vậy.
- Nhóm ≤ 5 người, làm song song. Không có văn bản thì quyết định bị quên hoặc bị làm ngược.
- Báo cáo 12–18 trang cần phần "lựa chọn thiết kế"; ADR là nguồn để viết phần đó.

## Quyết định

1. Mỗi quyết định quan trọng = một file `docs/decisions/NNNN-tieu-de-kebab.md` (số 4 chữ số, tăng dần, không dùng lại).
2. Viết bằng tiếng Việt; tên thư viện, lệnh, đường dẫn giữ tiếng Anh.
3. ADR đã chấp nhận **không sửa nội dung**. Muốn đổi: viết ADR mới, đổi trạng thái ADR cũ thành "Bị thay thế bởi ADR-XXXX".
4. Trạng thái hợp lệ: `Đề xuất` → `Đã chấp nhận` | `Bị từ chối` | `Bị thay thế bởi ADR-XXXX`.
5. ADR được thêm qua Pull Request như code, để cả nhóm review.
6. Mục lục ADR nằm ở `docs/decisions/README.md` (do trưởng nhóm duy trì).

### Mẫu (template)

```markdown
# ADR-NNNN: <Tiêu đề ngắn, dạng mệnh lệnh>

- **Trạng thái:** Đề xuất | Đã chấp nhận | Bị từ chối | Bị thay thế bởi ADR-XXXX
- **Ngày:** YYYY-MM-DD

## Bối cảnh
<Vấn đề gì? Ràng buộc nào (đề bài, thời gian, máy móc)? Giải thích khái niệm cho người mới.>

## Quyết định
<Chọn gì, cụ thể đến mức người khác làm theo được.>

## Các phương án đã cân nhắc
| Phương án | Ưu | Nhược | Lý do không chọn |
|---|---|---|---|
| ... | ... | ... | ... (ghi "Được chọn" cho phương án đã chọn) |

## Hệ quả
<Điều gì dễ hơn, điều gì khó hơn, việc cần làm tiếp theo.>

## Nguồn tham khảo
<Link tài liệu, bài báo, mục trong đề bài.>
```

## Các phương án đã cân nhắc

| Phương án | Ưu | Nhược | Lý do không chọn |
|---|---|---|---|
| ADR Markdown trong repo | Đi cùng code, có lịch sử git, review qua PR | Phải giữ kỷ luật viết | **Được chọn** |
| Google Docs / Notion chung | Dễ sửa cùng lúc | Tách khỏi code, không có trong lịch sử commit được chấm | Không gắn với commit, dễ lệch với code |
| Chỉ ghi trong báo cáo cuối kỳ | Ít việc | Viết lúc cuối thì quên lý do | Mất bối cảnh lúc ra quyết định |
| MADR đầy đủ (tiếng Anh) | Chuẩn phổ biến | Dài, nhiều mục thừa với đồ án | Rút gọn thành mẫu trên, tiếng Việt |

## Hệ quả

- Tốt: ai cũng tra được "vì sao", chuẩn bị vấn đáp nhanh; commit ADR cũng là minh chứng đóng góp.
- Chi phí: mỗi quyết định lớn tốn ~15 phút viết.
- Quy ước số: ADR 0001–0019 cho cấu trúc repo/môi trường, 0020–0039 cho quy trình làm việc với công cụ AI; các ADR sau dùng số tiếp theo còn trống.

## Nguồn tham khảo

- Michael Nygard, "Documenting Architecture Decisions" (2011): https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions
- adr.github.io và MADR: https://adr.github.io/madr/
- Đề bài `Project03_Multimodal_XAI.pdf`, §9 (vấn đáp từng thành viên).
