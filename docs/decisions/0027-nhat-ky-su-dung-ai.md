# ADR-0027: Ghi nhật ký dùng công cụ AI trong `docs/ai-usage-log.md`

- **Trạng thái:** Đề xuất
- **Ngày:** 2026-11-01

## Bối cảnh

PDF §9: được dùng công cụ AI nhưng phải **khai báo trong báo cáo**, ghi rõ **phần đã chỉnh sửa**, và
**tự giải thích được toàn bộ nội dung khi vấn đáp**. Điểm cá nhân dựa trên vấn đáp và minh chứng đóng
góp (lịch sử commit, bảng phân công). Nếu chỉ nhớ lại lúc viết báo cáo cuối kỳ thì sẽ sót và không
chính xác.

## Quyết định

- File `docs/ai-usage-log.md` với bảng: Ngày | Thành viên | Công cụ | Phần việc AI hỗ trợ | Phần đã
  chỉnh sửa | Đã tự kiểm tra/hiểu chưa.
- Mỗi phần việc đáng kể có AI hỗ trợ = một dòng, ghi ngay khi làm (skill `/log-ai-usage` hỗ trợ).
- Cột cuối chỉ thành viên tự điền `Rồi`; AI luôn ghi `Chưa`.
- Tên thành viên lấy từ `CLAUDE.local.md` (mỗi người tự tạo, gitignored) hoặc `git config user.name`.

## Các phương án đã cân nhắc

| Phương án | Ưu | Nhược | Lý do không chọn |
|---|---|---|---|
| File Markdown trong repo, ghi liên tục | Có lịch sử git, gần code, dễ tổng hợp vào báo cáo | Cần kỷ luật | **Được chọn** |
| Trailer `Co-Authored-By` trong commit | Tự động | Không ghi được "phần đã chỉnh sửa" và "đã hiểu chưa" | Thiếu thông tin §9 cần; dùng bổ sung |
| Chỉ viết mục khai báo lúc làm báo cáo | Ít việc | Quên, thiếu chính xác | Rủi ro khi vấn đáp |
| Hook Stop tự ghi log mỗi lượt | Không sót | Rất nhiều dòng rác, không phân biệt việc đáng kể | Gây nhiễu |

## Hệ quả

- Mục "Khai báo sử dụng AI" trong báo cáo được tổng hợp trực tiếp từ file này.
- Mỗi thành viên có danh sách phần cần tự ôn trước vấn đáp (các dòng còn `Chưa`).

## Nguồn tham khảo

- Đề bài `Project03_Multimodal_XAI.pdf` §9.
- How Claude remembers your project (`CLAUDE.local.md`): https://code.claude.com/docs/en/memory
