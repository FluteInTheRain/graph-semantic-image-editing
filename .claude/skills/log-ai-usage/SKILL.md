---
name: log-ai-usage
description: Thêm một dòng vào nhật ký dùng công cụ AI docs/ai-usage-log.md (PDF §9 yêu cầu khai báo). Dùng sau mỗi phần việc đáng kể có AI hỗ trợ (viết/sửa code, tài liệu, phân tích), hoặc khi người dùng gọi /log-ai-usage.
argument-hint: "[mô tả ngắn phần việc]"
allowed-tools: Read, Edit, Bash(git config user.name), Bash(git diff --stat *)
---

# Ghi nhật ký dùng AI

Phần việc: **$ARGUMENTS** (nếu trống, tóm tắt từ những gì vừa làm trong phiên này).

1. Đọc `docs/ai-usage-log.md` để theo đúng định dạng bảng.
2. Xác định thành viên: ưu tiên tên trong `CLAUDE.local.md`; nếu không có thì `git config user.name`;
   vẫn không rõ thì hỏi.
3. Thêm **một dòng** vào cuối bảng:
   - **Ngày**: hôm nay (YYYY-MM-DD).
   - **Thành viên**: như bước 2.
   - **Công cụ**: ví dụ `Claude Code (<tên model>)`; ghi thêm subagent/skill/MCP đã dùng nếu có.
   - **Phần việc AI hỗ trợ**: ngắn, kèm đường dẫn file chính (xem `git diff --stat`).
   - **Phần đã chỉnh sửa**: phần người dùng đã tự sửa/viết lại; nếu chưa có thì ghi `(chưa)`.
   - **Đã tự kiểm tra/hiểu chưa**: luôn ghi `Chưa`. **Không bao giờ** tự ghi `Rồi` thay cho thành viên.
4. Nhắc người dùng: tự đọc lại code, chạy `uv run pytest`, rồi tự sửa cột cuối thành `Rồi` khi đã
   giải thích được toàn bộ phần đó (giảng viên có thể hỏi bất kỳ dòng nào khi vấn đáp).
