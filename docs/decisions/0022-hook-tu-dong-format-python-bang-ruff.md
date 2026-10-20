# ADR-0022: Hook PostToolUse tự format file Python bằng ruff sau khi AI sửa

- **Trạng thái:** Đề xuất
- **Ngày:** 2026-11-01

## Bối cảnh

**Hook là gì?** Là lệnh shell mà Claude Code tự chạy tại một thời điểm cố định, ví dụ `PostToolUse`
= ngay sau khi AI dùng một công cụ (Edit, Write...). Hook nhận dữ liệu JSON qua stdin (có
`tool_name`, `tool_input.file_path`...). Khác với CLAUDE.md, hook **luôn** chạy, không phụ thuộc AI có nhớ hay không.

Repo dùng ruff cho lint + format (ADR-0004). Nếu AI quên format, diff bị lẫn thay đổi khoảng trắng,
khó review và pre-commit sẽ fail.

## Quyết định

- `.claude/hooks/format-python.sh`, đăng ký `PostToolUse` với matcher `Edit|Write`, timeout 30 giây.
- Chỉ xử lý **đúng file `.py` vừa sửa**: `ruff check --fix` → `ruff format` → `ruff check` (chỉ đọc)
  để lấy lỗi còn lại.
- Dùng ruff trong `.venv/` của dự án (cùng phiên bản với `uv run ruff`), nếu không có thì `ruff` trên
  PATH, nếu vẫn không có thì **bỏ qua im lặng**. Không gọi `uv run` vì có thể kích hoạt sync chậm.
- Luôn `exit 0`: hook không bao giờ chặn công việc. Lỗi lint không tự sửa được (ví dụ `F821` tên chưa
  định nghĩa) được gửi lại cho AI qua `hookSpecificOutput.additionalContext` để AI tự sửa.
- Đọc JSON bằng `jq`, nếu thiếu thì `python3`; thiếu cả hai thì bỏ qua.

## Các phương án đã cân nhắc

| Phương án | Ưu | Nhược | Lý do không chọn |
|---|---|---|---|
| Hook format từng file vừa sửa | Nhanh (~0,1–0,7 s), luôn sạch | Thêm một script phải bảo trì | **Được chọn** |
| Chỉ nhắc trong CLAUDE.md "nhớ chạy ruff" | Không cần script | AI có thể quên | Không đảm bảo |
| Hook chạy `ruff` cho cả repo | Bắt được mọi file | Chậm, sửa cả file người khác đang làm | Phạm vi quá rộng |
| Chặn (`decision: block`) khi còn lỗi lint | Ép sửa ngay | Gây gián đoạn, hiện như lỗi hook | `additionalContext` đủ để AI sửa |
| Chỉ dựa vào pre-commit | Có sẵn | Lỗi phát hiện muộn, lúc commit | Dùng song song, không thay thế |

## Hệ quả

- File `.py` do AI sửa luôn đúng format theo `pyproject.toml`.
- Trước khi báo "xong", AI vẫn phải chạy `uv run ruff check` và `uv run pytest` cho toàn repo.
- Notebook (`.ipynb`) không được hook xử lý; dựa vào pre-commit/nbstripout.
- Kết quả kiểm thử hook (output thật) ghi trong báo cáo setup và có thể chạy lại bằng JSON mẫu.

## Nguồn tham khảo

- Hooks reference (input JSON, exit code, `additionalContext`, matcher, timeout): https://code.claude.com/docs/en/hooks
- Ruff: https://docs.astral.sh/ruff/
