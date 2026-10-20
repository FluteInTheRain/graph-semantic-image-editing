# ADR-0023: Hook bảo vệ file split/query đã commit và chặn ghi file lớn/nhị phân

- **Trạng thái:** Đề xuất
- **Ngày:** 2026-11-01

## Bối cảnh

PDF §6 yêu cầu: mọi mô hình dùng **cùng một file chia** train/val/test, và bộ ≥ 20 truy vấn đánh giá
cấp độ 2 **chỉ để đánh giá**. Nếu AI "tiện tay" sửa một file trong `splits/` hay `queries/` (ví dụ
thêm truy vấn cho kết quả đẹp hơn), toàn bộ so sánh mất giá trị – và rất khó phát hiện khi review.

Ngoài ra, checkpoint (`*.pt`), đặc trưng (`*.npy`), index FAISS không được vào git (nặng, không tái lập).

Hook `PreToolUse` chạy **trước** khi AI sửa file và có thể từ chối bằng JSON
`{"hookSpecificOutput": {"permissionDecision": "deny", ...}}`. Đây là cơ chế cưỡng chế, khác với
CLAUDE.md chỉ là lời dặn.

## Quyết định

1. `.claude/hooks/guard-protected-files.sh` – `PreToolUse`, matcher `Edit|Write`:
   - Từ chối sửa file **đã được git theo dõi** dưới `splits/` hoặc `queries/`. Tạo file **mới** (chưa
     commit) vẫn được, để lần đầu sinh split.
   - Ghi đè có chủ đích: người dùng tự khởi động Claude với biến môi trường
     `MMXAI_ALLOW_PROTECTED_EDIT=1` và viết ADR giải thích.
   - Từ chối ghi file đuôi model/dữ liệu (`.pt .pth .ckpt .safetensors .bin .onnx .npy .npz .pkl
     .faiss .index .h5 .parquet .zip .tar .gz`...) hoặc nội dung > 1 MiB **vào đường dẫn không bị
     gitignore**. Đường dẫn đã gitignore (`data/`, `results/runs/`...) được phép, kiểm bằng `git check-ignore`.
   - Thiếu `jq`/`python3` thì "fail open": báo lỗi không chặn (exit 1), công việc vẫn tiếp tục.
2. `.claude/hooks/stop-check-protected.sh` – `Stop`: khi AI kết thúc lượt, nếu file split/query đã
   commit bị thay đổi (ví dụ qua lệnh Bash, mà hook PreToolUse không thấy), hiện **một** cảnh báo
   (`systemMessage`) cho người dùng – mỗi thay đổi chỉ cảnh báo một lần mỗi phiên, không chặn.

## Các phương án đã cân nhắc

| Phương án | Ưu | Nhược | Lý do không chọn |
|---|---|---|---|
| Hook PreToolUse + nhắc ở Stop | Chặn cứng với Edit/Write, nhắc khi Bash sửa | Hai script nhỏ | **Được chọn** |
| Luật `deny` `Edit(./splits/**)` | Không cần script | Chặn cả tạo file mới; không có cờ ghi đè | Quá cứng lúc sinh split lần đầu |
| Chỉ dặn trong CLAUDE.md | Đơn giản | Không đảm bảo | Rủi ro quá lớn với §6 |
| Stop hook chặn (exit 2) đến khi khôi phục | Mạnh | Gây vòng lặp khó chịu khi thay đổi là cố ý | Cảnh báo là đủ |
| Kiểm tra bằng test/CI (hash file split) | Bắt cả thay đổi thủ công | Phát hiện muộn | Có thể bổ sung sau, không thay thế |

## Hệ quả

- Không bảo vệ được thay đổi qua Bash ở thời điểm thực thi (chỉ cảnh báo sau). Review PR vẫn phải xem `splits/`, `queries/`.
- Biến `MMXAI_ALLOW_PROTECTED_EDIT` phải đặt **trước khi** mở Claude Code (hook kế thừa môi trường của Claude Code).
- Đã kiểm thử với 11 kịch bản JSON mẫu (chặn/cho phép/fallback) – xem báo cáo setup.

## Nguồn tham khảo

- Hooks reference (PreToolUse decision control, Stop input, `systemMessage`, exit code): https://code.claude.com/docs/en/hooks
- Đề bài `Project03_Multimodal_XAI.pdf` §6, §8.
