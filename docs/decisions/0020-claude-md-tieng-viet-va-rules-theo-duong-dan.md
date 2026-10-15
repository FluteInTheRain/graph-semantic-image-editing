# ADR-0020: Dùng một CLAUDE.md tiếng Việt ngắn gọn và `.claude/rules/` theo đường dẫn

- **Trạng thái:** Đề xuất
- **Ngày:** 2026-11-01

## Bối cảnh

**CLAUDE.md là gì?** Là file Markdown ở gốc repo mà Claude Code tự đọc vào đầu **mọi** phiên làm việc.
Nó giống "bản giao việc" cho trợ lý AI: lệnh chạy, cấu trúc repo, quy tắc bắt buộc.

Vấn đề:
- Đồ án có các quy tắc dễ vi phạm vô tình: rò rỉ dữ liệu (PDF §6), giải thích phải trung thực (PDF §2),
  CLIP đóng băng (PDF §4.1). Nếu AI không biết, nó có thể viết code sai mà trông vẫn chạy.
- Tài liệu chính thức khuyên CLAUDE.md **dưới 200 dòng**: file dài tốn context và AI tuân thủ kém hơn.
  Quy tắc chỉ liên quan một phần code nên chuyển sang **rules theo đường dẫn** (`.claude/rules/*.md`
  có frontmatter `paths:`), chỉ nạp khi AI đọc/sửa file khớp.
- Mỗi thành viên bị vấn đáp riêng (PDF §9) nên code phải đơn giản, đọc được.
- Repo đang ở giai đoạn setup; AI không được tự sinh hàng loạt code.

## Quyết định

1. `CLAUDE.md` ở gốc repo, **tiếng Việt** (lệnh, đường dẫn, tên code giữ tiếng Anh), mục tiêu ≤ 150 dòng.
   Gồm: mô tả một dòng, giai đoạn hiện tại, nguồn sự thật (PDF > README > ADR), lệnh, bố cục, quy ước
   code, checklist rò rỉ, quy tắc giải thích trung thực, tái lập, ADR, khai báo AI, việc cấm làm.
2. Chi tiết theo vùng code đặt ở `.claude/rules/`:
   - `explanations.md` – `src/mmxai/explain/**`, `rerank/**`, `models/**`, test giải thích.
   - `data-and-splits.md` – `splits/**`, `queries/**`, `src/mmxai/data/**`, `src/mmxai/graph/**`, `mmkg/**`.
   - `tests.md` – `tests/**`.
   - `docs.md` – `docs/**`, `reports/**`, `**/README.md`.
3. Ghi chú cá nhân (tên thành viên, máy, GPU) để trong `CLAUDE.local.md` (gitignored).
4. Không dùng `@import` cho README trong CLAUDE.md: README dài (~600 dòng) và import không giảm chi phí context.

## Các phương án đã cân nhắc

| Phương án | Ưu | Nhược | Lý do không chọn |
|---|---|---|---|
| CLAUDE.md ngắn + rules theo đường dẫn | Ít tốn context, quy tắc đúng chỗ | Phải giữ hai nơi đồng bộ | **Được chọn** |
| Một CLAUDE.md dài chứa tất cả | Một chỗ duy nhất | Vượt khuyến nghị 200 dòng, AI tuân thủ kém | Tốn context mọi phiên |
| `@README.md` import trong CLAUDE.md | Không lặp nội dung | README ~600 dòng nạp vào mọi phiên | Quá tốn context |
| CLAUDE.md tiếng Anh | AI hiểu tốt như nhau | Thành viên đọc chậm hơn, lệch quy ước tài liệu tiếng Việt | Nhóm thống nhất tài liệu tiếng Việt |
| `AGENTS.md` thay CLAUDE.md | Dùng chung cho nhiều công cụ AI | Nhóm dùng Claude Code là chính; tính năng rules/hook là của Claude Code | Có thể thêm sau nếu nhóm dùng công cụ khác |

## Hệ quả

- AI nhận checklist rò rỉ và quy tắc giải thích trong mọi phiên; quy tắc chi tiết chỉ nạp khi cần.
- Khi đổi cấu trúc repo hay lệnh, phải cập nhật CLAUDE.md (người review PR kiểm điểm này).
- CLAUDE.md chỉ là hướng dẫn, **không phải cơ chế cưỡng chế**; phần cần chặn cứng dùng hook (ADR-0023).
- Định kỳ chạy `/doctor prompt-audit` (hoặc tự rà) để tìm hướng dẫn mâu thuẫn.

## Nguồn tham khảo

- Claude Code – How Claude remembers your project (CLAUDE.md, kích thước, `.claude/rules/`, `paths`): https://code.claude.com/docs/en/memory
- Đề bài `Project03_Multimodal_XAI.pdf` §2, §4.1, §6, §9.
