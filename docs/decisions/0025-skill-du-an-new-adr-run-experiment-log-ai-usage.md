# ADR-0025: Ba skill dự án – `/new-adr`, `/run-experiment`, `/log-ai-usage`

- **Trạng thái:** Đề xuất
- **Ngày:** 2026-11-01

## Bối cảnh

**Skill là gì?** Là một quy trình nhiều bước đóng gói trong `.claude/skills/<tên>/SKILL.md`
(frontmatter `name`, `description`, ...). Gõ `/<tên>` để gọi; Claude cũng có thể tự gọi khi
`description` khớp việc đang làm (trừ khi đặt `disable-model-invocation: true`). Nội dung skill chỉ
nạp khi dùng, nên không tốn context mỗi phiên như CLAUDE.md.

Ba quy trình lặp lại nhiều lần và dễ làm sai: viết ADR đúng template, chạy thực nghiệm đủ seed và
truy vết được, khai báo dùng AI (PDF §9).

## Quyết định

| Skill | Làm gì | Ai gọi |
|---|---|---|
| `new-adr` | Đọc template ADR-0001, đánh số tiếp theo, tên kebab-case không dấu, trạng thái `Đề xuất`, cập nhật `docs/decisions/README.md` nếu file đó tồn tại (không tự tạo) | Người dùng hoặc Claude |
| `run-experiment` | Kiểm checklist rò rỉ + split + git sạch → chạy ≥ 3 seed → output vào `results/runs/` (kèm config, commit, seed) → bảng mean ± std vào `results/tables/` | **Chỉ người dùng** (`disable-model-invocation: true`, vì chạy lâu/tốn GPU) |
| `log-ai-usage` | Thêm một dòng vào `docs/ai-usage-log.md`; cột "đã hiểu chưa" luôn `Chưa` | Người dùng hoặc Claude |

`run-experiment` **không** tự viết pipeline: nếu script trong `scripts/` chưa có thì dừng và báo
(phù hợp giai đoạn hiện tại – cài đặt từng bước).

## Các phương án đã cân nhắc

| Phương án | Ưu | Nhược | Lý do không chọn |
|---|---|---|---|
| 3 skill nhỏ, mỗi skill một quy trình | Dễ đọc, dễ sửa, nạp khi cần | Thêm file | **Được chọn** |
| Viết quy trình vào CLAUDE.md | Một chỗ | Tốn context mọi phiên, CLAUDE.md dài | Trái khuyến nghị tài liệu |
| Script Python tự động hoá hoàn toàn | Chạy không cần AI | Phải viết code khi chưa có pipeline | Để sau khi có `scripts/` |
| Dùng skill chung của plugin | Không phải viết | Không biết template ADR và quy tắc §6 của nhóm | Không bám đề |

## Hệ quả

- ADR và nhật ký AI có định dạng thống nhất, ít sót mục.
- Khi `scripts/evaluate.py` được viết, cần cập nhật bước 2 của `run-experiment` theo CLI thật.
- Thành viên vẫn tự điền "Rồi" trong nhật ký AI – skill không làm thay.

## Nguồn tham khảo

- Extend Claude with skills (vị trí, frontmatter, `$ARGUMENTS`, `disable-model-invocation`, giới hạn 500 dòng): https://code.claude.com/docs/en/skills
- Đề bài `Project03_Multimodal_XAI.pdf` §6, §9.
