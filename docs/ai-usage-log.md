# Nhật ký sử dụng công cụ AI

**Vì sao có file này?** Đề bài (PDF §9) cho phép dùng mã nguồn mở và công cụ AI, nhưng nhóm phải
**khai báo trong báo cáo**, ghi rõ **phần đã chỉnh sửa**, và **mỗi thành viên tự giải thích được toàn
bộ nội dung khi vấn đáp**. File này là nguồn để viết mục "Khai báo sử dụng AI" trong báo cáo cuối kỳ.

## Cách ghi

- Mỗi phần việc đáng kể có AI hỗ trợ (viết/sửa code, viết tài liệu, phân tích kết quả, tìm bài báo) = **một dòng**.
- Có thể dùng skill `/log-ai-usage` trong Claude Code để thêm dòng tự động.
- Cột **Đã tự kiểm tra/hiểu chưa** chỉ do **chính thành viên** điền `Rồi`, sau khi đã đọc lại, chạy
  test và tự giải thích được phần đó. AI luôn ghi `Chưa`.
- Không xoá dòng cũ; nếu sau này viết lại phần đó thì thêm dòng mới ghi rõ.

| Cột | Ý nghĩa |
|---|---|
| Ngày | Ngày làm, dạng YYYY-MM-DD |
| Thành viên | Người chịu trách nhiệm phần việc (người sẽ trả lời khi vấn đáp) |
| Công cụ | Tên công cụ + model, ví dụ `Claude Code (Opus)`, kèm subagent/skill nếu có |
| Phần việc AI hỗ trợ | AI đã làm gì, file nào (đường dẫn) |
| Phần đã chỉnh sửa | Thành viên đã tự sửa/viết lại phần nào so với bản AI đề xuất |
| Đã tự kiểm tra/hiểu chưa | `Chưa` / `Rồi` – chỉ thành viên tự điền `Rồi` |

## Nhật ký

| Ngày | Thành viên | Công cụ | Phần việc AI hỗ trợ | Phần đã chỉnh sửa | Đã tự kiểm tra/hiểu chưa |
|---|---|---|---|---|---|
| 2026-11-01 | (trưởng nhóm) | Claude Code (Opus) | Dựng cấu hình công cụ AI: `CLAUDE.md`, `.claude/` (settings, hooks, rules, agents, skills), `.mcp.json`, ADR-0020 đến ADR-0027, file này | (chưa) | Chưa |
| 2026-11-04 | qphulong | Claude Code (Opus 5.5), skill `superpowers:brainstorming`, subagent builder check | Viết bản thiết kế `docs/superpowers/specs/2026-11-04-clip-embeddings-design.md` và hướng dẫn `docs/guides/chuan-bi-embedding-clip.md`; so sánh bản sao Flickr30k trên Kaggle | (chưa) | Chưa |
| 2026-11-04 | qphulong | Claude Code (Opus 5.5), skill `superpowers:executing-plans`, subagent `leakage-auditor` và code reviewer | Viết code trích embedding CLIP và test: `src/mmxai/utils/config.py`, `src/mmxai/data/{records,flickr30k,visual_genome}.py`, `src/mmxai/features/{cache,verify,clip_encoder,pipeline}.py`, `scripts/extract_features.py`, `scripts/sync_notebook_code.py`, `notebooks/10_prepare_clip_embeddings.ipynb`; ADR-0040, ADR-0041; data card | (chưa) | Chưa |
| 2026-11-04 | qphulong | Claude Code (Opus 5.5) | Kiểm tra embedding tải về từ Kaggle (`verify_features`, CRC, `code_hash`, vector trùng, truy vấn thử trên mẫu); cập nhật kết quả lần chạy vào `docs/guides/chuan-bi-embedding-clip.md`, `data_cards/flickr30k.md`, `data_cards/visual_genome.md` | (chưa) | Chưa |
