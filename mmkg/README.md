# mmkg/ — MMKG tiếng Việt tự xây (cấp độ 3)

**MMKG là gì?** Đồ thị tri thức đa phương thức: các **thực thể** (vd. "phở", "Hà Nội") nối với nhau bằng **triple** (đầu, quan hệ, đuôi) và mỗi thực thể có **ảnh** minh hoạ.

Quy mô mục tiêu theo đề: 50–150 thực thể, 300–1.000 triple, 200–500 ảnh, một miền hẹp.

| File | Nội dung | Commit? |
|---|---|---|
| `mmkg/entities.csv` | `entity_id, name_vi, name_en, type, source_url` | Có |
| `mmkg/triples.csv` | `head_id, relation, tail_id, source` | Có |
| `mmkg/images.csv` | `image_id, entity_id, file_name, source_url, license, author` | Có |
| `mmkg/images/` | File ảnh thật | **Không** (gitignore, tải từ link cloud) |

- Data card bắt buộc: `data_cards/mmkg_vi.md` (nguồn, giấy phép, cách tạo triple, cách liên kết ảnh - thực thể).
- Mỗi ảnh phải ghi nguồn và giấy phép; không thu thập dữ liệu cá nhân nhạy cảm.
- Chi tiết: `docs/decisions/0015-mmkg-tieng-viet-va-data-card.md`.
