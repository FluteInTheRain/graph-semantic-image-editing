# ADR-0015: MMKG tiếng Việt lưu dạng CSV trong `mmkg/`, data card trong `data_cards/`

- **Trạng thái:** Đã chấp nhận
- **Ngày:** 2026-11-01

## Bối cảnh

Cấp độ 3 (§5) yêu cầu **tự xây một MMKG nhỏ** cho một miền hẹp, ưu tiên tiếng Việt: 50–150 thực thể, 300–1.000 triple, 200–500 ảnh; mô tả nguồn, cách liên kết ảnh - thực thể, khả năng chuyển giao. §8 yêu cầu **data card**, nguồn ảnh/văn bản, cách tạo triple, điều kiện sử dụng. §9: ghi nguồn, tuân thủ giấy phép, không thu thập dữ liệu cá nhân nhạy cảm.

**Data card** = trang mô tả bộ dữ liệu (nguồn, giấy phép, thống kê, hạn chế) theo tinh thần "Datasheets for Datasets".

## Quyết định

Cấu trúc:

| File | Cột | Commit |
|---|---|---|
| `mmkg/entities.csv` | `entity_id, name_vi, name_en, type, source_url` | có |
| `mmkg/triples.csv` | `head_id, relation, tail_id, source` | có |
| `mmkg/images.csv` | `image_id, entity_id, file_name, source_url, license, author` | có |
| `mmkg/images/` | file ảnh | **không** (gitignore, link cloud) |
| `data_cards/mmkg_vi.md` | data card | có |
| `data_cards/<dataset>.md` | card cho bộ công khai đã dùng | có |

Quy tắc:
1. CSV UTF-8, mỗi dòng một bản ghi → review được trong PR, diff rõ ai thêm gì.
2. Quan hệ (`relation`) dùng tên tiếng Anh snake_case cố định (vd. `located_in`, `made_from`), có bảng giải nghĩa tiếng Việt trong data card.
3. Chỉ dùng ảnh có giấy phép cho phép (ưu tiên Wikimedia Commons CC-BY/CC-BY-SA, CC0); ghi `license` + `author` từng ảnh.
4. Không ảnh có mặt người nhận diện được nếu không cần thiết; không dữ liệu cá nhân.
5. MMKG tự xây có split riêng trong `splits/mmkg_vi/` theo cùng quy tắc ADR-0009.
6. Loader trong `src/mmxai/data/` chuyển MMKG về cùng định dạng đồ thị với bộ công khai, để chạy lại "phương pháp tốt nhất" không cần sửa code.

Mục bắt buộc trong data card: Mục đích & miền · Nguồn văn bản/ảnh · Giấy phép & điều kiện sử dụng · Cách tạo triple (thủ công/bán tự động, ai kiểm tra) · Cách liên kết ảnh - thực thể · Thống kê · Split · Hạn chế, thiên lệch · Người tạo & ngày.

## Các phương án đã cân nhắc

| Phương án | Ưu | Nhược | Lý do không chọn |
|---|---|---|---|
| CSV trong git + ảnh qua link | Nhỏ, review được, đúng quy mô | Phải giữ link ảnh | **Được chọn** |
| RDF/Turtle hoặc Neo4j | Chuẩn KG | Thêm công cụ, khó review cho người mới | Quá nặng với 1.000 triple |
| JSON một file lớn | Lồng được | Xung đột merge khi nhiều người cùng thêm | CSV mỗi dòng một bản ghi dễ merge hơn |
| Commit cả ảnh | Đủ bộ trong repo | 200–500 ảnh làm repo phình to | Vi phạm ADR-0008 |
| Data card tự do | Ít ràng buộc | Dễ thiếu mục giảng viên hỏi | Dùng danh sách mục bắt buộc |

## Hệ quả

- Việc xây MMKG chia được cho nhiều người (mỗi người một nhóm thực thể), lịch sử commit thể hiện rõ.
- Cần thêm test kiểm tra: mọi `head_id/tail_id` có trong `entities.csv`, mọi ảnh có `license`.

## Nguồn tham khảo

- Gebru et al., "Datasheets for Datasets", CACM (2021): https://arxiv.org/abs/1803.09010
- Pushkarna et al., "Data Cards", FAccT (2022): https://arxiv.org/abs/2204.01075
- Wikimedia Commons licensing: https://commons.wikimedia.org/wiki/Commons:Licensing
- Đề bài, §5 (cấp độ 3), §8, §9.
