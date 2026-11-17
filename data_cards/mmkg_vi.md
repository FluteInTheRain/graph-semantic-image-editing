# Data card: MMKG tiếng Việt – di tích và di sản Việt Nam

**Là gì?** Một đồ thị tri thức đa phương thức (MMKG) nhỏ do nhóm tự xây cho cấp độ 3 (PDF §5). Đỉnh là **thực thể** (di tích, hiện vật, khái niệm như "tượng", "đồng điếu", "Bảo tàng Lịch sử Việt Nam"), cạnh là **triple** (đầu, quan hệ, đuôi), và mỗi di tích có 3–5 **ảnh**. Định dạng theo ADR-0015; miền theo ADR-0050; cách thu thập, lọc và duyệt theo ADR-0051.

**Trạng thái (2026-11-17):** đã thu thập, dựng bản đầu và dựng lại với luật tác giả mới (mục 3), **chưa duyệt**. Chưa được dùng cho thí nghiệm nào. Chỉ dùng khi `scripts/build_mmkg.py --require-reviewed` chạy qua (không còn mục `pending`).

Đường dẫn tính từ gốc repo. Số liệu lấy từ `mmkg/build_report.json` và `mmkg/raw/20261116/meta.json`; số nào đếm thêm từ `mmkg/*.csv` thì ghi rõ.

## Mục lục

1. [Mục đích và miền](#1-mục-đích-và-miền)
2. [Nguồn văn bản và ảnh](#2-nguồn-văn-bản-và-ảnh)
3. [Giấy phép và điều kiện sử dụng](#3-giấy-phép-và-điều-kiện-sử-dụng)
4. [Cách tạo triple](#4-cách-tạo-triple)
5. [Cách liên kết ảnh – thực thể](#5-cách-liên-kết-ảnh--thực-thể)
6. [Thống kê](#6-thống-kê)
7. [Kết quả duyệt](#7-kết-quả-duyệt)
8. [Tải ảnh](#8-tải-ảnh)
9. [Split](#9-split)
10. [Hạn chế, thiên lệch](#10-hạn-chế-thiên-lệch)
11. [Người tạo và ngày](#11-người-tạo-và-ngày)

## 1. Mục đích và miền

- **Mục đích:** chạy lại phương pháp tốt nhất của cấp độ 1/2 trên dữ liệu tiếng Việt và phân tích khả năng chuyển giao (PDF §5).
- **Miền:** mọi thực thể Wikidata có danh hiệu di sản bất kỳ (`P1435`) và quốc gia là Việt Nam (`P17 = Q881`). Gồm cả công trình (chùa, thành, lăng) và hiện vật (bảo vật quốc gia như trống đồng, tượng). Lý do chọn miền: ADR-0050.
- **Hai loại thực thể** (cột `type` trong `mmkg/entities.csv`):

| `type` | Nghĩa | Có ảnh? |
|---|---|---|
| `item` | Di tích/hiện vật của miền, có 3–5 ảnh hợp lệ | Có |
| `concept` | Thực thể được ≥ 3 `item` cùng trỏ tới (ví dụ "lăng", "cát kết", "Bảo tàng Mỹ thuật Việt Nam") | Không |

## 2. Nguồn văn bản và ảnh

| Thành phần | Nguồn | Chi tiết |
|---|---|---|
| Thực thể, nhãn `vi`/`en`, triple | Wikidata, truy vấn SPARQL qua QLever | Endpoint `https://qlever.dev/api/wikidata` (khoá `sparql_url` trong `meta.json`) |
| Phiên bản dữ liệu Wikidata | Chuỗi mô tả index của QLever, ghi nguyên văn ở khoá `sparql_index` | `Complete Wikidata, TTL from https://dumps.wikimedia.org/wikidatawiki/entities (latest-all.ttl.bz2 from 10.08.2026 munged, latest-lexemes.ttl.bz2 from 07.08.2026), updated LIVE` |
| Ảnh và metadata ảnh (giấy phép, tác giả, kích thước, `sha1`) | Wikimedia Commons API | File trong `P18` (ảnh đại diện) và file nằm trực tiếp trong danh mục `P373` của từng thực thể |
| Ngày thu thập | `collected_on` trong `meta.json` | 2026-11-16 |
| Snapshot gốc | Thư mục đã commit | `mmkg/raw/20261116/` (`entities_raw.csv`, `triples_raw.csv`, `image_candidates.csv`, `image_credits.csv`, `meta.json`) |
| Phiên bản code thu thập | `git_commit` trong `meta.json` | `8a75f310259a646d75bbe33c4f33f49d1b9df299` |
| Câu truy vấn đã chạy | Khoá `queries` trong `meta.json` | 4 câu: `domain`, `triples`, `target_labels`, `concept_triples` |

Nếu không đọc được tên bản dump (endpoint lỗi mạng, client đã thử hết lượt và báo `RuntimeError`, JSON sai, thiếu khoá `name-index`), script ghi `sparql_index = unknown` và in cảnh báo, không dừng lần thu thập: đây chỉ là thông tin nguồn gốc (ADR-0051 mục 3, `_sparql_index` trong `src/mmxai/data/mmkg_collect.py`). Tên dump được đọc **trước** mọi câu truy vấn. Khoá `missing_files` của `meta.json` ghi các file Commons không trả thông tin; từ nay mỗi mục là `{name, reason}` (`missing`, `invalid`, `no_imageinfo`, `unresolved_redirect`), riêng snapshot `20261116` thu trước thay đổi này nên chỉ ghi tên file.

Vì sao dùng QLever thay cho Wikidata Query Service (WDQS): ngày 2026-11-16 WDQS trả HTTP 429 yêu cầu chờ 1000 s ngay ở câu truy vấn đầu, và người dùng yêu cầu thu thập nhanh nhất (ADR-0051). QLever chạy trên bản dump Wikidata nên dữ liệu có thể lệch Wikidata trực tiếp (mục 10).

Lệnh thu thập (chạy một lần; script từ chối chạy lại trong cùng ngày):

```bash
uv run python scripts/collect_mmkg.py --config configs/mmkg_vi_collect.yaml
```

Script gửi User-Agent `mmxai-course-project/0.1 (research script)`, không chứa thông tin cá nhân. Không thu thập dữ liệu cá nhân nào ngoài tên tác giả/người tải lên ảnh, là thông tin ghi công mà giấy phép yêu cầu.

## 3. Giấy phép và điều kiện sử dụng

| Phần | Giấy phép | Điều kiện |
|---|---|---|
| Dữ liệu Wikidata (nhãn, triple) | CC0 (https://www.wikidata.org/wiki/Wikidata:Licensing) | Dùng tự do; nhóm vẫn ghi nguồn qua cột `source` |
| Ảnh Commons | Theo từng file, cột `license` trong `mmkg/images.csv` | Chỉ nhận `Public domain`, `CC0`, `CC BY *`, `CC BY-SA *` (mọi phiên bản) |

**Luật ghi công** (lựa chọn của nhóm, `pick_author` trong `src/mmxai/data/mmkg_build.py`):

| Giấy phép | `Artist` nêu một người/tổ chức | `Artist` rỗng hoặc không nêu ai |
|---|---|---|
| CC BY, CC BY-SA | Ghi lại (đã chuẩn hoá) | Ghi `uploader: <tên người tải lên>`; nếu cũng rỗng thì **bỏ ảnh** |
| Public domain, CC0 | Ghi lại (đã chuẩn hoá) | Ghi `unknown` |

**Vấn đề tác giả ở lần dựng đầu.** Commons trả ô `Artist` dưới dạng HTML. Với ảnh không rõ tác giả, Commons đặt chữ "Unknown author" trong hai thẻ liền nhau; bộ thu thập cũ xoá thẻ mà không chèn dấu cách nên được `Unknown authorUnknown author`. Một ảnh khác có câu mẫu giấy phép ("... the copyright holder of this work, hereby publishes it under the following license:") nằm trong ô tác giả. Lần dựng đầu có **11 ảnh** mang tác giả hỏng như vậy, **6 ảnh** trong đó là CC BY-SA 3.0, tức là không ghi công đúng giấy phép.

**Luật sửa** (`normalize_artist`, `names_an_author`, `pick_author` trong `src/mmxai/data/mmkg_build.py`; ADR-0051 mục 2):

1. Làm sạch chữ `Artist`: bỏ ký tự rộng 0, gộp một cụm bị lặp liền nhau (`Unknown authorUnknown author.` → `Unknown author`), bỏ dấu câu ở cuối (`Chainwit.` → `Chainwit`).
2. Chữ (không phân biệt hoa thường) là `unknown`, `unknown author`, `author unknown`, `anonymous`, hoặc chứa `hereby publish`, thì coi như không nêu ai và áp cột phải của bảng trên.
3. Bộ thu thập (`strip_html` trong `src/mmxai/data/mmkg_collect.py`) nay thay mỗi thẻ HTML bằng dấu cách, nên lần thu thập sau không còn dính chữ. Snapshot `20261116` không bị sửa; luật 1–2 xử lý nó ở bước dựng.

Trước/sau (đếm từ `mmkg/images.csv` trước và sau commit `c6d7a97`):

| Mục | Trước | Sau |
|---|---|---|
| Ảnh có tác giả hỏng | 11 | 0 |
| Ảnh ghi `uploader: ...` (đều CC BY-SA 3.0) | 1 | 7 |
| Ảnh ghi `unknown` (6 Public domain, 1 CC0) | 1 | 7 |
| Ảnh bị bỏ vì `no_author` | 0 | 0 |
| Tổng số ảnh | 403 | 403 (cùng `image_id`, chỉ cột `author` đổi ở 36 dòng, phần lớn là bỏ dấu chấm cuối) |

**Cần nhóm kiểm tay:** ảnh `Bia Vinh Lang.JPG` (`Q10739843_3e40239b`, CC BY-SA 3.0) nay ghi `uploader: Vmenkov`, nhưng trang mô tả trên Commons ghi người giữ bản quyền là `Rungbachduong` (thành viên Wikipedia tiếng Việt). Người duyệt mở `source_url`, xác nhận tên cần ghi công; nếu cần thì `drop` ảnh với lý do `other` hoặc báo để sửa luật.

Phân bố giấy phép của 403 ảnh (đếm từ `mmkg/images.csv`):

| `license` | Số ảnh |
|---|---|
| `CC0` | 85 |
| `CC BY-SA 3.0` | 70 |
| `Public domain` | 68 |
| `CC BY 2.0` | 68 |
| `CC BY-SA 4.0` | 66 |
| `CC BY 4.0` | 23 |
| `CC BY-SA 2.0` | 17 |
| `CC BY 3.0` | 3 |
| `CC BY-SA 2.5` | 2 |
| `CC BY 2.5` | 1 |

Điều kiện khi dùng hoặc đăng lại:

- Ảnh CC BY / CC BY-SA: phải ghi tác giả (`author`), giấy phép (`license`) và link trang file (`source_url`). Ảnh CC BY-SA sửa đổi phải giữ cùng giấy phép.
- Ảnh **không** nằm trong git (`mmkg/images/` đã gitignore, ADR-0008); chia sẻ qua link cloud của nhóm (chưa có link, sẽ điền sau khi duyệt và tải xong). Người nhận kiểm từng file bằng `sha1` trong `mmkg/images_manifest.csv` (mục 8).
- ADR-0015 quy tắc 4: không dùng ảnh có mặt người nhận diện được. Việc này chỉ kiểm được khi duyệt bằng mắt (lý do `identifiable_person`, mục 7).

## 4. Cách tạo triple

Triple được tạo **bán tự động**: script lấy từ Wikidata, nhóm lọc bằng luật trong config, rồi thành viên duyệt từng triple.

1. **Thu thập** (`scripts/collect_mmkg.py`): mọi phát biểu trực tiếp (`wdt:P...`) của thực thể trong miền trỏ tới một thực thể Wikidata khác, cộng mọi triple giữa các thực thể đích đã có (câu `concept_triples`). Giữ nguyên mã `P`.
2. **Dựng** (`scripts/build_mmkg.py`, logic `merge_triples`, `find_concepts`, `keep_triples` trong `src/mmxai/data/mmkg_build.py`): chỉ giữ 18 mã `P` trong `configs/mmkg_vi_build.yaml`, đổi sang 17 tên snake_case, gộp `P276` và `P195` thành `kept_at`, chỉ giữ triple có cả hai đầu là thực thể được giữ.
3. **Duyệt** (mục 7): thành viên đánh dấu `ok`/`drop` từng triple trong `mmkg/review.csv`. **Chưa làm.**

Cột `source` = URL Wikidata của thực thể đầu, nên mọi triple kiểm lại được. Cột thêm `wikidata_property` ghi mã `P` gốc (triple gộp ghi `P195|P276`).

**Bảng quan hệ** (danh sách do Claude Code đề xuất, nhóm **chưa duyệt**; sửa được trong config):

| Tên trong MMKG | Mã Wikidata | Nghĩa tiếng Việt |
|---|---|---|
| `instance_of` | `P31` | là một (loại) |
| `heritage_designation` | `P1435` | có danh hiệu di sản |
| `kept_at` | `P276`, `P195` (gộp) | được lưu giữ / đặt tại |
| `located_in` | `P131` | nằm trong đơn vị hành chính |
| `made_from` | `P186` | làm bằng chất liệu |
| `time_period` | `P2348` | thuộc thời kỳ |
| `culture` | `P2596` | thuộc nền văn hoá |
| `depicts` | `P180` | khắc hoạ |
| `discovered_at` | `P189` | phát hiện tại |
| `part_of` | `P361` | là một phần của |
| `has_part` | `P527` | gồm có |
| `named_after` | `P138` | đặt tên theo |
| `creator` | `P170` | người tạo |
| `commissioned_by` | `P88` | người đặt làm |
| `influenced_by` | `P737` | chịu ảnh hưởng từ |
| `has_pattern` | `P5422` | có hoa văn |
| `significant_event` | `P793` | sự kiện quan trọng |

## 5. Cách liên kết ảnh – thực thể

- Ảnh gắn với thực thể **theo nguồn trên Commons**, không do nhóm hay mô hình đoán: file là `P18` của thực thể, hoặc nằm trực tiếp trong danh mục Commons `P373` của nó (không đi xuống danh mục con).
- File thuộc **hơn một** thực thể bị bỏ, và file trùng `sha1` chỉ giữ một bản, nên mỗi ảnh trong bộ chính thức thuộc đúng một thực thể.
- `image_id` = `<entity_id>_<8 ký tự đầu của sha1>` (`make_image_row`), ổn định khi dựng lại.
- Chỉ `item` có ảnh; `concept` không có ảnh.
- **Lưu ý rò rỉ:** cột `entity_id` của `images.csv` chính là đáp án "ảnh này là di tích nào". Khi dựng đồ thị, cạnh `ảnh → contains → thực thể` cho ảnh val/test không được lấy từ cột này (ADR-0051, Hệ quả). Cách dựng cạnh sẽ chốt ở ADR sau.

## 6. Thống kê

### 6.1. Snapshot gốc → bộ chính thức

| Mục | Snapshot gốc | Bộ chính thức | Mục tiêu (ADR-0050) |
|---|---|---|---|
| Thực thể | 848 (`raw_entities`; gồm thực thể của miền và thực thể đích) | 128 = 84 `item` + 44 `concept` | 50–150 |
| Triple | 3.581 (`raw_triples`) | 331 (`triples`) | 300–1.000 |
| Ảnh | 5.093 cặp (thực thể, file) ứng viên (`raw_image_candidates`) | 403 (`images`) | 200–500 |

Snapshot có 170 thực thể của miền (`in_domain = 1` trong `entities_raw.csv`), 4.979 file ảnh khác nhau (`image_credits` trong `meta.json`). Tất cả 44 `concept` đều là thực thể đích, không phải thực thể của miền (đếm từ `mmkg/entities.csv` và `entities_raw.csv`).

Số ảnh mỗi `item` (đếm từ `mmkg/images.csv`): 72 di tích có 5 ảnh, 7 có 4 ảnh, 5 có 3 ảnh.

Số triple theo quan hệ (đếm từ `mmkg/triples.csv`): `heritage_designation` 89, `instance_of` 69, `kept_at` 47, `made_from` 32, `located_in` 26, `part_of` 13, `culture` 11, `influenced_by` 11, `significant_event` 9, `depicts` 8, `time_period` 7, `named_after` 5, `has_part` 2, `discovered_at` 2. Ba quan hệ trong danh sách không có triple nào: `creator`, `commissioned_by`, `has_pattern`. 29 triple `kept_at` gộp từ cả `P195` lẫn `P276`.

### 6.2. Số dòng bị bỏ theo từng luật

Mọi luật là **lựa chọn của nhóm, không phải yêu cầu của đề**. Số lấy từ `mmkg/build_report.json`, theo đúng thứ tự áp luật.

**Ảnh** (5.093 cặp ứng viên → 403):

| Khoá trong report | Số bị bỏ | Luật | Lý do |
|---|---|---|---|
| `images_dropped_entity_review_drop` | 0 | Thực thể bị duyệt `drop` thì bỏ mọi ảnh của nó | Kết quả duyệt |
| `images_dropped_shared_between_entities` | 228 | File thuộc hơn một thực thể | Không biết ảnh cho thấy thực thể nào |
| `images_dropped_mime` | 41 | Chỉ `image/jpeg`, `image/png`, `image/webp` | CLIP chỉ đọc ảnh raster |
| `images_dropped_license` | 15 | Ngoài `Public domain`, `CC0`, `CC BY *`, `CC BY-SA *` | PDF §9, ADR-0015 quy tắc 3 |
| `images_dropped_too_small` | 6 | Cạnh ngắn < 224 px | Kích thước đầu vào của CLIP ViT-B-32 |
| `images_dropped_no_author` | 0 | CC BY/CC BY-SA mà không có tác giả lẫn người tải lên | Không ghi công được |
| `images_dropped_review_drop` | 0 | Ảnh bị duyệt `drop` | Kết quả duyệt |
| `images_dropped_duplicate_sha1` | 0 | Trùng `sha1`, giữ tên nhỏ nhất | Hai file giống hệt từng byte |
| `images_dropped_below_min_per_item` | 45 | Thực thể còn dưới 3 ảnh hợp lệ thì không thành `item` | Cần đủ ảnh để chia train/val/test sau này |
| `images_dropped_over_max_per_item` | 4.355 | Giữ tối đa 5 ảnh mỗi `item` (`P18` trước, rồi theo tên file) | Giữ quy mô gợi ý; không để di tích nhiều ảnh lấn át |

Kiểm lại: 5.093 − 228 − 41 − 15 − 6 − 45 − 4.355 = 403.

**Triple** (3.581 → 331):

| Khoá trong report | Số bị bỏ | Luật | Lý do |
|---|---|---|---|
| `triples_dropped_unknown_relation` | 1.835 | Mã `P` không có trong danh sách 18 mã `P` (17 tên quan hệ) | Bỏ quan hệ quản trị (quốc gia, bản quyền, danh mục...) |
| `triples_dropped_review` | 0 | Triple bị duyệt `drop` | Kết quả duyệt |
| `triples_dropped_end_not_kept` | 1.320 | Một đầu không phải thực thể được giữ, hoặc tự trỏ về chính nó | Đồ thị không có đầu mút treo |

Hai số trên không cộng thẳng ra 331, vì bước gộp `P276`/`P195` nằm giữa: 3.581 − 1.835 = 1.746 dòng; gộp các dòng cùng (đầu, `kept_at`, đuôi) còn 331 + 1.320 = 1.651 triple; bỏ 1.320 còn 331 (`merge_triples`, `keep_triples`).

**Thực thể:** `entities_dropped_review` = 0. Từ 170 thực thể của miền, 84 có đủ 3 ảnh hợp lệ thành `item`; 44 thực thể đích được ≥ 3 `item` trỏ tới thành `concept` (`min_items_per_concept: 3`, FluteInTheRain chọn 3 thay vì 2 để tổng số thực thể nằm trong 50–150, ADR-0050).

### 6.3. Thiếu nhãn tiếng Việt

`entities_missing_name_vi` = 7 thực thể không có nhãn `vi` trên Wikidata (cột `name_vi` rỗng): `Q120835889` (`item`, "The set of steps of Kinh Thien Palace") và 6 `concept`: `Q106379705`, `Q1459900`, `Q19683138`, `Q29778318`, `Q43113623`, `Q830393`. Người duyệt điền nhãn ở cột `fix_name_vi` của `mmkg/review.csv`.

## 7. Kết quả duyệt

**Trạng thái: chưa duyệt.** Mọi mục trong bộ chính thức đang `pending` (`mmkg/build_report.json`):

| Loại | Số mục chờ duyệt | Khoá trong report |
|---|---|---|
| Thực thể | 128 | `pending_entity` |
| Ảnh | 403 | `pending_image` |
| Triple | 360 | `pending_triple` |

Số triple chờ duyệt (360) lớn hơn số triple (331) vì file duyệt khoá theo mã `P` gốc: mỗi triple `kept_at` gộp từ `P195` và `P276` có hai dòng duyệt (331 + 29 = 360).

Cách duyệt: xem `mmkg/README.md`. Lý do `drop` cho ảnh: `not_the_object`, `identifiable_person`, `duplicate`, `other`.

Bảng điền sau khi duyệt (người duyệt cập nhật, số đếm từ `mmkg/review.csv` và `build_report.json` của lần dựng sau duyệt):

| Loại | Lý do `drop` | Số mục | Người duyệt | Ngày |
|---|---|---|---|---|
| Ảnh | `not_the_object` | (chưa duyệt) | | |
| Ảnh | `identifiable_person` | (chưa duyệt) | | |
| Ảnh | `duplicate` | (chưa duyệt) | | |
| Ảnh | `other` | (chưa duyệt) | | |
| Triple | (ghi lý do) | (chưa duyệt) | | |
| Thực thể | (ghi lý do) | (chưa duyệt) | | |
| Thực thể | sửa `name_vi` (`fix_name_vi`) | (chưa duyệt) | | |

Nếu duyệt bỏ nhiều tới mức triple dưới 300 hoặc thực thể ra ngoài 50–150, nhóm đổi ngưỡng trong `configs/mmkg_vi_build.yaml`, dựng lại và ghi ngưỡng mới, lý do và số mới vào mục này (ADR-0051).

## 8. Tải ảnh

Ảnh được tải dưới dạng **thumbnail rộng 960 px** từ các kích thước Wikimedia Commons đã tạo sẵn (`download_width: 960` trong `configs/mmkg_vi_build.yaml`). Lý do: lần tải bản gốc ngày 2026-11-16 bị Wikimedia chặn tốc độ (HTTP 429, rồi phải chờ 600 s sau 11/403 file); chính sách robot của Wikimedia cho phép tải "bản gốc hoặc một trong các kích thước thumbnail tạo sẵn"; CLIP chỉ cần 224 px nên 960 px không mất gì, và dung lượng giảm từ khoảng 1,1 GB xuống vài chục MB. File lưu ở `mmkg/images/<image_id><đuôi file>` (logic `download_images` trong `src/mmxai/data/mmkg_images.py`).

**Khi nào tải bản gốc thay cho thumbnail.** Script tải **bản gốc** và so `sha1` với cột `sha1` của `mmkg/images.csv` (là `sha1` Commons của bản gốc) trong ba trường hợp. Bản gốc sai `sha1` không được lưu.

| `why` trong manifest | `kind` | Khi nào |
|---|---|---|
| `thumbnail` | `thumbnail` | Bản gốc rộng hơn 960 px và thumbnail 960 px có cạnh ngắn ≥ 224 px (trường hợp thường gặp) |
| `original_not_wider` | `original` | Bản gốc rộng ≤ 960 px. Commons không phóng to ảnh, nên "thumbnail" nó trả về chính là file gốc (URL có `utm_content=thumbnail_unscaled`); tải như bản gốc để kiểm được `sha1`. Theo cột `width` của `images.csv` có 44 ảnh như vậy |
| `original_no_thumburl` | `original` | Commons không trả link thumbnail |
| `original_thumb_too_small` | `original` | Thumbnail 960 px có cạnh ngắn < 224 px (ảnh rất dẹt) |
| `original_already_present` | `original` | File đã có trên đĩa từ trước và khớp `sha1` Commons, nhưng không biết vì sao lúc đó tải bản gốc (ví dụ file của lần tải bản gốc đầu tiên) |

**Manifest `mmkg/images_manifest.csv`.** Commons không có `sha1` cho thumbnail, nên script tự tính `sha1` của file đã tải và ghi vào manifest: `image_id, file_name, kind, why, url, width, height, bytes, sha1`, sắp theo `image_id`. Manifest mô tả đúng các ảnh của `images.csv` **hiện tại** đang có trên đĩa, và được ghi lại **sau từng file**, nên lần chạy bị ngắt giữa chừng không mất dòng nào. Cách kiểm một bản copy trên cloud: chạy `shasum -a 1 <file>` rồi so với cột `sha1` của dòng có `image_id` tương ứng (tên file không kèm đuôi). Cột `sha1` trong `mmkg/images.csv` chỉ khớp với file có `kind = original`.

**Chạy lại.** Mỗi lần chạy, script kiểm file đã có trước khi tải:

- File khớp `sha1` của dòng manifest, hoặc khớp `sha1` Commons, được giữ (`already_present`).
- Dòng manifest cũ ghi `thumbnail` cho ảnh rộng ≤ 960 px được so lại với `sha1` Commons: khớp thì đổi nhãn thành `original_not_wider` (`relabelled_as_original`), không khớp thì tải lại.
- File sai `sha1` bị xoá (in một dòng log, `removed_bad_file`) rồi tải lại.
- **File lạ:** file trong `mmkg/images/` có tên (bỏ đuôi) không phải `image_id` nào của `images.csv`, ví dụ ảnh vừa bị duyệt `drop` hay file `.part` dở. Script liệt kê chúng ra stderr và dừng với mã 1 **trước mọi request**; chạy thêm `--delete-stray` thì script xoá từng file và in tên. Nhờ vậy ảnh bị bỏ (ví dụ `identifiable_person`) không lọt lên bản chia sẻ.
- **Báo lỗi theo tên:** mọi ảnh sai `sha1` (`image_id, file_name, expected_sha1, actual_sha1`) và mọi ảnh Commons không tìm thấy (`image_id, file_name, reason`) được in ra stderr và ghi vào `mmkg/download_report.json`. Còn lỗi thì script thoát mã 1.

```bash
uv run python scripts/download_mmkg_images.py \
    --collect-config configs/mmkg_vi_collect.yaml \
    --build-config configs/mmkg_vi_build.yaml
# thêm --delete-stray nếu script báo file lạ
```

Lần tải đầu bắt đầu ngày 2026-11-16, trước khi duyệt, và xong ngày 2026-11-17 sau nhiều lần chạy lại (bị `upload.wikimedia.org` giới hạn bằng HTTP 429; hết bị chặn khi User-Agent có email liên hệ, xem `configs/mmkg_vi_collect.yaml`). Số liệu của **lần chạy cuối** (in cuối lần chạy, nằm trong `mmkg/download_report.json`; các lần trước được tính vào `already_present`):

| Mục | Giá trị |
|---|---|
| `downloaded_thumbnail` | 50 |
| `downloaded_original` | 5 |
| `already_present` | 348 |
| `relabelled_as_original` | 0 |
| `removed_bad_file` | 0 |
| `sha1_mismatch` | 0 |
| `missing_on_commons` | 0 |
| Tổng ảnh trong `mmkg/images/` | 403 / 403 |
| Tổng dung lượng theo manifest (`bytes`) | 121.230.948 byte (khoảng 116 MB theo `du -sh`) |

Kết quả cuối theo cột `why` của `mmkg/images_manifest.csv` (403 dòng): `thumbnail` 350, `original_not_wider` 44, `original_already_present` 9.

| Link cloud | (điền sau khi upload) |
|---|---|

Sau khi duyệt và dựng lại, chạy lại lệnh trên. Script bỏ qua file đã có và đúng `sha1`, nên chỉ tải ảnh mới thay cho ảnh bị `drop`. Ảnh bị `drop` vẫn nằm trên đĩa và bị báo là file lạ: chạy với `--delete-stray` để xoá trước khi đưa lên cloud.

## 9. Split

**Chưa tạo.** Split sẽ nằm ở `splits/mmkg_vi/` theo cùng quy tắc ADR-0009 (ADR-0015 quy tắc 5), chốt ở một ADR sau, cùng với cách dựng cạnh `contains` và cách sinh truy vấn. Luật lọc hiện tại không đọc split hay nhãn nào (chỉ dùng số ảnh hợp lệ và triple Wikidata).

Những điểm ADR split phải chốt (từ lần kiểm rò rỉ ngày 2026-11-16, checklist PDF §6 trong `CLAUDE.md`):

| Điểm | Vấn đề | Việc phải làm |
|---|---|---|
| Đơn vị chia | Chưa chọn chia theo **ảnh** (cùng di tích có ảnh ở cả train và test) hay theo **item** (di tích test không có ảnh nào ở train). Hai cách đo hai năng lực khác nhau | Chọn và ghi lý do trong ADR split |
| `concept` và triple đếm trên mọi item | Luật `min_items_per_concept` và cạnh giữa các thực thể đang đếm trên **cả 84 item**, gồm cả item sẽ vào val/test | Nếu chia theo item: đếm lại trên item train, **hoặc** khai báo cách làm là transductive (đồ thị biết trước mọi thực thể) và chạy ablation bỏ thông tin đó |
| `entity_id` trong `images.csv` là đáp án | Cột này nói ảnh thuộc di tích nào | Không dùng nó để dựng cạnh `ảnh → contains → thực thể` cho ảnh val/test (mục 5, ADR-0051) |
| Đóng băng bộ dữ liệu | Dựng lại sau khi đã chia có thể đổi ảnh, thực thể, triple | Ghi `sha1` của các CSV chính thức (`entities.csv`, `triples.csv`, `images.csv`) vào file split; test kiểm khớp |
| Khoá ngưỡng dựng | Đổi ngưỡng sau khi có split (để kết quả đẹp hơn) là chọn siêu tham số trên test | Sau khi có split, ngưỡng trong `configs/mmkg_vi_build.yaml` khoá lại; muốn đổi thì ADR mới và chạy lại mọi mô hình |
| Ảnh gần trùng trong một item | Luật `sha1` chỉ bắt file giống hệt; ảnh chụp liên tiếp cùng góc vẫn còn. Chia theo ảnh thì một ảnh ở train và ảnh gần giống ở test làm điểm cao giả | Người duyệt `drop` với lý do `duplicate`; ADR split cân nhắc kiểm thêm bằng độ tương đồng đặc trưng |

## 10. Hạn chế, thiên lệch

- **Miền trộn hai loại:** công trình (chùa, đình, thành) và hiện vật (trống đồng, tượng) nằm chung, nên miền hơi rộng (ADR-0050).
- **Ảnh giống nhau:** nhiều chùa, đình có kiến trúc gần giống nhau; CLIP có thể khó phân biệt, kết quả truy vấn ảnh có thể thấp.
- **Thiên lệch độ phủ Wikidata:** chỉ di tích đã có trên Wikidata với `P1435` mới vào miền; di tích nổi tiếng, có trang Wikipedia, có nhiều thông tin hơn. Triple phản ánh những gì người đóng góp Wikidata đã nhập, không phải đầy đủ tri thức về di tích.
- **Nhiều di tích thiếu ảnh tự do:** trong 170 thực thể của miền, chỉ 121 có ít nhất một ảnh ứng viên (đếm từ `image_candidates.csv`) và 84 qua được luật 3 ảnh. Ví dụ trống đồng Ngọc Lũ (`Q3514747`) có 6 file nhưng chỉ 2 qua luật nên không thành `item`.
- **Thiếu nhãn tiếng Việt:** 7 thực thể không có `name_vi` (mục 6.3).
- **Khái niệm mang tính quản trị:** một số `concept` là danh hiệu hay trạng thái hồ sơ (ví dụ `Q29778318` "UNESCO World Heritage Site record modification", `Q106379705` "damaged") chứ không phải khái niệm nhìn thấy trên ảnh; người duyệt cân nhắc bỏ.
- **Dữ liệu theo bản dump:** Wikidata lấy qua QLever, dựng từ bản dump ghi trong `sparql_index` (mục 2); có thể lệch so với Wikidata trực tiếp cùng ngày.
- **Danh sách quan hệ chưa được nhóm duyệt** (mục 4).
- **Biên quy mô hẹp:** 331 triple, chỉ hơn mốc 300 là 31.

## 11. Người tạo và ngày

| Việc | Người | Ngày |
|---|---|---|
| Chọn miền (ADR-0050), duyệt thiết kế, chọn ngưỡng `min_items_per_concept: 3` | FluteInTheRain | 2026-11-16 |
| Viết code thu thập/dựng/tải ảnh, chạy thu thập và dựng bản đầu, data card | Claude Code (Opus 5.5), FluteInTheRain chịu trách nhiệm (`docs/ai-usage-log.md`) | 2026-11-16 |
| Đề xuất danh sách quan hệ (chưa duyệt) | Claude Code | 2026-11-16 |
| Sửa luật tác giả, file duyệt và trình tải ảnh; dựng lại (mục 3, 8) | Claude Code (Opus 5.5), FluteInTheRain chịu trách nhiệm | 2026-11-17 |
| Duyệt ảnh, triple, thực thể | (chưa phân công) | |
