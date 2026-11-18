# Data card: MMKG tiếng Việt – di tích và di sản Việt Nam

**Là gì?** Một đồ thị tri thức đa phương thức (MMKG) nhỏ do nhóm tự xây cho cấp độ 3 (PDF §5). Đỉnh là **thực thể** (di tích, hiện vật, khái niệm như "tượng", "đồng thanh", "Bảo tàng Lịch sử Việt Nam"), cạnh là **triple** (đầu, quan hệ, đuôi), và mỗi di tích có 3–5 **ảnh**. Định dạng theo ADR-0015; miền theo ADR-0050; cách thu thập, lọc và duyệt theo ADR-0051.

**Trạng thái (2026-11-17):** đã thu thập, dựng, **duyệt sơ bộ bằng AI** (Claude Code, `reviewer = claude-ai`) rồi **duyệt bởi người** (qphulong: mọi dòng AI `drop`, 36 ghi chú cần người xem, nhãn dịch, danh sách quan hệ; mục 7). `scripts/build_mmkg.py --require-reviewed` đã chạy qua (không còn mục `pending`). Bộ chính thức hiện có **119 thực thể (80 `item` + 39 `concept`), 303 triple, 380 ảnh**. Chưa được dùng cho thí nghiệm nào.

Đường dẫn tính từ gốc repo. Số liệu của bộ chính thức hiện tại lấy từ `mmkg/build_report.json` (sau duyệt của người) và `mmkg/raw/20261116/meta.json`; số nào đếm thêm từ `mmkg/*.csv` thì ghi rõ. Có ba mốc để so: **lần dựng đầu, trước duyệt** (128 thực thể, 331 triple, 403 ảnh), **sau duyệt AI** (119 thực thể, 302 triple, 382 ảnh) và **sau duyệt của người** (bộ hiện tại). Hai mốc đầu chỉ xuất hiện ở chỗ ghi rõ tên mốc và ở bảng trước/sau mục 7.

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

**Vấn đề tác giả ở lần dựng đầu (trước duyệt).** Commons trả ô `Artist` dưới dạng HTML. Với ảnh không rõ tác giả, Commons đặt chữ "Unknown author" trong hai thẻ liền nhau; bộ thu thập cũ xoá thẻ mà không chèn dấu cách nên được `Unknown authorUnknown author`. Một ảnh khác có câu mẫu giấy phép ("... the copyright holder of this work, hereby publishes it under the following license:") nằm trong ô tác giả. Lần dựng đầu có **11 ảnh** mang tác giả hỏng như vậy, **6 ảnh** trong đó là CC BY-SA 3.0, tức là không ghi công đúng giấy phép.

**Luật sửa** (`normalize_artist`, `name_before_licence_sentence`, `names_an_author`, `pick_author` trong `src/mmxai/data/mmkg_build.py`; ADR-0051 mục 2):

1. Làm sạch chữ `Artist`: bỏ ký tự rộng 0, gộp một cụm bị lặp liền nhau (`Unknown authorUnknown author.` → `Unknown author`), bỏ dấu câu ở cuối (`Chainwit.` → `Chainwit`).
2. Câu mẫu giấy phép có tên người ở đầu (`<tên>, the copyright holder of this work, hereby publishes it ...`) thì giữ phần `<tên>` làm tác giả: đó đúng là tên mà giấy phép trên Commons yêu cầu ghi công (thêm ngày 2026-11-17, sau duyệt của người; trước đó cả câu bị coi là không nêu ai).
3. Sau bước 2, chữ (không phân biệt hoa thường) là `unknown`, `unknown author`, `author unknown`, `anonymous`, hoặc vẫn chứa `hereby publish` (câu mẫu không có tên ở đầu), thì coi như không nêu ai và áp cột phải của bảng trên.
4. Bộ thu thập (`strip_html` trong `src/mmxai/data/mmkg_collect.py`) nay thay mỗi thẻ HTML bằng dấu cách, nên lần thu thập sau không còn dính chữ. Snapshot `20261116` không bị sửa; luật 1–3 xử lý nó ở bước dựng.

Trước/sau (đếm từ `mmkg/images.csv` trước và sau commit `c6d7a97`; đây là số của lần dựng đầu 403 ảnh, trước duyệt):

| Mục | Trước | Sau |
|---|---|---|
| Ảnh có tác giả hỏng | 11 | 0 |
| Ảnh ghi `uploader: ...` (đều CC BY-SA 3.0) | 1 | 7 |
| Ảnh ghi `unknown` (6 Public domain, 1 CC0) | 1 | 7 |
| Ảnh bị bỏ vì `no_author` | 0 | 0 |
| Tổng số ảnh | 403 | 403 (cùng `image_id`, chỉ cột `author` đổi ở 36 dòng, phần lớn là bỏ dấu chấm cuối) |

**Đã sửa – tên ghi công ảnh `Bia Vinh Lang.JPG`** (`Q10739843_3e40239b`, CC BY-SA 3.0): ô `Artist` là câu mẫu "Vietnames Wikipedia user Rungbachduong, the copyright holder of this work, hereby publishes it under the following license:". Luật cũ bỏ cả câu nên ảnh bị ghi `uploader: Vmenkov`, trong khi Vmenkov chỉ là người tải lên. Người duyệt (qphulong, 2026-11-17) đối chiếu trang Commons: mục Attribution yêu cầu ghi "Vietnames Wikipedia user Rungbachduong". Luật 2 ở trên được thêm cho trường hợp này; dựng lại chỉ đổi đúng cột `author` của dòng này thành `Vietnames Wikipedia user Rungbachduong`. Đây là dòng duy nhất trong snapshot có câu mẫu này (đếm từ `image_credits.csv`).

Phân bố giấy phép của 380 ảnh chính thức sau duyệt của người (đếm từ `mmkg/images.csv`):

| `license` | Số ảnh |
|---|---|
| `CC0` | 77 |
| `CC BY-SA 3.0` | 69 |
| `CC BY 2.0` | 65 |
| `Public domain` | 62 |
| `CC BY-SA 4.0` | 57 |
| `CC BY-SA 2.0` | 23 |
| `CC BY 4.0` | 21 |
| `CC BY 3.0` | 3 |
| `CC BY-SA 2.5` | 2 |
| `CC BY 2.5` | 1 |

Sau duyệt của người và sửa ghi công, ảnh ghi `uploader: ...` là 6 và ảnh ghi `unknown` là 8 (đếm từ `mmkg/images.csv`; sau duyệt AI: 7 và 7).

Điều kiện khi dùng hoặc đăng lại:

- Ảnh CC BY / CC BY-SA: phải ghi tác giả (`author`), giấy phép (`license`) và link trang file (`source_url`). Ảnh CC BY-SA sửa đổi phải giữ cùng giấy phép.
- Ảnh **không** nằm trong git (`mmkg/images/` đã gitignore, ADR-0008); chia sẻ qua link cloud của nhóm: https://drive.google.com/drive/folders/1KnsEIUrnbo0C58idfRK_sjhy8cRsahzG (mục 8). Người nhận kiểm từng file bằng `sha1` trong `mmkg/images_manifest.csv` (mục 8).
- ADR-0015 quy tắc 4: không dùng ảnh có mặt người nhận diện được. Việc này chỉ kiểm được khi duyệt bằng mắt (lý do `identifiable_person`, mục 7; 11 ảnh đã bị bỏ vì lý do này).

## 4. Cách tạo triple

Triple được tạo **bán tự động**: script lấy từ Wikidata, nhóm lọc bằng luật trong config, rồi thành viên duyệt từng triple.

1. **Thu thập** (`scripts/collect_mmkg.py`): mọi phát biểu trực tiếp (`wdt:P...`) của thực thể trong miền trỏ tới một thực thể Wikidata khác, cộng mọi triple giữa các thực thể đích đã có (câu `concept_triples`). Giữ nguyên mã `P`.
2. **Dựng** (`scripts/build_mmkg.py`, logic `merge_triples`, `find_concepts`, `keep_triples` trong `src/mmxai/data/mmkg_build.py`): chỉ giữ 18 mã `P` trong `configs/mmkg_vi_build.yaml`, đổi sang 17 tên snake_case, gộp `P276` và `P195` thành `kept_at`, chỉ giữ triple có cả hai đầu là thực thể được giữ.
3. **Duyệt** (mục 7): từng triple được đánh dấu `ok`/`drop` trong `mmkg/review.csv`. AI duyệt sơ bộ bỏ 16 khoá triple; người duyệt (qphulong) xem lại cả 16, giữ lại 1 (`Q43113623|P361|Q9259`), còn 15 khoá bị `drop`.

Cột `source` = URL Wikidata của thực thể đầu, nên mọi triple kiểm lại được. Cột thêm `wikidata_property` ghi mã `P` gốc (triple gộp ghi `P195|P276`).

**Bảng quan hệ** (danh sách do Claude Code đề xuất; qphulong kiểm và giữ nguyên ngày 2026-11-17, không có nhận xét; sửa được trong config):

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

Bộ chính thức dưới đây là bộ **sau duyệt của người** (mục 7). Hai cột ở giữa là các mốc trước đó, để so.

| Mục | Snapshot gốc | Lần dựng đầu (trước duyệt) | Sau duyệt AI | Bộ chính thức (sau duyệt của người) | Mục tiêu (ADR-0050) |
|---|---|---|---|---|---|
| Thực thể | 848 (`raw_entities`; gồm thực thể của miền và thực thể đích) | 128 = 84 `item` + 44 `concept` | 119 = 80 + 39 | 119 = 80 `item` + 39 `concept` (`entities_item`, `entities_concept`) | 50–150 |
| Triple | 3.581 (`raw_triples`) | 331 | 302 | 303 (`triples`) | 300–1.000 |
| Ảnh | 5.093 cặp (thực thể, file) ứng viên (`raw_image_candidates`) | 403 | 382 | 380 (`images`) | 200–500 |

Snapshot có 170 thực thể của miền (`in_domain = 1` trong `entities_raw.csv`), 4.979 file ảnh khác nhau (`image_credits` trong `meta.json`). Cả 39 `concept` đều là thực thể đích, không phải thực thể của miền (đếm từ `mmkg/entities.csv` và `entities_raw.csv`).

**Lưu ý biên quy mô:** 303 triple chỉ hơn mốc tối thiểu 300 là 3. Bỏ thêm triple (hoặc bỏ một `item`, vì triple của nó mất theo) có thể xuống dưới mốc; khi đó phải đổi ngưỡng trong `configs/mmkg_vi_build.yaml` (ví dụ `min_items_per_concept`) và ghi lý do (mục 7, ADR-0051).

Số ảnh mỗi `item` (đếm từ `mmkg/images.csv`): 65 di tích có 5 ảnh, 10 có 4 ảnh, 5 có 3 ảnh.

Số triple theo quan hệ (đếm từ `mmkg/triples.csv`): `heritage_designation` 86, `instance_of` 65, `kept_at` 45, `made_from` 30, `located_in` 26, `influenced_by` 11, `part_of` 11, `culture` 10, `time_period` 7, `depicts` 7, `has_part` 2, `discovered_at` 2, `named_after` 1. Bốn quan hệ trong danh sách không còn triple nào: `creator`, `commissioned_by`, `has_pattern` (không có ngay từ lần dựng đầu) và `significant_event` (lần dựng đầu có 9; AI duyệt `drop` cả 9 và người duyệt đồng ý, mục 7). 27 triple `kept_at` gộp từ cả `P195` lẫn `P276`.

### 6.2. Số dòng bị bỏ theo từng luật

Mọi luật là **lựa chọn của nhóm, không phải yêu cầu của đề**. Số lấy từ `mmkg/build_report.json` (bộ sau duyệt của người), theo đúng thứ tự áp luật. Các khoá `*_review_drop` và `*_dropped_review` là kết quả duyệt (AI duyệt sơ bộ, người duyệt lại; mục 7).

**Ảnh** (5.093 cặp ứng viên → 380):

| Khoá trong report | Số bị bỏ | Luật | Lý do |
|---|---|---|---|
| `images_dropped_entity_review_drop` | 0 | Thực thể bị duyệt `drop` thì bỏ mọi ảnh của nó | Kết quả duyệt |
| `images_dropped_shared_between_entities` | 228 | File thuộc hơn một thực thể | Không biết ảnh cho thấy thực thể nào |
| `images_dropped_mime` | 41 | Chỉ `image/jpeg`, `image/png`, `image/webp` | CLIP chỉ đọc ảnh raster |
| `images_dropped_license` | 15 | Ngoài `Public domain`, `CC0`, `CC BY *`, `CC BY-SA *` | PDF §9, ADR-0015 quy tắc 3 |
| `images_dropped_too_small` | 6 | Cạnh ngắn < 224 px | Kích thước đầu vào của CLIP ViT-B-32 |
| `images_dropped_no_author` | 0 | CC BY/CC BY-SA mà không có tác giả lẫn người tải lên | Không ghi công được |
| `images_dropped_review_drop` | 88 | Ảnh bị duyệt `drop` | Kết quả duyệt (mục 7): 68 do AI đề xuất và người đồng ý, 20 do người bỏ thêm |
| `images_dropped_duplicate_sha1` | 0 | Trùng `sha1`, giữ tên nhỏ nhất | Hai file giống hệt từng byte |
| `images_dropped_below_min_per_item` | 51 | Thực thể còn dưới 3 ảnh hợp lệ thì không thành `item` | Cần đủ ảnh để chia train/val/test sau này |
| `images_dropped_over_max_per_item` | 4.284 | Giữ tối đa 5 ảnh mỗi `item` (`P18` trước, rồi theo tên file) | Giữ quy mô gợi ý; không để di tích nhiều ảnh lấn át |

Kiểm lại: 5.093 − 228 − 41 − 15 − 6 − 88 − 51 − 4.284 = 380. (Sau duyệt AI: `review_drop` 68, `below_min_per_item` 53, `over_max_per_item` 4.300, tổng 382. Lần dựng đầu trước duyệt: 0, 45, 4.355, tổng 403.)

**Triple** (3.581 → 303):

| Khoá trong report | Số bị bỏ | Luật | Lý do |
|---|---|---|---|
| `triples_dropped_unknown_relation` | 1.835 | Mã `P` không có trong danh sách 18 mã `P` (17 tên quan hệ) | Bỏ quan hệ quản trị (quốc gia, bản quyền, danh mục...) |
| `triples_dropped_review` | 15 | Triple bị duyệt `drop` | Kết quả duyệt (mục 7) |
| `triples_dropped_end_not_kept` | 1.333 | Một đầu không phải thực thể được giữ, hoặc tự trỏ về chính nó | Đồ thị không có đầu mút treo |

Ba số trên không cộng thẳng ra 303, vì bước gộp `P276`/`P195` nằm giữa: 3.581 − 1.835 = 1.746 dòng; gộp các dòng cùng (đầu, `kept_at`, đuôi) còn 303 + 1.333 + 15 = 1.651 triple; bỏ 1.333 và 15 còn 303 (`merge_triples`, `keep_triples`). (Sau duyệt AI: 1.333 và 16, còn 302. Lần dựng đầu: 1.320 và 0, còn 331.)

**Thực thể:** `entities_dropped_review` = 4 (mục 7). Từ 170 thực thể của miền, 80 có đủ 3 ảnh hợp lệ thành `item` (lần dựng đầu: 84; bốn di tích ít hơn do ảnh bị `drop` hoặc thay đổi theo luật; `images_dropped_below_min_per_item` là 45 ở lần dựng đầu, 53 sau duyệt AI, 51 sau duyệt của người); 39 thực thể đích được ≥ 3 `item` trỏ tới thành `concept` (`min_items_per_concept: 3`, FluteInTheRain chọn 3 thay vì 2 để tổng số thực thể nằm trong 50–150, ADR-0050; lần dựng đầu: 44).

### 6.3. Thiếu nhãn tiếng Việt

Lần dựng đầu có 7 thực thể không có nhãn `vi` trên Wikidata (cột `name_vi` rỗng): `Q120835889` (`item`, "The set of steps of Kinh Thien Palace") và 6 `concept`: `Q106379705`, `Q1459900`, `Q19683138`, `Q29778318`, `Q43113623`, `Q830393`. Hiện `entities_missing_name_vi` = 0 (`mmkg/build_report.json`): 3 thực thể (`Q106379705`, `Q830393`, `Q29778318`) bị `drop` nên không còn trong bộ, và 4 thực thể còn lại (`Q120835889`, `Q1459900`, `Q19683138`, `Q43113623`) được điền nhãn bằng `fix_name_vi` do AI dịch (mục 7). Người duyệt (qphulong) đã kiểm và giữ nguyên cả 5 nhãn dịch (gồm `Q34095` "đồng thanh").

## 7. Kết quả duyệt

**Trạng thái: đã duyệt hai lượt.** Lượt 1 là **AI duyệt sơ bộ** mọi dòng (`reviewer = claude-ai`). Lượt 2 là **người duyệt** (qphulong, 2026-11-17): xem lại mọi dòng AI `drop`, mọi ca AI không chắc, nhãn dịch và danh sách quan hệ, rồi ký tên vào dòng mình quyết định. `mmkg/review.csv` hiện có 962 dòng, đều là `ok` hoặc `drop` (không còn `pending`). Số liệu dưới đây đếm từ `mmkg/review.csv`, riêng ảnh bộ chính thức lấy từ `mmkg/build_report.json`.

Cột `reviewer` cho biết ai chịu trách nhiệm từng dòng:

| `reviewer` | Số dòng | Nghĩa |
|---|---|---|
| `qphulong` | 158 | Người đã xem và quyết định dòng này (đồng ý hoặc sửa quyết định của AI) |
| `claude-ai` | 804 | Quyết định `ok` của AI mà người không xem riêng từng dòng (xem "Phạm vi" ở mục 7.2) |

### 7.1. Lượt 1: AI duyệt sơ bộ

- **Người duyệt:** Claude Code (Opus 5.5) làm bộ điều phối, dùng các subagent có khả năng xem ảnh để xem từng ảnh.
- **Sáu vòng:** mỗi lần dựng lại (`scripts/build_mmkg.py`) thay ảnh bị `drop` bằng ảnh kế tiếp của di tích đó; ảnh mới lại là dòng `pending` và được duyệt ở vòng sau. Lặp tới khi `scripts/build_mmkg.py --require-reviewed` thoát mã 0 (không còn `pending`, không có khoá duyệt lệch).
- **Tiêu chí:** `drop` ảnh với lý do `identifiable_person`, `not_the_object`, `duplicate`, `other`; **thận trọng**: ca không chắc thì giữ `ok` và ghi chú để người xem lại.
- **Kết quả:** 946 dòng; `drop` 68 ảnh (34 `not_the_object`, 21 `duplicate`, 11 `identifiable_person`, 2 `other`), 16 khoá triple, 4 thực thể; 5 nhãn `fix_name_vi` do AI dịch; 36 ghi chú `unsure:`/`check:` trong `mmkg/review_ai_notes.csv` (130 dòng; cột `kind, key, review_status, note`). Bộ sau lượt này: 119 thực thể, 302 triple, 382 ảnh.

**Bộ đối chiếu ảnh bị loại ở lượt 1:** 68 ảnh bị `drop` (bản rộng 480 px tải lại từ Commons ngày 2026-11-17) ghép thành 6 tấm lưới có lý do, tên hiện vật, `image_id`, ghi chú, kèm `dropped_images.csv` (link Commons từng file), ở thư mục con `review_drops/` trong thư mục Drive của MMKG (mục 8). Thư mục này chỉ dành cho nhóm (có ảnh người), không đăng lại công khai.

### 7.2. Lượt 2: người duyệt

**Cách làm.** Claude Code sinh một trang HTML chạy trên máy (`data/mmkg_review/review.html`, gitignored) từ các file trong `mmkg/`. Trang hiện từng mục cần người quyết: ảnh (tải thẳng từ Commons), tên di tích, quyết định và ghi chú của AI, link Wikidata/Commons. Người duyệt bấm giữ (`ok`) hoặc bỏ (`drop` kèm lý do), rồi xuất `review_changes.csv` theo đúng cột của `review.csv`. Claude Code ghép file này vào `mmkg/review.csv` theo khoá (`kind`, `key`) và dựng lại. Trước khi ghép, Claude Code chạy thử bản dựng trong thư mục tạm và báo cho người duyệt hai chỗ không nhất quán (9 ảnh của "Column engraved with dragons from Dam Pagoda" và "Da Lat Railway Station" bị đánh `not_the_object` nhầm; 7 triple `significant_event` được giữ trong khi thực thể đuôi vẫn bị bỏ); người duyệt chọn giữ 9 ảnh và bỏ 7 triple.

**Hai vòng.** Vòng 1 gồm 142 dòng. Ảnh bị bỏ thêm làm 16 ảnh thay thế vào bộ dưới dạng `pending`; vòng 2 người duyệt xem 16 ảnh này (trang `data/mmkg_review/review_round2.html`, mỗi ảnh kèm các ảnh khác của cùng di tích để dò trùng) và giữ cả 16. Sau vòng 2, `--require-reviewed` thoát mã 0.

| Nhóm mục | Số mục | Người đồng ý với AI | Người sửa |
|---|---|---|---|
| Ảnh AI `drop` | 68 | 68 (giữ nguyên lý do) | 0 |
| Ảnh AI ghi chú `unsure:`/`check:` (đều là AI `ok`) | 36 | 20 giữ `ok` | 16 đổi sang `drop` (`not_the_object`) |
| Ảnh AI `ok` khác, người đánh dấu khi lướt toàn bộ ảnh | 13 | 9 giữ `ok` | 4 đổi sang `drop` (`not_the_object`) |
| Ảnh thay thế mới (vòng 2) | 16 | – (giữ cả 16, `ok`) | – |
| Khoá triple AI `drop` | 16 | 15 | 1 giữ lại: `Q43113623\|P361\|Q9259` ("một phần của Di sản thế giới UNESCO" là một phần của "Di sản thế giới") |
| Thực thể AI `drop` | 4 | 4 | 0 |
| Nhãn tiếng Việt AI dịch | 5 | 5 (giữ nguyên) | 0 |
| Danh sách 17 quan hệ (mục 4) | 1 | Giữ nguyên, không nhận xét | 0 |

20 ảnh người bỏ thêm thuộc: "Ba Bể Lake" (5), "The Amitabha Triad statues of Thầy Temple" (5), "The Weaver Girl and the Cowherd" (2, tượng ở Nhật), "The Revolutionary Road" (2, hình vẽ trong sách), và mỗi di tích một ảnh: "Hoi An Ancient Town", "Hội An", "Tràng An Landscape Complex", "Noi temple", "Cam Giang I bronze drum", "Sảnh Môn Hạ Seal". Lý do chung: ảnh chụp cảnh xung quanh, không cho thấy chính di tích/hiện vật.

**Phạm vi.** Trang duyệt có một tab lưới toàn bộ ảnh đang `ok` để lướt tìm mặt người và ảnh sai đối tượng; trang chỉ xuất dòng người đã bấm, nên 804 dòng còn ký `claude-ai` là quyết định `ok` của AI mà người không đánh dấu gì. Trang không ghi lại người đã lướt hết lưới hay chưa.

### 7.3. Số mục `drop` và `ok` sau hai lượt

| Loại | Lý do `drop` | Số mục `drop` | Số mục `ok` |
|---|---|---|---|
| Ảnh | `not_the_object` | 54 (AI 34 + người 20) | 386 (tổng mọi lý do) |
| Ảnh | `duplicate` | 21 | |
| Ảnh | `identifiable_person` | 11 | |
| Ảnh | `other` | 2 | |
| Ảnh | tổng | 88 | 386 |
| Triple (khoá `head\|P\|tail`) | (không ghi `reason`; lý do của AI nằm trong `review_ai_notes.csv`) | 15 | 345 |
| Thực thể | (không ghi `reason`) | 4 | 124 |

Tổng 962 dòng = 474 ảnh + 360 khoá triple + 128 thực thể. Ảnh có 474 dòng vì mỗi vòng thêm ảnh thay thế; chỉ 380 ảnh `ok` nằm trong bộ chính thức (386 `ok` lớn hơn 380: một số ảnh `ok` không còn trong bộ vì luật khác áp sau, ví dụ tối đa 5 ảnh mỗi `item`).

**4 thực thể bị `drop`:** `Q106379705` ("damaged"), `Q830393` ("preservation"), `Q29778318` ("UNESCO World Heritage Site record modification"), `Q532` ("làng"/Village). Đều là khái niệm quản trị hoặc quá chung, không nhìn thấy trên ảnh (mục 10).

**15 khoá triple bị `drop`:** 9 triple `significant_event` (`P793`) tới khái niệm meta như "damaged", "preservation", "UNESCO record modification"; 4 triple `named_after` (`P138`) tới "làng" (`Q532`) chung chung; 2 triple `part_of` (`P361`): `Q1459900` tới `Q9259` và `Q386724` tới `Q2668072`. Danh sách đầy đủ: lọc `kind = triple`, `review_status = drop` trong `mmkg/review.csv`.

**5 nhãn tiếng Việt do AI dịch** (cột `fix_name_vi`, `review_status = ok`, người duyệt giữ nguyên): `Q120835889` "Bậc thềm điện Kính Thiên", `Q1459900` "Di sản thế giới dự kiến", `Q19683138` "Khu Ramsar", `Q43113623` "một phần của Di sản thế giới UNESCO", `Q34095` "đồng thanh" (sửa nhãn Wikidata "đồng điếu", vì "bronze" là đồng thanh).

### 7.4. Ảnh hưởng lên bộ chính thức (trước → sau)

| Mục | Lần dựng đầu (trước duyệt) | Sau duyệt AI | Sau duyệt của người (`mmkg/build_report.json`) |
|---|---|---|---|
| Thực thể | 128 (84 `item` + 44 `concept`) | 119 (80 + 39) | 119 (80 `item` + 39 `concept`) |
| Triple | 331 | 302 | 303 |
| Ảnh | 403 | 382 | 380 |
| Thực thể thiếu `name_vi` | 7 | 0 | 0 |
| Dòng còn `pending` (thực thể / ảnh / triple) | 128 / 403 / 360 | 0 / 0 / 0 | 0 / 0 / 0 |
| `entities_dropped_review` / `images_dropped_review_drop` / `triples_dropped_review` | 0 / 0 / 0 | 4 / 68 / 16 | 4 / 88 / 15 |

So với bộ sau duyệt AI: 18 ảnh rời bộ và 16 ảnh thay thế vào (đếm theo `image_id` của `mmkg/images.csv`), và một triple `part_of` trở lại. Số triple 303 chỉ hơn mốc 300 là 3. Số khoá duyệt triple là 360 (345 `ok` + 15 `drop`, không phải 303) vì file duyệt khoá theo mã `P` gốc (triple `kept_at` gộp có hai dòng duyệt) và vì có khoá thuộc triple đã rơi khỏi bộ do luật khác (thực thể bị bỏ, `item` rớt dưới 3 ảnh).

Nếu nhóm duyệt bỏ thêm tới mức triple dưới 300 hoặc thực thể ra ngoài 50–150, đổi ngưỡng trong `configs/mmkg_vi_build.yaml`, dựng lại và ghi ngưỡng mới, lý do và số mới vào mục này (ADR-0051). Duyệt lại thì sửa dòng trong `mmkg/review.csv`, ghi tên mình vào `reviewer`, rồi dựng lại (`mmkg/README.md` mục 3).

## 8. Tải ảnh

Ảnh được tải dưới dạng **thumbnail rộng 960 px** từ các kích thước Wikimedia Commons đã tạo sẵn (`download_width: 960` trong `configs/mmkg_vi_build.yaml`). Lý do: lần tải bản gốc ngày 2026-11-16 bị Wikimedia chặn tốc độ (HTTP 429, rồi phải chờ 600 s sau 11/403 file); chính sách robot của Wikimedia cho phép tải "bản gốc hoặc một trong các kích thước thumbnail tạo sẵn"; CLIP chỉ cần 224 px nên 960 px không mất gì, và dung lượng giảm từ khoảng 1,1 GB xuống vài chục MB. File lưu ở `mmkg/images/<image_id><đuôi file>` (logic `download_images` trong `src/mmxai/data/mmkg_images.py`).

**Khi nào tải bản gốc thay cho thumbnail.** Script tải **bản gốc** và so `sha1` với cột `sha1` của `mmkg/images.csv` (là `sha1` Commons của bản gốc) trong ba trường hợp. Bản gốc sai `sha1` không được lưu.

| `why` trong manifest | `kind` | Khi nào |
|---|---|---|
| `thumbnail` | `thumbnail` | Bản gốc rộng hơn 960 px và thumbnail 960 px có cạnh ngắn ≥ 224 px (trường hợp thường gặp) |
| `original_not_wider` | `original` | Bản gốc rộng ≤ 960 px. Commons không phóng to ảnh, nên "thumbnail" nó trả về chính là file gốc (URL có `utm_content=thumbnail_unscaled`); tải như bản gốc để kiểm được `sha1`. Theo cột `width` của `images.csv` hiện có 37 ảnh như vậy (sau duyệt AI: 36; lần dựng đầu: 44) |
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

Lần tải đầu bắt đầu ngày 2026-11-16, trước khi duyệt, và xong ngày 2026-11-17 sau nhiều lần chạy lại (bị `upload.wikimedia.org` giới hạn bằng HTTP 429; hết bị chặn khi User-Agent có email liên hệ, xem `configs/mmkg_vi_collect.yaml`). Lần chạy xong đó tải đủ 403/403 ảnh của bộ trước duyệt (350 thumbnail, 44 bản gốc không rộng hơn 960 px, 9 `original_already_present`).

**Sau duyệt của người** (2026-11-17): ảnh được tải lại từ đầu trên một máy khác (máy của qphulong, `mmkg/images/` trống), nên không có file cũ hay file lạ nào. Số liệu lần chạy này (`mmkg/download_report.json`):

| Mục | Giá trị |
|---|---|
| `already_present` | 0 |
| `downloaded_thumbnail` | 343 |
| `downloaded_original` | 37 |
| `relabelled_as_original` | 0 |
| `removed_bad_file` | 0 |
| `sha1_mismatch` | 0 |
| `missing_on_commons` | 0 |
| Tổng ảnh trong `mmkg/images/` | 380 / 380 |
| Tổng dung lượng theo manifest (`bytes`) | 94.314.505 byte (đếm từ `mmkg/images_manifest.csv`) |

Kết quả theo cột `why` của `mmkg/images_manifest.csv` (380 dòng): `thumbnail` 343, `original_not_wider` 37. Không còn dòng `original_already_present`: 9 ảnh trước đây là bản gốc của lần tải đầu, nay là thumbnail 960 px, nên `sha1` trong manifest của 9 ảnh đó (và dung lượng tổng) khác bản sau duyệt AI (382 ảnh, 115.741.097 byte, `thumbnail` 337, `original_not_wider` 36, `original_already_present` 9).

| Link cloud | https://drive.google.com/drive/folders/1KnsEIUrnbo0C58idfRK_sjhy8cRsahzG |
|---|---|

Thư mục Google Drive `MMKG-di-tich-viet-nam` (ai có link đều xem được, chủ sở hữu: Khang) chứa bản sao của `mmkg/` (ảnh, CSV chính thức, snapshot gốc, manifest, báo cáo tải) và data card này. Lịch sử đồng bộ:

| Ngày | Ai chạy | Bộ trên Drive | `rclone check` |
|---|---|---|---|
| 2026-11-17 | Khang | sau duyệt AI: 382 ảnh (đã gỡ ảnh `drop`, kể cả 11 ảnh có mặt người) | 396 khớp, 0 khác biệt |
| 2026-11-18 | qphulong (quyền Editor) | sau duyệt của người: thêm 16 ảnh mới, thay các file có `sha1` đổi (9 ảnh, CSV, manifest, báo cáo) | 394 khớp, 0 sai nội dung, **18 file thừa trên Drive** |

**Còn 18 ảnh thừa trên Drive** (cả 18 là ảnh qphulong đánh `drop` / `not_the_object`, không có ảnh mặt người). Editor ghi đè và thêm file được nhưng không xoá được file do người khác sở hữu trong thư mục My Drive (lỗi `403 insufficientFilePermissions`), nên chỉ chủ sở hữu xoá được. Khang chạy lại đúng lệnh `rclone sync` bên dưới từ một bản checkout có `mmkg/images/` mới (hoặc xoá tay 18 file mà `rclone check` báo "file not in Local file system"); sau đó `rclone check` phải báo 0 khác biệt. Cho tới khi đó, ai tải cả thư mục về sẽ có 398 ảnh; chỉ 380 ảnh có dòng trong `images.csv` / `images_manifest.csv`, 18 ảnh còn lại phải bỏ.

**Đồng bộ từ máy khác** (sau mỗi lần duyệt lại và dựng lại):

1. Xin chủ sở hữu quyền **Editor** cho đúng tài khoản Google sẽ đăng nhập rclone (Viewer/Commenter chỉ đọc được).
2. `rclone config create gdrive drive scope=drive` (đăng nhập bằng tài khoản ở bước 1; sai tài khoản thì `rclone config reconnect gdrive:`).
3. Thư mục được chia sẻ không nằm trong My Drive của mình, nên mọi lệnh thêm `--drive-shared-with-me` (hoặc tạo shortcut vào My Drive).
4. Chạy thử: `rclone --drive-shared-with-me sync mmkg gdrive:MMKG-di-tich-viet-nam/mmkg --exclude ".DS_Store" --dry-run`, rồi chạy thật (bỏ `--dry-run`), rồi `rclone --drive-shared-with-me check mmkg gdrive:MMKG-di-tich-viet-nam/mmkg --exclude ".DS_Store"`. Người không phải chủ sở hữu sẽ gặp đúng giới hạn xoá nói trên.

Kiểm một file tải về bằng `shasum -a 1 <file>` so với cột `sha1` trong `images_manifest.csv`.

Mỗi lần nhóm sửa duyệt rồi dựng lại, chạy lại lệnh trên. Script bỏ qua file đã có và đúng `sha1`, nên chỉ tải ảnh mới thay cho ảnh bị `drop`. Ảnh bị `drop` vẫn nằm trên đĩa và bị báo là file lạ: chạy với `--delete-stray` để xoá trước khi đưa lên cloud.

## 9. Split

**Chưa tạo.** Split sẽ nằm ở `splits/mmkg_vi/` theo cùng quy tắc ADR-0009 (ADR-0015 quy tắc 5), chốt ở một ADR sau, cùng với cách dựng cạnh `contains` và cách sinh truy vấn. Luật lọc hiện tại không đọc split hay nhãn nào (chỉ dùng số ảnh hợp lệ và triple Wikidata).

Những điểm ADR split phải chốt (từ lần kiểm rò rỉ ngày 2026-11-16, checklist PDF §6 trong `CLAUDE.md`):

| Điểm | Vấn đề | Việc phải làm |
|---|---|---|
| Đơn vị chia | Chưa chọn chia theo **ảnh** (cùng di tích có ảnh ở cả train và test) hay theo **item** (di tích test không có ảnh nào ở train). Hai cách đo hai năng lực khác nhau | Chọn và ghi lý do trong ADR split |
| `concept` và triple đếm trên mọi item | Luật `min_items_per_concept` và cạnh giữa các thực thể đang đếm trên **cả 80 item**, gồm cả item sẽ vào val/test | Nếu chia theo item: đếm lại trên item train, **hoặc** khai báo cách làm là transductive (đồ thị biết trước mọi thực thể) và chạy ablation bỏ thông tin đó |
| `entity_id` trong `images.csv` là đáp án | Cột này nói ảnh thuộc di tích nào | Không dùng nó để dựng cạnh `ảnh → contains → thực thể` cho ảnh val/test (mục 5, ADR-0051) |
| Đóng băng bộ dữ liệu | Dựng lại sau khi đã chia có thể đổi ảnh, thực thể, triple | Ghi `sha1` của các CSV chính thức (`entities.csv`, `triples.csv`, `images.csv`) vào file split; test kiểm khớp |
| Khoá ngưỡng dựng | Đổi ngưỡng sau khi có split (để kết quả đẹp hơn) là chọn siêu tham số trên test | Sau khi có split, ngưỡng trong `configs/mmkg_vi_build.yaml` khoá lại; muốn đổi thì ADR mới và chạy lại mọi mô hình |
| Ảnh gần trùng trong một item | Luật `sha1` chỉ bắt file giống hệt; ảnh chụp liên tiếp cùng góc vẫn còn. Chia theo ảnh thì một ảnh ở train và ảnh gần giống ở test làm điểm cao giả | Người duyệt `drop` với lý do `duplicate`; ADR split cân nhắc kiểm thêm bằng độ tương đồng đặc trưng |

## 10. Hạn chế, thiên lệch

- **Miền trộn hai loại:** công trình (chùa, đình, thành) và hiện vật (trống đồng, tượng) nằm chung, nên miền hơi rộng (ADR-0050).
- **Ảnh giống nhau:** nhiều chùa, đình có kiến trúc gần giống nhau; CLIP có thể khó phân biệt, kết quả truy vấn ảnh có thể thấp.
- **Thiên lệch độ phủ Wikidata:** chỉ di tích đã có trên Wikidata với `P1435` mới vào miền; di tích nổi tiếng, có trang Wikipedia, có nhiều thông tin hơn. Triple phản ánh những gì người đóng góp Wikidata đã nhập, không phải đầy đủ tri thức về di tích.
- **Nhiều di tích thiếu ảnh tự do:** trong 170 thực thể của miền, chỉ 121 có ít nhất một ảnh ứng viên (đếm từ `image_candidates.csv`) và 80 qua được luật 3 ảnh và duyệt (84 trước duyệt). Ví dụ trống đồng Ngọc Lũ (`Q3514747`) có 6 file nhưng chỉ 2 qua luật nên không thành `item`.
- **Thiếu nhãn tiếng Việt:** lần dựng đầu có 7 thực thể không có `name_vi`; nay đã điền, trong đó 5 nhãn do AI dịch và được một người duyệt giữ nguyên (mục 6.3, 7).
- **Khái niệm mang tính quản trị:** lần dựng đầu có một số `concept` là danh hiệu hay trạng thái hồ sơ (ví dụ `Q29778318` "UNESCO World Heritage Site record modification", `Q106379705` "damaged"); 3 khái niệm như vậy đã bị bỏ (AI đề xuất, người đồng ý). Các khái niệm danh hiệu còn lại (ví dụ "Di sản thế giới", "Khu Ramsar") được giữ có chủ ý.
- **Dữ liệu theo bản dump:** Wikidata lấy qua QLever, dựng từ bản dump ghi trong `sparql_index` (mục 2); có thể lệch so với Wikidata trực tiếp cùng ngày.
- **Biên quy mô hẹp:** 303 triple, chỉ hơn mốc 300 là 3 (lần dựng đầu: 331).
- **Duyệt của người chưa phủ hết từng dòng:** một người (qphulong) đã xem mọi dòng AI `drop`, mọi ca AI không chắc, nhãn dịch và danh sách quan hệ, và có tab lướt toàn bộ ảnh `ok`; nhưng 804 dòng `ok` vẫn chỉ mang chữ ký `claude-ai` (mục 7.2). Mỗi dòng chỉ có một người duyệt, chưa có người thứ hai kiểm chéo. Vẫn có thể sót ảnh có mặt người nhận diện được.

## 11. Người tạo và ngày

| Việc | Người | Ngày |
|---|---|---|
| Chọn miền (ADR-0050), duyệt thiết kế, chọn ngưỡng `min_items_per_concept: 3` | FluteInTheRain | 2026-11-16 |
| Viết code thu thập/dựng/tải ảnh, chạy thu thập và dựng bản đầu, data card | Claude Code (Opus 5.5), FluteInTheRain chịu trách nhiệm (`docs/ai-usage-log.md`) | 2026-11-16 |
| Đề xuất danh sách quan hệ (chưa duyệt) | Claude Code | 2026-11-16 |
| Sửa luật tác giả, file duyệt và trình tải ảnh; dựng lại (mục 3, 8) | Claude Code (Opus 5.5), FluteInTheRain chịu trách nhiệm | 2026-11-17 |
| Duyệt sơ bộ bằng AI toàn bộ ảnh, triple, thực thể (946 dòng, 6 vòng, `reviewer = claude-ai`) | Claude Code (Opus 5.5) cùng subagent xem ảnh, FluteInTheRain chịu trách nhiệm | 2026-11-17 |
| Duyệt của người: mọi dòng AI `drop`, 36 ghi chú cần kiểm, lướt ảnh `ok`, nhãn dịch, danh sách quan hệ; 2 vòng, 158 dòng ký tên (mục 7.2) | qphulong | 2026-11-17 |
| Trang duyệt HTML, ghép quyết định vào `mmkg/review.csv`, dựng lại, tải ảnh mới, cập nhật data card | Claude Code (Opus 5.5), qphulong chịu trách nhiệm | 2026-11-17 |
