# Thiết kế: thu thập và dựng MMKG tiếng Việt (di tích và di sản Việt Nam)

Ngày: 2026-11-16. Người yêu cầu và quyết định: FluteInTheRain. Trạng thái: đã duyệt (FluteInTheRain, 2026-11-16).
Đường dẫn tính từ gốc repo. Số liệu khảo sát đo ngày 2026-11-16 trên Wikidata/Commons, có thể lệch khi đo lại.

## 1. Mục đích và lý do

Cấp độ 3 (PDF §5) cần một MMKG nhỏ tự xây, ưu tiên tiếng Việt; khi nộp phải có data card, nguồn ảnh/văn bản,
cách tạo triple, điều kiện sử dụng (PDF §8), và tuân thủ giấy phép, không thu thập dữ liệu cá nhân nhạy cảm
không cần thiết (PDF §9). ADR-0015 đã chốt định dạng CSV; ADR-0050 đã chốt miền **di tích và di sản Việt Nam**
(Wikidata `P1435`, `P17 = Q881`) và cách làm **script thu thập tự động, thành viên duyệt**.

Tài liệu này chốt **cách thu thập và dựng bộ dữ liệu**: lấy gì, lưu ở đâu, lọc thế nào, duyệt thế nào. Mục tiêu là
một bộ dữ liệu **cố định, tái lập được, truy được nguồn từng dòng**, và trả lời được câu hỏi vấn đáp "phần nào nhóm
tự xây, vì sao bỏ dữ liệu này".

## 2. Mục tiêu, không làm, không được xảy ra

**Mục tiêu**
- Một snapshot gốc (thực thể, triple, danh sách ảnh ứng viên) lấy **toàn bộ** miền, có ngày tải, commit vào repo.
- Một bộ chính thức theo định dạng ADR-0015 (`mmkg/entities.csv`, `mmkg/triples.csv`, `mmkg/images.csv`), dựng
  từ snapshot bằng luật lọc đọc từ config; quy mô mục tiêu theo ADR-0050 mục 4: **50–150 thực thể** (tính cả
  `item` lẫn `concept`), 300–1.000 triple, 200–500 ảnh. Ước tính với ngưỡng khái niệm 2 là 191 thực thể, 508 triple,
  426 ảnh; với ngưỡng 3 (đã chọn) số thực thể dự kiến dưới 150, số chính xác đo ở lần dựng đầu.
- Kết quả duyệt của thành viên lưu riêng, không mất khi đổi luật lọc và dựng lại.
- `data_cards/mmkg_vi.md` đủ các mục bắt buộc của ADR-0015, kèm số liệu "gốc → dùng" và lý do từng luật lọc.

**Không làm** (để thiết kế/ADR sau)
- Truy vấn trên MMKG (nguồn, ngôn ngữ, cách xác định đáp án).
- Split `splits/mmkg_vi/`.
- Dựng đồ thị cho mô hình, cách dựng cạnh `contains` cho ảnh val/test, chọn mô hình (một lớp hay có cạnh khái niệm).
- Trích đặc trưng CLIP, huấn luyện, đánh giá giải thích.
- Tự viết thêm triple ngoài Wikidata (có thể bổ sung sau; định dạng đã cho phép qua cột `source`).

**Không được xảy ra**
- Snapshot gốc bị sửa tay. Mọi sửa đổi của người duyệt nằm trong file duyệt, không trong snapshot.
- Một ảnh trong bộ chính thức thiếu `license`, `author` hoặc `source_url`, hoặc có giấy phép ngoài danh sách cho phép.
- Ảnh có mặt người nhận diện được nằm trong bộ chính thức (ADR-0015 quy tắc 4).
- Đổi luật lọc làm mất kết quả duyệt đã có.
- Luật lọc đọc thông tin của split val/test (hiện chưa có split; luật lọc chỉ dùng số ảnh và triple Wikidata, không
  dùng nhãn hay kết quả truy vấn).

## 3. Ràng buộc

- ADR-0015: định dạng CSV và cột; ảnh để ngoài git (`mmkg/images/` đã gitignore), chia sẻ qua link cloud; data card
  bắt buộc; quan hệ dùng tên tiếng Anh snake_case cố định, có bảng giải nghĩa tiếng Việt.
- ADR-0015 quy tắc 4 giữ nguyên: không dùng ảnh có mặt người nhận diện được. Không viết ADR thay thế.
- ADR-0008: file văn bản nhỏ cần tái lập được commit; ảnh và dữ liệu lớn để ngoài git.
- `CLAUDE.md`: logic trong `src/mmxai/`, script mỏng trong `scripts/`, tham số trong `configs/*.yaml`, test không
  tải mạng. **Không thêm dependency**: gọi HTTP bằng thư viện chuẩn (`urllib`), đọc/ghi CSV bằng `csv`.
- Tôn trọng giới hạn của Wikimedia: User-Agent mô tả (không chứa thông tin cá nhân), nghỉ giữa các request, tự thử
  lại khi gặp HTTP 429 theo `Retry-After`. Khi khảo sát đã gặp 429 hai lần với khoảng nghỉ ngắn.

## 4. Các phương án đã cân nhắc

| Phương án | Ưu | Nhược | Chọn? |
|---|---|---|---|
| **Lấy toàn bộ làm snapshot gốc, lọc ở bước dựng bằng config** | Không mất dữ liệu; đổi ngưỡng không phải tải lại; data card báo được "gốc → dùng" | Thêm một script dựng và một config | **Được chọn** (FluteInTheRain, 2026-11-16) |
| Lọc ngay lúc thu thập | Ít file, đơn giản | Dữ liệu bị bỏ không còn; đổi ngưỡng phải tải lại; khó giải thích khi vấn đáp | Không |
| Không lọc gì | Không can thiệp | Khoảng 5.000 ảnh phải duyệt mặt người (ADR-0015); mất cân bằng giữa di tích nhiều và ít ảnh; vượt quy mô gợi ý | Không, vì công duyệt quá lớn |

## 5. Thiết kế đã chốt

### 5.1. Hai lớp dữ liệu

```
Wikidata (SPARQL) + Commons (API)
      │  scripts/collect_mmkg.py  → lấy HẾT, không lọc nội dung
      ▼
mmkg/raw/<YYYYMMDD>/                     (commit; không sửa tay)
  entities_raw.csv     mọi thực thể của miền + mọi thực thể đích của triple
  triples_raw.csv      mọi triple Wikidata trỏ tới thực thể khác (giữ nguyên mã P)
  image_candidates.csv mọi file trong P18 + danh mục P373 (chỉ metadata, chưa tải ảnh)
  meta.json            ngày tải, câu truy vấn, số dòng, phiên bản script (commit git)
      │  scripts/build_mmkg.py --config configs/mmkg_vi_build.yaml
      │  đọc mmkg/review.csv (kết quả duyệt) và áp luật lọc
      ▼
mmkg/entities.csv, mmkg/triples.csv, mmkg/images.csv   (bộ chính thức, ADR-0015)
mmkg/build_report.json                                 (số dòng bị bỏ theo từng luật)
      │  tải đúng các ảnh được chọn
      ▼
mmkg/images/<image_id>.<ext>   (gitignored, đưa lên cloud; images.csv ghi sha1 để kiểm)
```

`mmkg/raw/` có thư mục theo ngày để snapshot sau (nếu có) không ghi đè snapshot cũ. Bộ chính thức ghi rõ dựng từ
snapshot ngày nào.

### 5.2. Thu thập (snapshot gốc)

- **Thực thể của miền:** mọi item có `P1435` bất kỳ và `P17 = Q881`. Lấy nhãn `vi` và `en`, loại (`P31`), danh mục
  Commons (`P373`), ảnh đại diện (`P18`).
- **Triple:** mọi phát biểu trực tiếp (`wdt:P...`) của các thực thể này trỏ tới một item Wikidata khác. Giữ nguyên mã
  quan hệ `P...`; việc đổi sang tên snake_case nằm ở bước dựng.
- **Thực thể đích:** mọi item được trỏ tới bởi triple trên, lấy nhãn `vi`, `en`.
- **Triple giữa các thực thể đã có** (quyết định của Claude Code, chưa được nhóm duyệt): thêm mọi triple trực tiếp mà
  **cả hai đầu** đều đã nằm trong tập thực thể ở trên (ví dụ `Hà Nội –P131→ Đồng bằng sông Hồng` nếu cả hai đã có).
  Không thêm thực thể mới ở bước này, nên snapshot không phình ra; đổi lại, mô hình hai bước (ADR-0048) và giải thích
  nhiều bước có sẵn cạnh khái niệm – khái niệm mà không phải tải snapshot thứ hai.
- **Ảnh ứng viên:** file `P18` và các file nằm **trực tiếp** trong danh mục `P373` (không đi xuống danh mục con).
  Mỗi file ghi: tên file, kích thước, loại MIME, giấy phép (`LicenseShortName`), tác giả (`Artist`, bỏ thẻ HTML),
  người tải lên (`user`), sha1, và nguồn (`p18` hay `category`). URL trang file và URL ảnh **không lưu**, vì suy ra
  được từ tên file.
- **Giới hạn 1 MiB mỗi file** (pre-commit `check-added-large-files --maxkb=1024`, hook
  `.claude/hooks/guard-protected-files.sh`; không được dùng `.gz`): ước tính khoảng 5.000 dòng ảnh ứng viên với đủ cột
  là khoảng 2 MB. Quyết định của Claude Code: tách cột tác giả và người tải lên sang `image_credits.csv` (khoá theo tên
  file), và nếu một file vẫn vượt 1 MiB thì chia theo dòng thành `<tên>_000.csv`, `<tên>_001.csv`... Không nới giới hạn
  của hook. Có test kiểm mọi file trong `mmkg/` nhỏ hơn 1 MiB.

### 5.3. Luật lọc ở bước dựng (`configs/mmkg_vi_build.yaml`)

Mọi luật là **lựa chọn của nhóm**, không phải yêu cầu của đề; data card ghi lý do từng luật.

| Luật | Giá trị khởi điểm | Lý do |
|---|---|---|
| Loại file | chỉ ảnh raster (JPEG, PNG, WebP; bỏ PDF, SVG, âm thanh, video) | CLIP chỉ đọc ảnh |
| Giấy phép cho phép | public domain, CC0, CC BY, CC BY-SA (mọi phiên bản) | PDF §9, ADR-0015 quy tắc 3 |
| Kích thước tối thiểu | cạnh ngắn ≥ 224 px | Kích thước đầu vào của CLIP ViT-B-32 |
| Ảnh có `review_status = drop` | bỏ | Kết quả duyệt (mục 5.4) |
| Số ảnh tối thiểu của một di tích | 3 (đếm sau các luật trên) | Đủ để chia ảnh vào train/val/test sau này |
| Số ảnh tối đa mỗi di tích | 5, chọn tất định: ảnh `P18` trước, sau đó theo tên file | Giữ quy mô gợi ý; không để di tích nhiều ảnh lấn át |
| Quan hệ giữ lại | danh sách mã `P` trong config, kèm tên snake_case | Bỏ quan hệ quản trị (quốc gia, bản quyền, danh mục) |
| Gộp quan hệ | `P276` (location) và `P195` (collection) → `kept_at` khi trỏ cùng đích | Tránh đếm hai lần cùng một bảo tàng |
| Khái niệm giữ lại | được ≥ 3 di tích (đã giữ) dùng chung | Khái niệm chỉ nối một di tích không tạo đường đi giữa các di tích; ngưỡng 3 (thay vì 2) để tổng số thực thể nằm trong 50–150 của ADR-0050 (FluteInTheRain chọn, 2026-11-16). Nếu lần dựng đầu vẫn vượt 150 hoặc dưới 50, nhóm chỉnh ngưỡng trong config và ghi lại trong data card |
| Triple giữ lại | cả hai đầu đều là thực thể được giữ | Đồ thị không có đầu mút treo |

**Danh sách quan hệ khởi điểm** (quyết định của Claude Code, đúng danh sách đã dùng để ước tính quy mô; nhóm duyệt
trước lần dựng đầu, sửa được trong config):

| Mã Wikidata | Tên trong MMKG | Nghĩa tiếng Việt |
|---|---|---|
| `P31` | `instance_of` | là một (loại) |
| `P1435` | `heritage_designation` | có danh hiệu di sản |
| `P276`, `P195` | `kept_at` (gộp, mục trên) | được lưu giữ / đặt tại |
| `P131` | `located_in` | nằm trong đơn vị hành chính |
| `P186` | `made_from` | làm bằng chất liệu |
| `P2348` | `time_period` | thuộc thời kỳ |
| `P2596` | `culture` | thuộc nền văn hoá |
| `P180` | `depicts` | khắc hoạ |
| `P189` | `discovered_at` | phát hiện tại |
| `P361` | `part_of` | là một phần của |
| `P527` | `has_part` | gồm có |
| `P138` | `named_after` | đặt tên theo |
| `P170` | `creator` | người tạo |
| `P88` | `commissioned_by` | người đặt làm |
| `P737` | `influenced_by` | chịu ảnh hưởng từ |
| `P5422` | `has_pattern` | có hoa văn |
| `P793` | `significant_event` | sự kiện quan trọng |

**Thực thể của miền có ít hơn 3 ảnh** (quyết định của Claude Code): không làm `item`; nếu được ≥ 3 `item` dùng chung
(cùng ngưỡng khái niệm ở bảng trên; ví dụ một bảo tàng cũng có danh hiệu di sản) thì giữ làm `concept`, không thì bỏ. Ảnh của nó không vào bộ chính thức.
Thứ tự áp luật: xác định `item` trước (theo số ảnh hợp lệ), rồi mới đếm khái niệm dùng chung.

**Một file ảnh gắn với nhiều thực thể, hoặc trùng nội dung** (quyết định của Claude Code):
- File thuộc **hơn một** thực thể của miền (ví dụ là `P18` của thực thể này và nằm trong danh mục của thực thể khác) bị
  loại khỏi bộ chính thức, vì sau này đáp án truy vấn sẽ không rõ. `build_report.json` ghi số file loại vì lý do này.
- Các file **trùng sha1**: giữ file có tên nhỏ nhất theo thứ tự chữ cái, loại các file còn lại. Đây là luật tự động ở
  bước dựng; lý do `duplicate` trong file duyệt dành cho ảnh gần giống nhau mà sha1 khác.
- Nhờ hai luật trên, mỗi file trong bộ chính thức thuộc đúng một thực thể, nên `review.csv` khoá ảnh theo tên file là đủ.

**Tác giả** (quyết định của Claude Code): ảnh public domain hoặc CC0 được giữ dù `Artist` rỗng hay ghi "Unknown author";
cột `author` ghi nguyên văn, rỗng thì ghi `unknown`. Ảnh CC BY và CC BY-SA bắt buộc ghi công: nếu `Artist` rỗng thì dùng
người tải lên (`user`) và ghi `uploader: <tên>`; nếu cả hai rỗng thì loại ảnh. Data card ghi rõ luật này.

Mỗi luật ghi số dòng bị bỏ vào `mmkg/build_report.json`. Script dựng **tất định**: cùng snapshot, cùng config, cùng
file duyệt thì ra đúng cùng bộ chính thức.

### 5.4. Duyệt

- Thời điểm duyệt: sau khi chạy dựng lần đầu, chỉ duyệt **những gì nằm trong bộ chính thức** (khoảng 430 ảnh, vài trăm triple,
  dưới 150 nhãn thực thể), không duyệt toàn bộ snapshot.
- File `mmkg/review.csv`, mỗi dòng một mục, khoá theo mã ổn định:
  `kind` (`image` | `triple` | `entity`), `key` (tên file Commons, hoặc `head|P|tail`, hoặc mã Q), `review_status`
  (`ok` | `drop`), `reason`, `fix_name_vi` (chỉ cho `entity`, nhãn tiếng Việt sửa lại), `reviewer`, `date`.
- Lý do `drop` cho ảnh dùng danh sách cố định: `not_the_object` (ảnh không cho thấy hiện vật, ví dụ bảng chú
  thích, tủ kính trống), `identifiable_person`, `duplicate`, `other` (ghi chú thêm).
- Khi một ảnh bị `drop`, lần dựng sau tự lấy ảnh kế tiếp của di tích đó (nếu còn) và đánh dấu nó `pending`.
- Mục chưa có trong `review.csv` coi là `pending`. Bộ chính thức chỉ được dùng cho thí nghiệm khi không còn mục
  `pending` (script dựng có cờ kiểm tra và báo số mục còn chờ).
- Chia việc duyệt theo nhóm thực thể cho từng thành viên, để lịch sử commit thể hiện đóng góp (ADR-0015).

### 5.5. Định dạng bộ chính thức

Giữ đúng cột của ADR-0015, thêm cột cần cho truy vết (ADR-0015 không cấm thêm cột):

- `entities.csv`: `entity_id, name_vi, name_en, type, source_url` với `type` là `item` (di tích/hiện vật có ảnh)
  hoặc `concept` (thực thể đích).
- `triples.csv`: `head_id, relation, tail_id, source` với `relation` snake_case, `source` là URL thực thể Wikidata
  của đầu triple; thêm `wikidata_property` (mã `P` gốc).
- `images.csv`: `image_id, entity_id, file_name, source_url, license, author`; thêm `sha1`, `width`, `height`.
- Bảng giải nghĩa quan hệ (tên snake_case, mã `P`, nghĩa tiếng Việt) nằm trong data card.

### 5.6. Kiểm tra tự động (test không tải mạng)

- Mọi `head_id`, `tail_id`, `entity_id` có trong `entities.csv` (ADR-0015 Hệ quả).
- Mọi ảnh có `license` thuộc danh sách cho phép, có `author` và `source_url` không rỗng.
- Mỗi di tích trong bộ chính thức có từ 3 đến 5 ảnh.
- Dựng hai lần trên cùng đầu vào cho ra file giống hệt.
- Đổi ngưỡng trong config không xoá dòng nào của `review.csv`.
- Phần gọi mạng được tách để test thay bằng dữ liệu giả (không gọi Wikidata/Commons trong test).

### 5.7. Tài liệu đi kèm

- `data_cards/mmkg_vi.md` theo các mục bắt buộc của ADR-0015, có bảng số "snapshot gốc → bộ chính thức" lấy từ
  `build_report.json`, lý do từng luật lọc, tóm tắt kết quả duyệt (bao nhiêu mục bị bỏ, vì sao, ai duyệt).
- ADR mới cho quyết định "snapshot gốc + lọc ở bước dựng" và các ngưỡng (ADR-0050 chỉ chốt miền và nguồn).
- `mmkg/README.md` cập nhật luồng: thu thập → dựng → duyệt → dựng lại → tải ảnh.
- Một dòng trong `docs/ai-usage-log.md`.

## 6. Để người cài đặt tự quyết

- Chia module trong `src/mmxai/data/` (ví dụ tách phần gọi mạng, phần lọc, phần đọc/ghi CSV), tên hàm.
- Khoảng nghỉ và số lần thử lại cụ thể (bắt đầu 1,5 giây giữa request, tối đa 5 lần thử).
- Cách đặt `image_id` (ví dụ `<entity_id>_<số thứ tự>`), miễn ổn định khi dựng lại.
- Có tải ảnh bản thu nhỏ (thumbnail) thay cho bản gốc hay không, miễn cạnh ngắn vẫn ≥ 224 px và ghi rõ trong data card.
