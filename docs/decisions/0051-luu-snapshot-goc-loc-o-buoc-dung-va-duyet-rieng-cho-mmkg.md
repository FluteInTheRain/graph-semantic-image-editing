# ADR-0051: Lưu snapshot gốc nguyên vẹn, lọc MMKG ở bước dựng bằng config, lưu kết quả duyệt riêng

- **Trạng thái:** Đề xuất
- **Ngày:** 2026-11-16

## Bối cảnh

ADR-0050 đã chốt **miền** của MMKG tiếng Việt (di tích và di sản Việt Nam: Wikidata `P1435`, `P17 = Q881`) và **nguồn** (Wikidata + Wikimedia Commons, thu thập tự động rồi nhóm duyệt). ADR-0050 chưa chốt: lấy dữ liệu bằng cách nào, lưu ở đâu, lọc theo luật nào, duyệt ra sao. ADR này chốt các điểm đó. Thiết kế chi tiết đã được duyệt ở `docs/superpowers/specs/2026-11-16-mmkg-vi-thu-thap-design.md` (FluteInTheRain, 2026-11-16).

Khái niệm cho người mới:

- **Snapshot** = bản chụp dữ liệu tại một ngày. Wikidata và Commons thay đổi liên tục; nếu không lưu lại bản đã tải thì lần chạy sau ra dữ liệu khác và không tái lập được.
- **Luật lọc** = điều kiện để một ảnh/triple/thực thể được vào bộ chính thức (ví dụ "ảnh phải có giấy phép CC BY"). Mọi luật trong ADR này là **lựa chọn của nhóm**, không phải yêu cầu của đề.
- **SPARQL endpoint** = địa chỉ web nhận câu truy vấn SPARQL và trả kết quả. Wikidata có endpoint chính thức (WDQS, `query.wikidata.org`); QLever là một endpoint khác, chạy trên bản dump của Wikidata.
- **HTTP 429 và `Retry-After`** = máy chủ báo "gửi quá nhiều request" và nói phải chờ bao nhiêu giây trước khi gửi lại.

Ràng buộc:

- PDF §8: data card phải nêu nguồn ảnh/văn bản, cách tạo triple, điều kiện sử dụng. PDF §9: ghi nguồn, tuân thủ giấy phép, không thu thập dữ liệu cá nhân nhạy cảm.
- ADR-0015: định dạng CSV, ảnh để ngoài git, không dùng ảnh có mặt người nhận diện được.
- Repo chặn file commit từ 1 MiB trở lên (pre-commit `check-added-large-files --maxkb=1024`).
- Vấn đáp sẽ hỏi "phần nào nhóm tự xây, vì sao bỏ dữ liệu này", nên mỗi dòng bị bỏ phải đếm được theo luật.

## Quyết định

### 1. Hai lớp dữ liệu và một file duyệt riêng

```
Wikidata (SPARQL qua QLever) + Commons (API)
      │  scripts/collect_mmkg.py --config configs/mmkg_vi_collect.yaml   → lấy HẾT, không lọc
      ▼
mmkg/raw/<YYYYMMDD>/   snapshot gốc, commit nguyên trạng, KHÔNG sửa tay
      │  scripts/build_mmkg.py --config configs/mmkg_vi_build.yaml       → áp luật lọc + kết quả duyệt
      ▼
mmkg/entities.csv, mmkg/triples.csv, mmkg/images.csv, mmkg/build_report.json   (bộ chính thức, ADR-0015)
mmkg/review.csv        kết quả duyệt của thành viên (script chỉ thêm dòng `pending`, không ghi đè)
```

| Lớp | Nội dung | Ai ghi | Quy tắc |
|---|---|---|---|
| Snapshot gốc `mmkg/raw/<YYYYMMDD>/` | Toàn bộ thực thể, triple, ảnh ứng viên của miền, kèm `meta.json` (ngày, câu truy vấn, endpoint, commit git) | `scripts/collect_mmkg.py` (logic: `collect_snapshot`, `write_snapshot` trong `src/mmxai/data/mmkg_collect.py`) | Commit nguyên trạng, không sửa tay; script từ chối chạy hai lần trong cùng một ngày để không ghi đè |
| Bộ chính thức `mmkg/*.csv` | Bộ theo định dạng ADR-0015, dựng lại được bất cứ lúc nào | `scripts/build_mmkg.py` (logic: `build_mmkg` trong `src/mmxai/data/mmkg_build.py`) | Tất định: cùng snapshot + config + file duyệt thì ra đúng cùng file |
| Kết quả duyệt `mmkg/review.csv` | Mỗi dòng một quyết định `ok`/`drop` cho một ảnh, triple hoặc thực thể | Thành viên sửa cột duyệt; script dựng chỉ **thêm** dòng `pending` cho mục mới (`append_pending`) | Đổi ngưỡng rồi dựng lại không làm mất dòng duyệt nào |

Vì sao tách: đổi một ngưỡng (ví dụ số ảnh tối đa) chỉ cần chạy lại bước dựng, không phải tải lại dữ liệu, và công duyệt đã làm vẫn còn.

### 2. Luật lọc và ngưỡng (`configs/mmkg_vi_build.yaml`)

Mọi luật là **lựa chọn của nhóm**. Bảng xếp đúng thứ tự code áp dụng: luật đầu nằm trong `build_mmkg`, các luật ảnh tiếp theo trong `select_images` (`src/mmxai/data/mmkg_build.py`).

| Luật | Giá trị | Lý do |
|---|---|---|
| Thực thể bị duyệt `drop` | bỏ mọi ảnh của thực thể đó | Kết quả duyệt; thực thể bị bỏ không được làm `item` hay `concept` |
| File thuộc hơn một thực thể | bỏ | Không biết ảnh cho thấy thực thể nào; sau này đáp án truy vấn sẽ không rõ |
| Loại file (MIME) | chỉ `image/jpeg`, `image/png`, `image/webp` | CLIP chỉ đọc ảnh raster (bỏ SVG, GIF, PDF, âm thanh, video) |
| Giấy phép | `Public domain`, `CC0`, mọi bản `CC BY ` và `CC BY-SA ` | PDF §9, ADR-0015 quy tắc 3 |
| Kích thước | cạnh ngắn ≥ 224 px | Kích thước đầu vào của CLIP ViT-B-32 |
| Ghi công | CC BY / CC BY-SA bắt buộc có tác giả thật: nếu `Artist` không nêu ai (xem dưới bảng) thì ghi `uploader: <tên người tải lên>`; không có người tải lên thì bỏ ảnh. Public domain / CC0 thì ghi `unknown` | Giấy phép CC BY / CC BY-SA yêu cầu ghi công (`pick_author`) |
| Ảnh bị duyệt `drop` | bỏ | Kết quả duyệt |
| Trùng `sha1` | giữ file có tên nhỏ nhất theo thứ tự chữ cái | Hai file giống hệt nhau từng byte |
| Số ảnh mỗi di tích | 3–5; dưới 3 thì di tích không thành `item`; trên 5 thì chọn ảnh `P18` trước, rồi theo tên file | Đủ ảnh để chia train/val/test sau này; không để di tích nhiều ảnh lấn át |
| Khái niệm (`concept`) | được ≥ 3 `item` cùng trỏ tới | Khái niệm chỉ nối một di tích không tạo đường đi giữa các di tích. FluteInTheRain chọn 3 (thay vì 2) để tổng số thực thể nằm trong 50–150 (ADR-0050) |
| Quan hệ | 18 mã `P` trong config, đổi sang 17 tên snake_case; `P276` và `P195` cùng thành `kept_at` và gộp lại khi trỏ cùng đích | Bỏ quan hệ quản trị (quốc gia, bản quyền, danh mục). **Danh sách do Claude Code đề xuất, nhóm chưa duyệt** |
| Triple | cả hai đầu là thực thể được giữ, không tự trỏ về chính nó | Đồ thị không có đầu mút treo |

**Chuẩn hoá tác giả** (`normalize_artist`, `names_an_author` trong `src/mmxai/data/mmkg_build.py`): trước khi áp luật ghi công, chữ `Artist` được làm sạch: bỏ ký tự rộng 0 (U+200B), gộp một cụm bị lặp liền nhau ("Unknown authorUnknown author." → "Unknown author"), bỏ dấu câu ở cuối. Sau đó, nếu chữ (không phân biệt hoa thường) là `unknown`, `unknown author`, `author unknown`, `anonymous` hoặc chứa `hereby publish` (câu mẫu giấy phép dán nhầm vào ô tác giả) thì coi như **không có tác giả**. Lý do: lần dựng đầu có 11 ảnh mang tác giả hỏng như vậy, 6 ảnh trong đó là CC BY-SA 3.0, tức là không ghi công được đúng giấy phép. Số trước/sau ở `data_cards/mmkg_vi.md` mục 3. Gốc của lỗi dính chữ nằm ở bước thu thập: `strip_html` (`src/mmxai/data/mmkg_collect.py`) từng xoá thẳng thẻ HTML nên hai thẻ "Unknown author" liền nhau dính thành một từ; nay mỗi thẻ được thay bằng một dấu cách, áp dụng cho các lần thu thập sau.

Danh sách 18 mã `P` (17 tên quan hệ) và nghĩa tiếng Việt nằm trong `data_cards/mmkg_vi.md`. Các quyết định nhỏ khác của Claude Code (chưa được nhóm duyệt, ghi trong design §5.2–5.3): thu thêm triple giữa các thực thể đích đã có; thực thể của miền có dưới 3 ảnh vẫn được làm `concept` nếu đủ 3 `item` trỏ tới.

### 3. Cách lấy dữ liệu

| Việc | Cách làm | Lý do |
|---|---|---|
| Thực thể và triple | SPARQL qua **QLever** `https://qlever.dev/api/wikidata`; chuỗi mô tả index của endpoint được ghi vào `meta.json` (khoá `sparql_index`) | Ngày 2026-11-16, WDQS trả HTTP 429 yêu cầu chờ 1000 s ngay ở câu truy vấn đầu; người dùng (FluteInTheRain) yêu cầu cách thu thập nhanh nhất. QLever chạy trên bản dump Wikidata, trả cùng định dạng JSON |
| Ảnh và metadata ảnh | Commons API (`commons.wikimedia.org/w/api.php`) | Nguồn duy nhất có giấy phép, tác giả, `sha1`, kích thước từng file |
| Giới hạn tốc độ và thử lại | Nghỉ 1,5 s giữa request, tối đa 5 lần thử. Chỉ thử lại khi gặp 429 và 5xx (500/502/503/504) và lỗi mạng tạm thời (`URLError`, hết thời gian chờ, kết nối bị ngắt, `IncompleteRead`); mã khác (400, 404, 501...) báo lỗi ngay. Có `Retry-After` hợp lệ thì chờ đúng số giây đó nếu ≤ 1800 s, quá 1800 s thì dừng với lỗi; không có thì chờ tăng dần (1,5 s × 2^lần thử). Mỗi lần chờ in một dòng log. Hết lượt thử thì lỗi chỉ ghi máy chủ và đường dẫn, không ghi cả câu truy vấn (`WikimediaClient._get` trong `src/mmxai/data/wikimedia.py`) | Chính sách Wikimedia yêu cầu tôn trọng `Retry-After`; gửi lại sớm có thể bị chặn lâu hơn. Lỗi mạng chốc lát không nên làm hỏng một lần thu thập dài |
| Tên bản dump (`sparql_index`) | Đọc `?cmd=stats` của endpoint **đầu tiên**, trước mọi câu truy vấn. Không đọc được (lỗi mạng, `RuntimeError` sau khi client đã thử hết lượt, JSON sai, thiếu khoá) thì ghi `unknown` và in cảnh báo (`_sparql_index` trong `src/mmxai/data/mmkg_collect.py`) | Chỉ là thông tin nguồn gốc, không đáng dừng cả lần thu thập; đọc trước để tên dump khớp dữ liệu các câu truy vấn đọc |
| Câu trả lời rỗng | Câu `domain` trả 0 thực thể hoặc câu `triples` trả 0 dòng thì dừng với `RuntimeError` | Rỗng nghĩa là sai config hoặc endpoint hỏng, không phải miền rỗng; lưu lại sẽ đóng băng một snapshot rỗng |
| File ảnh không có thông tin | `meta.json` khoá `missing_files` ghi `{name, reason}`, `reason` là `missing` (đã xoá), `invalid` (tên sai), `no_imageinfo` hoặc `unresolved_redirect`; script in 10 file đầu và đường dẫn `meta.json` | Biết vì sao thiếu để kiểm tay. Snapshot `20261116` thu trước thay đổi này nên chỉ ghi tên file |
| User-Agent | `mmxai-course-project/0.1 (research script)`, không có email hay tên | Không công bố thông tin cá nhân; nhóm chưa đồng ý đăng thông tin liên hệ |
| Tải ảnh | Thumbnail rộng `download_width` = 960 px (`scripts/download_mmkg_images.py`, logic `download_images` trong `src/mmxai/data/mmkg_images.py`). Tải **bản gốc** và so `sha1` với Commons khi: bản gốc rộng ≤ 960 px (Commons không phóng to, nên "thumbnail" khi đó chính là bản gốc), Commons không trả `thumburl`, hoặc thumbnail có cạnh ngắn < 224 px. Cột `why` của `mmkg/images_manifest.csv` ghi lý do: `thumbnail`, `original_not_wider`, `original_no_thumburl`, `original_thumb_too_small`, `original_already_present`. Manifest được ghi lại **sau từng file**. File trên đĩa sai `sha1` bị xoá (có log) rồi tải lại. File trong `mmkg/images/` không có trong `images.csv` (ảnh đã bị bỏ, file `.part` dở) làm script dừng mã 1 trước mọi request, trừ khi chạy với `--delete-stray` (xoá và log từng file). Mọi file sai `sha1` hoặc không tìm thấy được ghi tên ra stderr và vào `mmkg/download_report.json`; còn lỗi thì thoát mã 1 | Tải bản gốc bị chặn tốc độ (429) ngày 2026-11-16; chính sách robot cho phép thumbnail tạo sẵn; CLIP chỉ cần 224 px. Ảnh bị duyệt bỏ (ví dụ có mặt người) không được lọt lên bản chia sẻ |
| Giới hạn 1 MiB | Tác giả/người tải lên tách sang `image_credits.csv`; bảng nào vẫn vượt 1 MiB thì chia thành `<tên>_000.csv`, `<tên>_001.csv`... (`write_rows` trong `src/mmxai/data/mmkg_csv.py`) | Không nới giới hạn của hook |

## Các phương án đã cân nhắc

| Phương án | Ưu | Nhược | Lý do không chọn |
|---|---|---|---|
| **Snapshot gốc + lọc ở bước dựng bằng config, duyệt lưu riêng** | Không mất dữ liệu; đổi ngưỡng không phải tải lại; data card báo được "gốc → dùng" theo từng luật; công duyệt không mất | Thêm một script dựng, một config và một file duyệt | **Được chọn** (FluteInTheRain, 2026-11-16) |
| Lọc ngay lúc thu thập | Ít file, đơn giản | Dữ liệu bị bỏ không còn; đổi ngưỡng phải tải lại; khó giải thích khi vấn đáp | Không truy được "vì sao bỏ" |
| Không lọc gì | Không can thiệp | 5.093 ảnh ứng viên phải duyệt mặt người (ADR-0015); lệch giữa di tích nhiều và ít ảnh; vượt quy mô gợi ý | Công duyệt quá lớn |
| Chỉ dùng WDQS (`query.wikidata.org`) | Endpoint chính thức, dữ liệu mới nhất | Ngày 2026-11-16 trả 429 yêu cầu chờ 1000 s ngay câu đầu | Quá chậm so với yêu cầu thu thập nhanh nhất |
| **QLever trên dump Wikidata** | Nhanh, cùng định dạng kết quả, ghi được tên bản dump vào `meta.json` | Dữ liệu theo bản dump nên có thể lệch Wikidata trực tiếp; không có sẵn `PREFIX` nên mỗi câu truy vấn phải khai báo | **Được chọn** cho phần SPARQL |
| Tự tải bản dump Wikidata | Toàn quyền, biết chính xác ngày dữ liệu | Phải tải và nạp toàn bộ Wikidata vào kho riêng chỉ để lấy một miền nhỏ | Quá nặng so với quy mô 50–150 thực thể |

## Hệ quả

- **Dễ hơn:** mỗi dòng bị bỏ có tên luật trong `mmkg/build_report.json`; dựng lại tất định; snapshot gốc làm bằng chứng nguồn.
- **Rò rỉ dữ liệu (ADR-0009):** luật dựng chỉ dùng số ảnh hợp lệ và triple Wikidata, không đọc split, nhãn hay kết quả truy vấn (hiện chưa có split). Nhưng `images.csv` ghi ảnh thuộc thực thể nào; khi dựng đồ thị, cạnh `ảnh → contains → thực thể` cho ảnh val/test **không được** lấy từ `images.csv` vì đó chính là đáp án. Cách dựng cạnh này để cho ADR sau, theo nguyên tắc ADR-0046, và gọi `leakage-auditor` khi cài đặt.
- **Giải thích trung thực (ADR-0016):** mô hình một lớp (ADR-0046) bỏ qua triple thực thể – thực thể, nên các triple này chỉ được xuất hiện trong giải thích nếu dùng mô hình có cạnh khái niệm – khái niệm (ADR-0048). Quyết định ở ADR sau.
- **Biên quy mô hẹp:** lần dựng đầu có 331 triple, chỉ hơn mốc tối thiểu 300 là 31. Nếu duyệt bỏ nhiều triple hoặc thực thể, có thể phải đổi ngưỡng (ví dụ `min_items_per_concept`); khi đó ghi lý do và số mới vào data card.
- **Một số di tích bị loại vì thiếu ảnh hợp lệ:** ví dụ trống đồng Ngọc Lũ (`Q3514747`) có 6 file trên Commons nhưng chỉ 2 qua luật (2 GIF và 1 SVG bị loại vì MIME, 1 JPEG cạnh ngắn 200 px), nên không thành `item`.
- **Danh sách quan hệ chưa được nhóm duyệt:** cần duyệt và đổi trạng thái ADR này trước khi dùng bộ dữ liệu cho thí nghiệm.
- **File duyệt chịu được bảng tính:** `read_review` nhận file có BOM (Excel), ô thừa khoảng trắng, cột đổi thứ tự; thiếu hoặc thừa cột thì báo lỗi nêu tên cột. Dòng duyệt có `key` không khớp gì trong snapshot gốc (thường là gõ sai) được liệt kê ở `review_keys_unmatched` trong `build_report.json`, và `--require-reviewed` thoát mã 1 khi danh sách này khác rỗng.
- **Việc tiếp theo:** thành viên duyệt `mmkg/review.csv`; dựng lại tới khi `scripts/build_mmkg.py --require-reviewed` qua; tải ảnh (`scripts/download_mmkg_images.py`) và đưa lên cloud; ADR cho split `splits/mmkg_vi/`, cạnh `contains` và truy vấn trên MMKG.
- **Ghi chú 2026-11-17 (sau duyệt sơ bộ bằng AI):** các số "lần dựng đầu" ở trên (331 triple, mục `pending`) là của bản trước duyệt. Một AI pre-review (Claude Code, `reviewer = claude-ai`, 6 vòng) đã quyết định mọi dòng của `mmkg/review.csv` và `scripts/build_mmkg.py --require-reviewed` đã qua. Bộ chính thức hiện có 119 thực thể, 302 triple, 382 ảnh (`mmkg/build_report.json`); 302 chỉ hơn mốc 300 là 2, nên "biên quy mô hẹp" nghiêm trọng hơn. Đây **chưa phải duyệt của người**: thành viên vẫn phải kiểm các dòng `drop` và 36 ghi chú trong `mmkg/review_ai_notes.csv`, và ADR này chưa đổi trạng thái. Chi tiết và số: `data_cards/mmkg_vi.md` mục 7.

## Nguồn tham khảo

- Đề bài `Project03_Multimodal_XAI.pdf`: §4.1, §5 (cấp độ 3), §8, §9.
- Thiết kế đã duyệt: `docs/superpowers/specs/2026-11-16-mmkg-vi-thu-thap-design.md` §4–5.
- ADR-0015, ADR-0050; ADR-0009, ADR-0016, ADR-0046, ADR-0048.
- Số liệu: `mmkg/build_report.json`, `mmkg/raw/20261116/meta.json`, `data_cards/mmkg_vi.md`.
- QLever Wikidata: https://qlever.dev/wikidata
- Wikimedia User-Agent policy: https://meta.wikimedia.org/wiki/User-Agent_policy
- Wikimedia API etiquette (`Retry-After`, giới hạn tốc độ): https://www.mediawiki.org/wiki/API:Etiquette
