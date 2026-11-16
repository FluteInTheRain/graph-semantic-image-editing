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

Mọi luật là **lựa chọn của nhóm**. Thứ tự áp dụng theo `select_images` trong `src/mmxai/data/mmkg_build.py`.

| Luật | Giá trị | Lý do |
|---|---|---|
| File thuộc hơn một thực thể | bỏ | Không biết ảnh cho thấy thực thể nào; sau này đáp án truy vấn sẽ không rõ |
| Loại file (MIME) | chỉ `image/jpeg`, `image/png`, `image/webp` | CLIP chỉ đọc ảnh raster (bỏ SVG, GIF, PDF, âm thanh, video) |
| Giấy phép | `Public domain`, `CC0`, mọi bản `CC BY ` và `CC BY-SA ` | PDF §9, ADR-0015 quy tắc 3 |
| Ghi công | CC BY / CC BY-SA bắt buộc có tác giả: `Artist` rỗng thì ghi `uploader: <tên người tải lên>`; cả hai rỗng thì bỏ ảnh. Public domain / CC0 rỗng tác giả thì ghi `unknown` | Giấy phép CC BY / CC BY-SA yêu cầu ghi công (`pick_author`) |
| Kích thước | cạnh ngắn ≥ 224 px | Kích thước đầu vào của CLIP ViT-B-32 |
| Ảnh bị duyệt `drop` | bỏ | Kết quả duyệt |
| Trùng `sha1` | giữ file có tên nhỏ nhất theo thứ tự chữ cái | Hai file giống hệt nhau từng byte |
| Số ảnh mỗi di tích | 3–5; dưới 3 thì di tích không thành `item`; trên 5 thì chọn ảnh `P18` trước, rồi theo tên file | Đủ ảnh để chia train/val/test sau này; không để di tích nhiều ảnh lấn át |
| Khái niệm (`concept`) | được ≥ 3 `item` cùng trỏ tới | Khái niệm chỉ nối một di tích không tạo đường đi giữa các di tích. FluteInTheRain chọn 3 (thay vì 2) để tổng số thực thể nằm trong 50–150 (ADR-0050) |
| Quan hệ | 18 mã `P` trong config, đổi sang tên snake_case; `P276` và `P195` cùng thành `kept_at` và gộp lại khi trỏ cùng đích | Bỏ quan hệ quản trị (quốc gia, bản quyền, danh mục). **Danh sách do Claude Code đề xuất, nhóm chưa duyệt** |
| Triple | cả hai đầu là thực thể được giữ, không tự trỏ về chính nó | Đồ thị không có đầu mút treo |

Danh sách 18 mã quan hệ và nghĩa tiếng Việt nằm trong `data_cards/mmkg_vi.md`. Các quyết định nhỏ khác của Claude Code (chưa được nhóm duyệt, ghi trong design §5.2–5.3): thu thêm triple giữa các thực thể đích đã có; thực thể của miền có dưới 3 ảnh vẫn được làm `concept` nếu đủ 3 `item` trỏ tới.

### 3. Cách lấy dữ liệu

| Việc | Cách làm | Lý do |
|---|---|---|
| Thực thể và triple | SPARQL qua **QLever** `https://qlever.dev/api/wikidata`; chuỗi mô tả index của endpoint được ghi vào `meta.json` (khoá `sparql_index`) | Ngày 2026-11-16, WDQS trả HTTP 429 yêu cầu chờ 1000 s ngay ở câu truy vấn đầu; người dùng (FluteInTheRain) yêu cầu cách thu thập nhanh nhất. QLever chạy trên bản dump Wikidata, trả cùng định dạng JSON |
| Ảnh và metadata ảnh | Commons API (`commons.wikimedia.org/w/api.php`) | Nguồn duy nhất có giấy phép, tác giả, `sha1`, kích thước từng file |
| Giới hạn tốc độ | Nghỉ 1,5 s giữa request, tối đa 5 lần thử; gặp 429 thì chờ đúng `Retry-After` nếu ≤ 1800 s và in một dòng log, quá 1800 s thì dừng với lỗi (`WikimediaClient` trong `src/mmxai/data/wikimedia.py`) | Chính sách Wikimedia yêu cầu tôn trọng `Retry-After`; gửi lại sớm có thể bị chặn lâu hơn |
| User-Agent | `mmxai-course-project/0.1 (research script)`, không có email hay tên | Không công bố thông tin cá nhân; nhóm chưa đồng ý đăng thông tin liên hệ |
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
- **Việc tiếp theo:** thành viên duyệt `mmkg/review.csv`; dựng lại tới khi `scripts/build_mmkg.py --require-reviewed` qua; tải ảnh (`scripts/download_mmkg_images.py`) và đưa lên cloud; ADR cho split `splits/mmkg_vi/`, cạnh `contains` và truy vấn trên MMKG.

## Nguồn tham khảo

- Đề bài `Project03_Multimodal_XAI.pdf`: §4.1, §5 (cấp độ 3), §8, §9.
- Thiết kế đã duyệt: `docs/superpowers/specs/2026-11-16-mmkg-vi-thu-thap-design.md` §4–5.
- ADR-0015, ADR-0050; ADR-0009, ADR-0016, ADR-0046, ADR-0048.
- Số liệu: `mmkg/build_report.json`, `mmkg/raw/20261116/meta.json`, `data_cards/mmkg_vi.md`.
- QLever Wikidata: https://qlever.dev/wikidata
- Wikimedia User-Agent policy: https://meta.wikimedia.org/wiki/User-Agent_policy
- Wikimedia API etiquette (`Retry-After`, giới hạn tốc độ): https://www.mediawiki.org/wiki/API:Etiquette
