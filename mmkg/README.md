# mmkg/ — MMKG tiếng Việt tự xây (cấp độ 3)

**MMKG là gì?** Đồ thị tri thức đa phương thức: các **thực thể** (ví dụ "Văn Miếu – Quốc Tử Giám", "tượng", "đồng điếu") nối với nhau bằng **triple** (đầu, quan hệ, đuôi) và mỗi di tích có **ảnh** minh hoạ.

Miền: **di tích và di sản Việt Nam** (Wikidata `P1435`, `P17 = Q881`, ADR-0050). Dữ liệu lấy tự động từ Wikidata + Wikimedia Commons, lọc bằng luật trong config, rồi thành viên duyệt (ADR-0051). Nguồn, giấy phép, thống kê, hạn chế: `data_cards/mmkg_vi.md`.

**Trạng thái (2026-11-17):** đã thu thập snapshot `raw/20261116/` và dựng (lần sau cùng với luật tác giả mới): 84 `item` + 44 `concept`, 331 triple, 403 ảnh (`build_report.json`). **Chưa duyệt**, chưa dùng cho thí nghiệm.

## 1. Luồng làm việc

```
1. collect   → raw/<YYYYMMDD>/          (một lần; commit nguyên trạng, không sửa tay)
2. build     → entities/triples/images.csv, build_report.json, thêm dòng pending vào review.csv
3. review    → thành viên sửa review.csv (ok / drop)
4. rebuild   → build lại; lặp 3–4 tới khi --require-reviewed qua
5. download  → images/ (gitignored) + images_manifest.csv, kiểm sha1, đưa lên cloud
```

| Bước | Lệnh (chạy từ gốc repo) | Logic |
|---|---|---|
| 1. Thu thập | `uv run python scripts/collect_mmkg.py --config configs/mmkg_vi_collect.yaml` | `collect_snapshot`, `write_snapshot` trong `src/mmxai/data/mmkg_collect.py` |
| 2. Dựng | `uv run python scripts/build_mmkg.py --config configs/mmkg_vi_build.yaml` | `build_mmkg`, `write_built_mmkg` trong `src/mmxai/data/mmkg_build.py` |
| 3. Duyệt | Sửa `mmkg/review.csv` (mục 3) | `read_review` kiểm định dạng khi dựng |
| 4. Dựng lại + kiểm | `uv run python scripts/build_mmkg.py --config configs/mmkg_vi_build.yaml --require-reviewed` | Thoát mã 1 nếu còn mục `pending` (in số mục) hoặc có dòng duyệt không khớp gì trong snapshot (in danh sách `review_keys_unmatched`) |
| 5. Tải ảnh | `uv run python scripts/download_mmkg_images.py --collect-config configs/mmkg_vi_collect.yaml --build-config configs/mmkg_vi_build.yaml` (thêm `--delete-stray` khi script báo file lạ) | `download_images`, `find_stray_files` trong `src/mmxai/data/mmkg_images.py` |

- Bước 1 cần mạng, chạy lâu vì nghỉ giữa các request; script từ chối chạy lại trong cùng một ngày để không ghi đè snapshot. Muốn thu thập lại thì đổi `snapshot_dir` trong `configs/mmkg_vi_build.yaml` sang thư mục ngày mới và ghi lý do trong data card.
- Bước 2 và 4 không cần mạng, tất định (cùng đầu vào ra cùng file). Đổi ngưỡng trong `configs/mmkg_vi_build.yaml` rồi dựng lại không làm mất dòng duyệt nào.
- Bước 2 in số mục chính, mọi luật có dòng bị bỏ, đường dẫn `build_report.json`, cảnh báo khi có thực thể thiếu `name_vi`, và báo "a new one will be created" khi chưa có `review.csv`.
- Bước 5 tải thumbnail rộng 960 px từ các kích thước Commons tạo sẵn (`download_width` trong config). Lý do: tải bản gốc bị Wikimedia chặn tốc độ (HTTP 429) ngày 2026-11-16, chính sách robot cho phép thumbnail tạo sẵn, và CLIP chỉ cần 224 px. Script tải **bản gốc** (và so `sha1` với Commons; sai thì không lưu) khi bản gốc rộng ≤ 960 px, khi Commons không có thumbnail, hoặc khi thumbnail có cạnh ngắn < 224 px; cột `why` của manifest ghi lý do (`data_cards/mmkg_vi.md` mục 8).
- Bước 5 dừng mã 1 **trước mọi request** nếu `images/` có file lạ (không phải `image_id` của `images.csv`, ví dụ ảnh vừa bị `drop`); `--delete-stray` xoá từng file và in tên. Manifest được ghi lại sau từng file. Mọi file sai `sha1` hoặc không tìm thấy được in tên ra stderr và ghi vào `download_report.json`; còn lỗi thì thoát mã 1.
- **Kiểm một bản copy ảnh:** `shasum -a 1 images/<file>` rồi so với cột `sha1` của dòng `image_id` tương ứng trong `images_manifest.csv`. Cột `sha1` của `images.csv` là của bản gốc trên Commons nên chỉ khớp file `kind = original`.

## 2. Các file trong `mmkg/`

| File | Nội dung | Ai ghi | Commit? |
|---|---|---|---|
| `raw/<YYYYMMDD>/entities_raw.csv` | Mọi thực thể của miền và mọi thực thể đích: `entity_id, name_vi, name_en, in_domain, commons_category, p18_files` | Bước 1 | Có, không sửa tay |
| `raw/<YYYYMMDD>/triples_raw.csv` | Mọi triple Wikidata tới thực thể khác: `head_id, property, tail_id` (mã `P` gốc) | Bước 1 | Có, không sửa tay |
| `raw/<YYYYMMDD>/image_candidates.csv` | Mọi file ảnh ứng viên: `file_name, entity_id, origin` (`p18`/`category`), `mime, width, height, license, sha1` | Bước 1 | Có, không sửa tay |
| `raw/<YYYYMMDD>/image_credits.csv` | Tác giả và người tải lên của từng file: `file_name, artist, uploader` | Bước 1 | Có, không sửa tay |
| `raw/<YYYYMMDD>/meta.json` | Ngày thu thập, 4 câu SPARQL, số dòng, endpoint (`sparql_url`, `sparql_index`), commit git, config | Bước 1 | Có, không sửa tay |
| `entities.csv` | `entity_id, name_vi, name_en, type, source_url`; `type` là `item` (có ảnh) hoặc `concept` | Bước 2 | Có |
| `triples.csv` | `head_id, relation, tail_id, source, wikidata_property` | Bước 2 | Có |
| `images.csv` | `image_id, entity_id, file_name, source_url, license, author, sha1, width, height`. **Cảnh báo rò rỉ:** `entity_id` là đáp án "ảnh này là di tích nào"; không dùng nó để dựng cạnh `contains` cho ảnh val/test (ADR-0051, Hệ quả) | Bước 2 | Có |
| `build_report.json` | Số dòng gốc, số dòng chính thức, số dòng bị bỏ theo từng luật, số mục `pending`, `review_keys_unmatched` (dòng duyệt không khớp gì) | Bước 2 | Có |
| `review.csv` | Kết quả duyệt (mục 3) | Bước 2 thêm dòng `pending`; thành viên sửa | Có |
| `images/` | File ảnh `<image_id><đuôi>`: thumbnail 960 px, hoặc bản gốc trong các trường hợp ở bước 5 | Bước 5 | **Không** (gitignore, chia sẻ qua link cloud) |
| `images_manifest.csv` | `image_id, file_name, kind, why, url, width, height, bytes, sha1` của các ảnh trong `images.csv` hiện tại đang có trên đĩa; `kind` là `thumbnail` hoặc `original`, `why` là `thumbnail`, `original_not_wider`, `original_no_thumburl`, `original_thumb_too_small` hoặc `original_already_present` | Bước 5 | Có |
| `download_report.json` | Số đếm của lần tải gần nhất và tên từng file lỗi (`sha1_mismatches`, `missing`) | Bước 5 | Nhóm quyết định |

**Giới hạn 1 MiB:** repo chặn file commit từ 1 MiB trở lên. Bảng nào vượt sẽ được ghi thành nhiều mảnh `<tên>_000.csv`, `<tên>_001.csv`..., mỗi mảnh lặp lại dòng tiêu đề (`write_rows` trong `src/mmxai/data/mmkg_csv.py`); code đọc (`read_rows`) tự ghép lại. Hiện mọi bảng đều nằm trong một file. `tests/test_mmkg_committed.py` kiểm mọi file trong `mmkg/` (trừ `images/`) nhỏ hơn 1 MiB.

## 3. Cách duyệt một dòng

Mỗi dòng của `mmkg/review.csv` là một quyết định cho **một** mục của bộ chính thức:

| Cột | Ý nghĩa |
|---|---|
| `kind` | `image`, `triple` hoặc `entity` |
| `key` | `image`: tên file Commons (`file_name`); `triple`: `head_id\|P...\|tail_id` với mã `P` gốc; `entity`: mã `Q` |
| `review_status` | `pending` (chưa duyệt), `ok` hoặc `drop` |
| `reason` | Lý do khi `drop`. Với ảnh bắt buộc là một trong: `not_the_object` (ảnh không cho thấy di tích/hiện vật, ví dụ bảng chú thích, tủ kính trống), `identifiable_person` (có mặt người nhận diện được, ADR-0015 quy tắc 4), `duplicate` (gần giống ảnh khác dù `sha1` khác), `other` (ghi chú thêm) |
| `fix_name_vi` | Chỉ cho `entity`: nhãn tiếng Việt sửa lại (ví dụ điền cho 7 thực thể thiếu `name_vi`) |
| `reviewer` | Tên người duyệt |
| `date` | Ngày duyệt, `YYYY-MM-DD` |

Các bước:

1. **Ảnh:** mở `https://commons.wikimedia.org/wiki/File:<file_name>` (hoặc cột `source_url` trong `images.csv`). Xem ảnh có đúng di tích/hiện vật không, có mặt người nhận diện được không. Kiểm luôn cột `author` và `license` của dòng đó trong `images.csv` so với trang Commons: tác giả ghi công đúng người chưa, giấy phép có khớp không. Sai mà không sửa được thì `drop` với lý do `other` và ghi chú (ví dụ `Bia Vinh Lang.JPG`, data card mục 3).
2. **Triple:** mở `https://www.wikidata.org/wiki/<head_id>`, tìm phát biểu có mã `P` trong `key`, kiểm nó đúng và có nghĩa.
3. **Thực thể:** mở `https://www.wikidata.org/wiki/<Q...>`, kiểm nhãn tiếng Việt; sai hoặc thiếu thì ghi vào `fix_name_vi` (giữ `review_status = ok`).
4. **Sửa ngay dòng `pending` có sẵn**, đổi `review_status` thành `ok` hoặc `drop`, điền `reason`, `reviewer`, `date`. Không thêm dòng thứ hai cho cùng (`kind`, `key`): bước dựng sẽ báo lỗi `repeated (kind, key)`.
5. Dựng lại (bước 4). Ảnh bị `drop` được thay bằng ảnh kế tiếp của di tích đó (nếu còn), và ảnh mới xuất hiện dưới dạng dòng `pending` mới ở cuối file.

Lưu ý:

- Triple `kept_at` gộp từ `P195` và `P276` có **hai** dòng duyệt (một cho mỗi mã). `drop` chỉ một dòng thì triple **vẫn còn** (mã kia vẫn sinh ra nó); muốn bỏ triple thì `drop` cả hai dòng.
- `drop` một thực thể sẽ bỏ luôn mọi ảnh và triple của nó.
- Script dựng chỉ **thêm** dòng vào cuối `review.csv`, không bao giờ ghi đè dòng đã có.
- Mở `review.csv` bằng Excel hay Google Sheets rồi lưu lại vẫn đọc được (BOM, khoảng trắng thừa, cột đổi thứ tự đều được chấp nhận), nhưng **không đổi tên hay xoá cột**: thiếu hoặc thừa cột thì bước dựng báo lỗi nêu tên cột.
- **Đừng mở CSV bằng cách nhấp đúp vào Excel.** Tên file, tác giả, nhãn lấy từ Wikimedia (ai cũng sửa được), nên một ô bắt đầu bằng `=`, `+`, `-`, `@` sẽ bị Excel chạy như công thức (CSV/formula injection). Hiện có một tên file thật như vậy: `-i---i- (39312478014).jpg`. Hãy dùng trình soạn văn bản (VS Code), hoặc trong Excel chọn *Data → From Text/CSV* và đặt mọi cột là *Text*. Code không thêm dấu `'` vào ô, vì tên file là khoá để khớp `review.csv`, manifest và Commons.
- `key` gõ sai (không khớp file, triple hay thực thể nào trong snapshot) không bị bỏ qua im lặng: nó hiện trong `review_keys_unmatched` và làm `--require-reviewed` thất bại.

**Quy trình nhóm** (tránh xung đột git ở cuối `review.csv`):

1. Chia việc duyệt theo nhóm thực thể cho từng thành viên, để lịch sử commit thể hiện đóng góp (ADR-0015).
2. Mỗi thành viên làm trên nhánh riêng và **chỉ sửa các dòng `pending` của thực thể được giao** (dòng `entity` của chúng, ảnh của chúng, triple có chúng ở đầu). Không thêm, xoá hay sắp xếp lại dòng.
3. Mở PR duyệt; không chạy `scripts/build_mmkg.py` trên nhánh duyệt, vì bước dựng thêm dòng vào cuối file và các nhánh sẽ xung đột.
4. Chỉ dựng lại trên `main` **sau khi** mọi PR duyệt đã merge, rồi commit các bảng mới và dòng `pending` mới (nếu có) trong một PR riêng.
- Sau khi duyệt xong, cập nhật mục "Kết quả duyệt" trong `data_cards/mmkg_vi.md`.

## Tài liệu liên quan

- `docs/decisions/0015-mmkg-tieng-viet-va-data-card.md`: định dạng.
- `docs/decisions/0050-mien-di-tich-di-san-viet-nam-cho-mmkg-thu-thap-tu-wikidata-commons.md`: miền và nguồn.
- `docs/decisions/0051-luu-snapshot-goc-loc-o-buoc-dung-va-duyet-rieng-cho-mmkg.md`: snapshot, luật lọc, duyệt.
- `data_cards/mmkg_vi.md`: data card.
