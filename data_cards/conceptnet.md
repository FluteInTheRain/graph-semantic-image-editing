# Data card: ConceptNet 5.7

**Là gì?** Đồ thị tri thức thường thức công khai: mỗi cạnh là (đầu, quan hệ, cuối, weight), ví dụ `racket -UsedFor-> tennis`. Đồ án dùng nó làm tri thức ngoài để viết truy vấn gián tiếp mô tả một khái niệm ẩn (ADR-0057).

## Nguồn và phiên bản

| Thành phần | Bản dùng | Nguồn |
|---|---|---|
| `conceptnet-assertions-5.7.0.csv.gz` | 5.7.0 | https://github.com/commonsense/conceptnet5/wiki/Downloads |

| Mục cần người tải điền | Giá trị |
|---|---|
| Ngày tải | 2026-11-15, từ https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz |
| Người tải | _(chưa điền)_ |
| Dung lượng file tải về | 497963447 byte |
| sha256 file tải về | `ACCD65FE94038584295574DDC26E1500C1919C8C4532BF771811CAFD0948AF7E` |

## Giấy phép

CC BY-SA 4.0 theo trang ConceptNet; một số nguồn con có điều khoản riêng, cần đối chiếu trang gốc trước khi đăng lại. Không commit file dữ liệu (`data/` và `artifacts/` bị gitignore).

## Cách dùng trong đồ án

Lọc một lần bằng `scripts/prepare_conceptnet.py` (config `configs/conceptnet_en.yaml`):

```bash
uv run python scripts/prepare_conceptnet.py --config configs/conceptnet_en.yaml
```

| Mục | Giá trị |
|---|---|
| Ngôn ngữ | Chỉ giữ cạnh có cả hai đầu là tiếng Anh (`/c/en/...`) |
| Quan hệ giữ lại | `IsA`, `UsedFor`, `AtLocation`, `HasA`, `PartOf`, `CapableOf` |
| Ngưỡng weight | `weight >= 1.0` |
| Output | `artifacts/conceptnet/en_edges.tsv` (cột `uri`, `rel`, `start`, `end`, `weight`) |

Mỗi cạnh giữ nguyên URI gốc để từng truy vấn trích dẫn được cạnh nó dựa vào.

| Mục cần người chạy điền | Giá trị |
|---|---|
| Số dòng đọc | 34074917 (`malformed_lines` = 0) |
| Số cạnh giữ | 302709 |
| sha256 của TSV | `20ecee542cd3955078a60c7719c13a2b93e5417c8e4047a5fc63659949fe589e` |
| Cạnh theo quan hệ | IsA 198018, UsedFor 39756, AtLocation 25652, CapableOf 22672, PartOf 11066, HasA 5545 |

## Rò rỉ dữ liệu

ConceptNet là tri thức ngoài, không xây từ ảnh hay truy vấn của Visual Genome nên không phụ thuộc split (PDF §6).

## Hạn chế và rủi ro

- ConceptNet có cạnh nhiễu hoặc chủ quan; weight < 1 đã bị loại nhưng chưa đảm bảo đúng hoàn toàn.
- Cạnh bị lọc theo chuỗi từ nên các nghĩa khác nhau của một từ có thể bị gộp.
