# Data card: Visual Genome

**Là gì?** Bộ ảnh kèm đối tượng, quan hệ và mô tả vùng ảnh (region description). Xem lý do chọn ở ADR-0040.

## Nguồn và phiên bản

| Thành phần | Bản dùng | Nguồn |
|---|---|---|
| Ảnh, `image_data.json`, `region_descriptions.json` | v1.2 | https://homes.cs.washington.edu/~ranjay/visualgenome/api.html |
| Ảnh (trên Kaggle) | Bản sao | https://www.kaggle.com/datasets/lyte69/visualgenome-part1-part2 |

Đã đối chiếu **danh sách file** của bản sao ngày 2026-11-04: có đủ 108.077 ảnh; 170 file rỗng và các ảnh thừa khác không nằm trong danh sách cần dùng. Chưa kiểm tra nội dung ảnh.

Ngày chạy trên Kaggle: 2026-11-04. Người chạy cần ghi rõ ảnh lấy từ bản sao đã gắn hay do notebook tự tải.

## Giấy phép

CC BY 4.0 theo dataset card trên Hugging Face (https://huggingface.co/datasets/visual_genome). Phải ghi nguồn khi dùng hoặc đăng lại embedding. Nên đối chiếu thêm với trang chính thức trước khi đăng.

## Thống kê

Đếm từ file JSON v1.2 ngày 2026-11-04:

| Mục | Số lượng |
|---|---|
| Ảnh | 108.077 (`VG_100K` 64.345, `VG_100K_2` 43.732) |
| Region | 5.408.689, trong đó 7 `phrase` rỗng bị bỏ |
| Region mỗi ảnh | Ít nhất 3, nhiều nhất 267 |
| Văn bản sau khi chọn 5 region mỗi ảnh | 540.383 |

Lần chạy ngày 2026-11-04 mã hoá đủ 108.077 ảnh và 540.383 văn bản, `failed_image_ids` rỗng. Một ảnh chỉ có 3 văn bản, các ảnh còn lại có 5.

## Split

Visual Genome không có split chính thức nên nhóm tự tạo (ADR-0009, ADR-0042): chia theo ảnh, test 5.000, val 5.000, train 98.077; ảnh có embedding gần trùng (cosine ≥ 0,99) luôn nằm cùng một tập. Lần tạo ngày 2026-11-05 có 247 nhóm 2 ảnh, không có nhóm lớn hơn; chạy lại cho ra file giống hệt từng byte. Tạo bằng:

```bash
uv run python scripts/make_splits.py --config configs/splits_visual_genome.yaml
```

Kết quả: `splits/visual_genome/split_seed0.json` (đã commit, đóng băng).

## Cách liên kết ảnh - thực thể

Dùng đối tượng và quan hệ có sẵn trong dữ liệu. Phần dựng đồ thị thuộc task sau.

Bộ truy vấn đánh giá Nhiệm vụ 2 đọc `objects.json` + `relationships.json` (ADR-0044, `docs/query_benchmark.md`).

Đồ thị ảnh – khái niệm (ADR-0045, `docs/guides/xay-do-thi.md`) chỉ đọc `objects.json`:

| Mục | Giá trị |
|---|---|
| Bản | `objects.json.zip` (v1.4) từ https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/objects.json.zip |
| Ngày tải | 2026-11-07 (qphulong), file ghi `Last-Modified: 2023-06-28` |
| sha256 `objects.json.zip` | `efa456069aedb420b7dabf54f4608c89c78a00de550f1433066e9b6f84fb629c` |
| sha256 `objects.json` | `94238b6187c0ea1f97acc1fdaec106bd3dee12bb0f81b23dc8b07e9173514985` |
| Thống kê | 108.077 ảnh (khớp split), 2.516.939 đồ vật, 19,6% không có synset; 1.793 ảnh không có đồ vật; 13.320 ảnh có đồ vật nhưng không đồ vật nào có synset |

Đồ vật thiếu synset được tra synset theo tên bằng từ điển học trên train (ADR-0045). Sau bước này, chỉ còn 1.805 ảnh không có cạnh (1.793 ảnh không có đồ vật nào + 12 ảnh train).

`relationships.json` chưa tải.

## Embedding

Model `ViT-B-32`, pretrained `laion2b_s34b_b79k`. Chọn 5 region mỗi ảnh ngẫu nhiên với `sampling_seed: 0`, chỉ bỏ `phrase` rỗng, và làm trước khi có bất kỳ kết quả truy vấn nào. Tạo trên Kaggle ngày 2026-11-04. Tải từ tab Output của notebook https://www.kaggle.com/code/qphulong/prepare-clip-embeddings-for-mmkg-xai/, cách làm ở §2 của `docs/guides/chuan-bi-embedding-clip.md`. ADR-0041 dự kiến dùng Kaggle dataset công khai `clip-features-visual-genome`; hiện nhóm chia sẻ qua Output của notebook.

## Hạn chế và rủi ro

- Region chỉ mô tả một phần ảnh, thường ngắn và chung chung, nên nhiều ảnh có thể khớp cùng một câu. Điều này ảnh hưởng tới cách diễn giải Recall@K.
- Ảnh Visual Genome lấy một phần từ MS-COCO.
- Có 5 vector ảnh trùng hệt một ảnh khác, nhiều khả năng là ảnh trùng. Khi tạo split, các ảnh này phải nằm cùng một tập để tránh rò rỉ (PDF §6).
- 140.886 vector văn bản (26%) trùng một vector khác, nhiều khả năng do cùng một câu ngắn xuất hiện ở nhiều ảnh; chưa đối chiếu với văn bản gốc. Cần quyết định cách xử lý trước khi dùng làm truy vấn hoặc cặp dương khi huấn luyện.
