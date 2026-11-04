# Data card: Visual Genome

**Là gì?** Bộ ảnh kèm đối tượng, quan hệ và mô tả vùng ảnh (region description). Xem lý do chọn ở ADR-0040.

## Nguồn và phiên bản

| Thành phần | Bản dùng | Nguồn |
|---|---|---|
| Ảnh, `image_data.json`, `region_descriptions.json` | v1.2 | https://homes.cs.washington.edu/~ranjay/visualgenome/api.html |
| Ảnh (trên Kaggle) | Bản sao | https://www.kaggle.com/datasets/lyte69/visualgenome-part1-part2 |

Đã đối chiếu **danh sách file** của bản sao ngày 2026-11-04: có đủ 108.077 ảnh; 170 file rỗng và các ảnh thừa khác không nằm trong danh sách cần dùng. Chưa kiểm tra nội dung ảnh.

Ngày tải: chưa tải đầy đủ. Hai file JSON được tải để đếm ngày 2026-11-04. Ghi ngày tải ảnh sau lần chạy thật.

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

Số ảnh mã hoá được thật, và số ảnh lỗi nếu có, sẽ ghi sau lần chạy trên Kaggle.

## Split

Chưa có. Visual Genome không có split chính thức nên phải tạo bằng `scripts/make_splits.py` (chưa viết) trước khi huấn luyện (ADR-0009).

## Cách liên kết ảnh - thực thể

Dùng đối tượng và quan hệ có sẵn trong dữ liệu. Phần dựng đồ thị thuộc task sau.

## Embedding

Model `ViT-B-32`, pretrained `laion2b_s34b_b79k`. Chọn 5 region mỗi ảnh ngẫu nhiên với `sampling_seed: 0`, chỉ bỏ `phrase` rỗng, và làm trước khi có bất kỳ kết quả truy vấn nào. Kaggle dataset dự kiến: `clip-features-visual-genome`, công khai. Link: chưa đăng.

## Hạn chế và rủi ro

- Region chỉ mô tả một phần ảnh, thường ngắn và chung chung, nên nhiều ảnh có thể khớp cùng một câu. Điều này ảnh hưởng tới cách diễn giải Recall@K.
- Ảnh Visual Genome lấy một phần từ MS-COCO.
