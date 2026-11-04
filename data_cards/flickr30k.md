# Data card: Flickr30k

**Là gì?** Bộ ảnh người và hoạt động hằng ngày, mỗi ảnh có 5 caption do người viết. Xem lý do chọn ở ADR-0040.

## Nguồn và phiên bản

| Thành phần | Nguồn |
|---|---|
| Caption và split | `dataset_flickr30k.json` trong https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip |
| Ảnh (trên Kaggle) | Bản sao https://www.kaggle.com/datasets/eeshawn/flickr30k, không phải bản nhóm tải trực tiếp từ tác giả |
| Ảnh (chính thức) | Form tại https://shannon.cs.illinois.edu/DenotationGraph/ |

Đã đối chiếu **danh sách file** của bản sao ngày 2026-11-04: 31.783 ảnh, không trùng tên, có đủ 31.014 ảnh trong `dataset_flickr30k.json`. Lần chạy ngày 2026-11-04 mã hoá được toàn bộ, `failed_image_ids` rỗng.

## Giấy phép và điều kiện sử dụng

Trang chính thức ghi: ảnh chỉ dành cho nghiên cứu/giáo dục phi thương mại, phải tuân thủ Flickr Terms of Use, và tác giả không sở hữu bản quyền ảnh.

- **Không đăng lại ảnh.**
- Điều khoản **không nói rõ** về embedding. Vì vậy Kaggle dataset để **riêng tư, chia sẻ cho nhóm** cho tới khi xác nhận được, ví dụ qua điều khoản nhận được khi điền form hoặc hỏi giảng viên.
- Một số bản sao khác trên Kaggle ghi nhãn CC0. Nhãn đó không khớp với điều khoản chính thức, nên nhóm không dựa vào nó.

## Thống kê

Đếm từ `dataset_flickr30k.json` ngày 2026-11-04:

| Mục | Số lượng |
|---|---|
| Ảnh | 31.014 |
| Caption | 155.070, đúng 5 caption mỗi ảnh |
| Split Karpathy | 29.000 train / 1.014 val / 1.000 test |

769 ảnh còn lại của bản đầy đủ không có trong file này nên không được dùng. Lần chạy ngày 2026-11-04 mã hoá đủ 31.014 ảnh và 155.070 caption.

## Split

Dùng split Karpathy, chuyển sang định dạng `splits/flickr30k/` (chưa làm). Loader cố ý không đọc trường `split`, để `splits/` là nguồn duy nhất (ADR-0009).

## Cách liên kết ảnh - thực thể

Dữ liệu không có đồ thị. Dự kiến liên kết từ trong caption sang ConceptNet ở task sau. ConceptNet có giấy phép riêng: ví dụ cạnh trong tài liệu ghi `cc:by-sa/4.0`, cần kiểm tra trước khi đăng đồ thị.

## Embedding

Model `ViT-B-32`, pretrained `laion2b_s34b_b79k`, toàn bộ caption. `image_id` là số trong tên file, ví dụ `1000092795.jpg` → `1000092795`. Tạo trên Kaggle ngày 2026-11-04. Tải từ tab Output của notebook https://www.kaggle.com/code/qphulong/prepare-clip-embeddings-for-mmkg-xai/, cách làm ở §2 của `docs/guides/chuan-bi-embedding-clip.md`. ADR-0041 dự kiến dùng Kaggle dataset riêng tư `clip-features-flickr30k`; hiện nhóm chia sẻ qua Output của notebook.

## Hạn chế và rủi ro

- Ảnh lấy qua bản sao của người khác, nguồn gốc không do nhóm kiểm soát.
- Cần tự dựng liên kết ConceptNet; chất lượng liên kết ảnh hưởng tới điểm đồ thị.
- Có 1 vector ảnh trùng hệt một ảnh khác, nhiều khả năng là ảnh trùng. Khi tạo split, các ảnh này phải nằm cùng một tập để tránh rò rỉ (PDF §6). Có 500 vector caption trùng, nhiều khả năng do câu giống nhau; chưa đối chiếu với văn bản gốc.
