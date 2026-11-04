# Data card: Flickr30k

**Là gì?** Bộ ảnh người và hoạt động hằng ngày, mỗi ảnh có 5 caption do người viết. Xem lý do chọn ở ADR-0040.

## Nguồn và phiên bản

| Thành phần | Nguồn |
|---|---|
| Caption và split | `dataset_flickr30k.json` trong https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip |
| Ảnh (trên Kaggle) | Bản sao https://www.kaggle.com/datasets/eeshawn/flickr30k, không phải bản nhóm tải trực tiếp từ tác giả |
| Ảnh (chính thức) | Form tại https://shannon.cs.illinois.edu/DenotationGraph/ |

Đã đối chiếu **danh sách file** của bản sao ngày 2026-11-04: 31.783 ảnh, không trùng tên, có đủ 31.014 ảnh trong `dataset_flickr30k.json`. Chưa kiểm tra nội dung ảnh; lần chạy thật sẽ báo ảnh hỏng trong `failed_image_ids`.

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

769 ảnh còn lại của bản đầy đủ không có trong file này nên không được dùng. Số ảnh mã hoá được thật sẽ ghi sau lần chạy trên Kaggle.

## Split

Dùng split Karpathy, chuyển sang định dạng `splits/flickr30k/` (chưa làm). Loader cố ý không đọc trường `split`, để `splits/` là nguồn duy nhất (ADR-0009).

## Cách liên kết ảnh - thực thể

Dữ liệu không có đồ thị. Dự kiến liên kết từ trong caption sang ConceptNet ở task sau. ConceptNet có giấy phép riêng: ví dụ cạnh trong tài liệu ghi `cc:by-sa/4.0`, cần kiểm tra trước khi đăng đồ thị.

## Embedding

Model `ViT-B-32`, pretrained `laion2b_s34b_b79k`, toàn bộ caption. `image_id` là số trong tên file, ví dụ `1000092795.jpg` → `1000092795`. Kaggle dataset dự kiến: `clip-features-flickr30k`, riêng tư. Link: chưa đăng.

## Hạn chế và rủi ro

- Ảnh lấy qua bản sao của người khác, nguồn gốc không do nhóm kiểm soát.
- Cần tự dựng liên kết ConceptNet; chất lượng liên kết ảnh hưởng tới điểm đồ thị.
