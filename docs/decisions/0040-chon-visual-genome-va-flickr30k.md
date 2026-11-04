# ADR-0040: Dùng Visual Genome và Flickr30k + ConceptNet làm hai tập công khai

- **Trạng thái:** Đề xuất
- **Ngày:** 2026-11-04

## Bối cảnh

Cấp độ 2 yêu cầu thực nghiệm trên **ít nhất 2 tập công khai** (PDF §5). Đề gợi ý nhiều bộ dữ liệu (PDF §10), nhưng bài toán chính là **tìm ảnh bằng văn bản**, và pipeline cần trích đặc trưng từ **ảnh gốc** bằng CLIP đóng băng (PDF §4.1). Vì vậy một bộ dữ liệu phù hợp cần có ảnh gốc tải được, văn bản làm truy vấn, và một đường rõ ràng tới đồ thị.

## Quyết định

| Bộ dữ liệu | Vai trò | Nguồn đồ thị | Bản dùng |
|---|---|---|---|
| Visual Genome | Có sẵn scene graph, dễ giải thích bằng quan hệ | Đối tượng và quan hệ trong dữ liệu | Ảnh, metadata, region description v1.2 |
| Flickr30k | Benchmark văn bản - ảnh với caption thật | Liên kết từ trong caption sang ConceptNet (task sau) | Caption và split Karpathy; 31.014 ảnh |

Số đếm từ file gốc ngày 2026-11-04: Visual Genome có 108.077 ảnh và 5.408.689 region; `dataset_flickr30k.json` có 31.014 ảnh, 155.070 caption, chia 29.000 / 1.014 / 1.000.

Trên Kaggle, ảnh Flickr30k lấy từ bản sao `eeshawn/flickr30k`. Danh sách file của bản này có đủ 31.014 ảnh, không trùng tên, và có nêu điều khoản gốc.

## Các phương án đã cân nhắc

| Phương án | Ưu | Nhược | Lý do không chọn |
|---|---|---|---|
| Visual Genome + Flickr30k | Có ảnh gốc, có văn bản thật, hợp truy vấn gián tiếp | Flickr30k phải tự dựng liên kết ConceptNet | **Được chọn** |
| Visual Genome + FB15k-237-IMG | Đề nêu tên; hợp phương pháp hoàn thiện KG | Ảnh là danh sách URL từ `mmkb`, giấy phép không rõ, MKGformer cảnh báo lỗi tiền xử lý | Khó chuẩn bị và khó chia sẻ |
| Visual Genome + MKG-W/MKG-Y | Dùng được ngay với NativE, AdaMF-MAT | README chỉ nhắc embedding có sẵn, không nhắc ảnh gốc | Có thể không trích được CLIP |
| Visual Genome + MS-COCO Captions | Caption tốt, tải dễ | Ảnh Visual Genome lấy một phần từ COCO, hai tập không độc lập | Kiểm chứng kém hơn |

## Hệ quả

- **Phương pháp cấp độ 2:** NativE và AdaMF-MAT là phương pháp hoàn thiện KG nên không chạy thẳng trên Flickr30k. Nhóm cần chọn phương pháp graph-enhanced retrieval, hoặc xem lại ADR này.
- **Giấy phép:** ảnh Flickr30k chỉ dùng cho nghiên cứu/giáo dục phi thương mại và không được đăng lại. Xem ADR-0041.
- **Rò rỉ dữ liệu:** Visual Genome chưa có split chính thức, phải tạo bằng `scripts/make_splits.py` trước khi huấn luyện. Flickr30k dùng lại split Karpathy.
- **Việc tiếp theo:** viết data card, tạo split, viết phần liên kết ConceptNet.

## Nguồn tham khảo

- Đề bài, §4.1, §5, §9, §10.
- Visual Genome: https://homes.cs.washington.edu/~ranjay/visualgenome/api.html
- Dataset card Visual Genome: https://huggingface.co/datasets/visual_genome
- Flickr30k: https://shannon.cs.illinois.edu/DenotationGraph/
- Bản sao trên Kaggle: https://www.kaggle.com/datasets/eeshawn/flickr30k
- NativE: https://github.com/zjukg/NATIVE
- AdaMF-MAT: https://github.com/zjukg/AdaMF-MAT
- MKGformer: https://github.com/zjunlp/MKGformer
