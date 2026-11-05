# data/ — dữ liệu thô (KHÔNG commit)

**Là gì?** Thư mục chứa dữ liệu tải về. Toàn bộ nội dung bị `.gitignore` bỏ qua (`/data/*`), chỉ riêng file này được commit. Lý do: dữ liệu rất lớn (Visual Genome ~ 15 GB ảnh), git không phù hợp. Xem `docs/decisions/0008-luu-tru-du-lieu-va-artifact.md`.

## Cấu trúc mong đợi

| Đường dẫn | Nội dung |
|---|---|
| `data/visual_genome/` | `images/`, `objects.json`, `relationships.json`, `image_data.json`, `region_descriptions.json` |
| `data/flickr30k/` | `images/`, `dataset_flickr30k.json` |
| `data/fb15k237_img/` | Triple train/valid/test + thư mục ảnh thực thể |
| `data/mkg_w/`, `data/mkg_y/` | Triple + ảnh của MKG-W / MKG-Y |
| `data/conceptnet/` | File assertions ConceptNet (đã lọc tiếng Anh/Việt) |

Đặc trưng CLIP, chỉ mục FAISS, checkpoint GNN **không** để ở đây mà để ở `artifacts/` (cũng gitignore).

## Nguồn tải (tự kiểm chứng phiên bản khi tải)

| Bộ dữ liệu | Nguồn | Ghi chú |
|---|---|---|
| Visual Genome | Trang chính thức Visual Genome (homes.cs.washington.edu/~ranjay/visualgenome) | Ảnh, metadata, region description: v1.2; đối tượng và quan hệ: v1.4 hoặc v1.2, ghi rõ vào data card |
| Flickr30k | Form tại shannon.cs.illinois.edu/DenotationGraph; caption và split từ `caption_datasets.zip` của Karpathy | Chỉ dùng cho nghiên cứu/giáo dục phi thương mại. Xem `docs/guides/chuan-bi-embedding-clip.md` |
| FB15k-237-IMG | Kho GitHub `zjunlp/MKGformer` | Theo hướng dẫn tải trong README của kho |
| MKG-W / MKG-Y | Kho GitHub của NativE / AdaMF-MAT (`zjukg`) | Kiểm tra giấy phép |
| ConceptNet | conceptnet.io (bản assertions CSV) | Rất lớn, nên lọc trước |

## Quy tắc

1. Không sửa file gốc trong `data/`. Mọi tiền xử lý ghi ra `artifacts/`.
2. Ghi ngày tải + phiên bản vào data card tương ứng trong `data_cards/`.
3. Khi nộp bài: dữ liệu và artifact lớn đưa lên cloud (Google Drive), dán link vào `README.md` (theo §9 của đề).
