# Hướng dẫn: tải dữ liệu và tạo lại embedding CLIP

> ⚠️ **Chưa chạy thật.** Code và notebook đã có, nhưng CLIP thật mới chạy lần đầu khi bạn chạy notebook trên Kaggle. Link Kaggle ở §2 và thời gian ở §5 sẽ được cập nhật sau lần chạy đó. Thiết kế: [`docs/superpowers/specs/2026-11-04-clip-embeddings-design.md`](../superpowers/specs/2026-11-04-clip-embeddings-design.md).

## 1. Bạn cần làm gì?

**Embedding** là một dãy 512 số biểu diễn một ảnh hoặc một câu. Ảnh và câu có nghĩa gần nhau thì có embedding gần nhau.

| Bạn muốn | Làm theo |
|---|---|
| Dùng embedding có sẵn (đa số thành viên) | §2. Không cần GPU, không cần tải ảnh |
| Tạo lại embedding trên máy | §3 → §4 → §6 |
| Tạo lại và đăng lên Kaggle | §3 → §5 → §6 |

## 2. Dùng embedding có sẵn

Link Kaggle sẽ được điền sau khi đăng: `TODO`.

Đặt file tải về như sau:

```text
artifacts/features/
├── visual_genome/ViT-B-32__laion2b_s34b_b79k/
│   ├── image_features.npz
│   ├── text_features.npz
│   └── meta.json
└── flickr30k/ViT-B-32__laion2b_s34b_b79k/
    ├── image_features.npz
    ├── text_features.npz
    └── meta.json
```

`artifacts/` đã có trong `.gitignore`, nên các file này không bị commit nhầm.

Cách đọc (chỉ cần numpy):

```python
import numpy as np

folder = "artifacts/features/visual_genome/ViT-B-32__laion2b_s34b_b79k"
images = np.load(f"{folder}/image_features.npz")
image_embeddings = images["embeddings"]  # shape (số ảnh, 512)
image_ids = images["image_ids"]  # hàng i của embeddings là ảnh image_ids[i]

texts = np.load(f"{folder}/text_features.npz")
text_embeddings = texts["embeddings"]  # shape (số văn bản, 512)
text_image_ids = texts["image_ids"]  # văn bản i thuộc về ảnh text_image_ids[i]
```

Embedding đã chuẩn hoá L2, nên tích vô hướng chính là cosine:

```python
# Filter to the split you are allowed to use first (see the note below).
scores = image_embeddings @ text_embeddings[0]
```

**Lưu ý chống rò rỉ:** Kaggle không chứa split. Luôn lọc theo file trong `splits/` trước khi huấn luyện, và không dùng
ảnh hay văn bản test để huấn luyện hoặc chọn tham số (PDF §6).

## 3. Tải dữ liệu thô và đặt đúng chỗ

Mọi đường dẫn tính từ thư mục gốc của repo. Không sửa file gốc trong `data/`.

### 3.1. Cấu trúc cần có

```text
data/
├── visual_genome/
│   ├── images/
│   │   ├── VG_100K/*.jpg
│   │   └── VG_100K_2/*.jpg
│   ├── image_data.json
│   └── region_descriptions.json
└── flickr30k/
    ├── images/*.jpg
    └── dataset_flickr30k.json
```

### 3.2. Visual Genome (khoảng 15 GB ảnh)

Giấy phép theo dataset card: CC BY 4.0. Các URL dưới đây đã kiểm tra ngày 2026-11-04.

```bash
mkdir -p data/visual_genome/images
cd data/visual_genome

# Images v1.2. -C - lets an interrupted download resume.
curl -L -C - -O https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip    # ~9.7 GB
curl -L -C - -O https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip   # ~5.5 GB
unzip -q images.zip -d images
unzip -q images2.zip -d images

# Annotations v1.2
curl -L -O https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip
curl -L -O https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/region_descriptions.json.zip
unzip -q image_data.json.zip
unzip -q region_descriptions.json.zip

cd ../..
```

Sau khi giải nén, kiểm tra số ảnh. Kỳ vọng khoảng 108.077 (theo dataset card):

```bash
find data/visual_genome/images -name '*.jpg' | wc -l
```

Nếu thư mục sau khi giải nén khác `images/VG_100K/` và `images/VG_100K_2/`, hãy chuyển về đúng cấu trúc ở §3.1.

### 3.3. Flickr30k

Trang chính thức yêu cầu **điền form** và chỉ cho phép dùng vào nghiên cứu/giáo dục phi thương mại. Có thể phải chờ.
**Không đăng lại ảnh ở bất kỳ đâu.**

1. Điền form tại trang [Denotation Graph](https://shannon.cs.illinois.edu/DenotationGraph/).
2. Đặt các file `.jpg` vào `data/flickr30k/images/`.
3. Tải caption và split của Karpathy:

```bash
mkdir -p data/flickr30k/images
curl -L -o data/flickr30k/caption_datasets.zip \
  https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
unzip -j data/flickr30k/caption_datasets.zip dataset_flickr30k.json -d data/flickr30k
```

Kiểm tra số ảnh. Bản đầy đủ có khoảng 31.783 ảnh, nhưng `dataset_flickr30k.json` chỉ liệt kê **31.014 ảnh** (155.070 caption, chia
29.000 / 1.014 / 1.000). Code chỉ dùng 31.014 ảnh này và bỏ qua phần thừa, nên trong thư mục có nhiều ảnh hơn là bình thường:

```bash
find data/flickr30k/images -name '*.jpg' | wc -l
```

**Trên Kaggle** thì không cần form để lấy ảnh: gắn bản sao [`eeshawn/flickr30k`](https://www.kaggle.com/datasets/eeshawn/flickr30k), thư mục
`flickr30k_images/`. Điều khoản của Flickr30k vẫn áp dụng, và phải ghi nguồn này vào `data_cards/flickr30k.md`. Vẫn dùng
`dataset_flickr30k.json` để lấy caption và split. Các bản khác như `hsankesara/flickr-image-dataset` ghi nhãn CC0, nhưng nhãn đó
không khớp với điều khoản chính thức, nên đừng dựa vào nó.

### 3.4. Ghi lại

Ghi ngày tải, nguồn và phiên bản vào `data_cards/visual_genome.md` và `data_cards/flickr30k.md`.

## 4. Tạo lại embedding trên máy

CLIP đóng băng, không huấn luyện. Máy không có GPU mạnh vẫn chạy được nhưng rất chậm, nên với toàn bộ dữ liệu hãy dùng
Kaggle (§5).

```bash
# Install uv first if needed: https://docs.astral.sh/uv/getting-started/installation/
uv sync --group ml
```

**Bước 1: chạy thử.** Nhanh, phát hiện sớm lỗi đường dẫn và cho biết tốc độ thật. Kết quả ghi vào
`artifacts/features/<dataset>_partial/` và không được đăng.

```bash
uv run python scripts/extract_features.py --config configs/features_flickr30k.yaml \
  --limit-images 1000 --limit-texts 10000
```

**Bước 2: chạy đầy đủ.**

```bash
uv run python scripts/extract_features.py --config configs/features_visual_genome.yaml
uv run python scripts/extract_features.py --config configs/features_flickr30k.yaml
```

Kết quả nằm trong `artifacts/features/`, cấu trúc như §2.

## 5. Tạo lại embedding trên Kaggle

Chỉ có một notebook: `notebooks/10_prepare_clip_embeddings.ipynb`. Code nằm ngay trong notebook nên Kaggle không cần quyền truy cập repo. Thứ tự: config, code, smoke, dataset 1, dataset 2. Bạn chỉ đổi một dòng ở ô đầu tiên:

```python
DATASET = "flickr30k"  # or "visual_genome"
```

Mỗi lần chạy xử lý một bộ dữ liệu. Muốn làm song song thì mở 2 phiên, mỗi phiên một giá trị:

| `DATASET` | Bộ dữ liệu | Thời gian ước tính (chưa đo) |
|---|---|---|
| `"visual_genome"` | Visual Genome | 20–40 phút nếu gắn sẵn ảnh; tới ~75 phút nếu phải tải |
| `"flickr30k"` | Flickr30k | 10–25 phút |

1. Tải notebook lên Kaggle, bật GPU và Internet.
2. Gắn ảnh bằng **Add Input**: `eeshawn/flickr30k` cho Flickr30k (bắt buộc), `lyte69/visualgenome-part1-part2` cho Visual Genome (nên có, nếu không notebook tự tải khoảng 15 GB, và chưa rõ dung lượng thư mục tạm có đủ không). Không cần làm theo §3 trên Kaggle: notebook tự tải file JSON.
3. **Đổi `DATASET`.** Không đổi code giữa hai lần chạy, nếu không hai bộ embedding có thể không so sánh được.
4. Chọn **Save Version → Save & Run All**. Chỉ cách này mới giữ lại output: file trong phiên làm việc thường sẽ mất khi phiên kết thúc. Notebook chạy smoke trước, lỗi thì dừng; đạt thì chạy đầy đủ rồi tự kiểm tra kết quả.
5. Mở phần Output của phiên bản vừa chạy và kiểm tra `meta.json` (§6). Nếu kiểm tra thất bại, thư mục kết quả sẽ không được tạo.
6. Tạo Kaggle dataset từ output đó:
   - `clip-features-visual-genome`: công khai, ghi nguồn và CC BY 4.0.
   - `clip-features-flickr30k`: **riêng tư, chia sẻ cho nhóm**, cho tới khi xác nhận được điều khoản cho phép đăng công khai.
7. Dán link vào §2 của file này.

Không lưu token Kaggle trong repo hoặc trong nội dung notebook.

**Khi sửa code:** sửa trong `src/mmxai/` trước, rồi chạy `uv run python scripts/sync_notebook_code.py` để cập nhật notebook. `tests/test_notebooks_match_src.py` sẽ báo lỗi nếu bản sao bị lệch. Khi gỡ lỗi trên Kaggle, sau khi sửa một ô code hãy **Restart Kernel** rồi chạy lại, để code đang chạy khớp với `code_hash`. Sửa module sẽ đổi `code_hash`, nên phải chạy lại cả hai bộ dữ liệu trước khi đăng.

## 6. Kiểm tra kết quả

Script tự kiểm tra sau khi chạy. Có lỗi thì script báo thất bại.

| Kiểm tra | Vì sao |
|---|---|
| Số hàng của mọi mảng khớp nhau và khớp `meta.json` | Lệch hàng thì mọi kết quả về sau đều sai |
| Không có NaN/Inf, độ dài vector xấp xỉ 1 | Cosine chỉ đúng khi đã chuẩn hoá |
| Id không trùng, mỗi văn bản trỏ tới một ảnh có thật | Đáp án của truy vấn dựa vào liên kết này |
| Mỗi ảnh Visual Genome có tối đa 5 văn bản, Flickr30k đúng 5 | Đúng quy tắc đã chốt |
| Cặp văn bản-ảnh đúng có cosine trung bình cao hơn cặp xáo trộn | Phát hiện lệch hàng |

Mở `meta.json` và xác nhận: `is_partial` là `false`, `code_hash` và `open_clip_version` giống nhau ở hai bộ dữ liệu, và
`failed_image_ids` rỗng hoặc rất ít.

## 7. Lỗi thường gặp

| Hiện tượng | Xử lý |
|---|---|
| Không tìm thấy file | So thư mục với §3.1, kiểm tra `data_dir` trong config hoặc `--data-dir` |
| Script dừng vì quá nhiều ảnh lỗi (trên 0,1%) | Nhiều khả năng tải hoặc giải nén chưa xong. Đếm lại số ảnh theo §3 |
| Hết bộ nhớ GPU | Giảm `image_batch_size` hoặc `text_batch_size` trong config |
| Chạy rất chậm dù có GPU | Kiểm tra `num_workers`. Đọc và thay đổi kích thước ảnh diễn ra trên CPU |
| `meta.json` không khớp config | Không dùng cache đó. Tạo lại hoặc sửa đúng config |
| Hết dung lượng trên Kaggle | Gắn sẵn ảnh bằng Add Input thay vì để notebook tải |
| Báo "text and image rows look misaligned" | Đây là lỗi lệch hàng, chỉ sửa lỗi đó. **Không** đổi model, seed hay số region để check này đạt, vì như vậy là chọn tham số theo kết quả (PDF §6) |
| In ra `device: cpu` | Chưa bật GPU. Dừng lại, bật GPU rồi chạy lại |

## 8. Câu hỏi vấn đáp thường gặp

| Câu hỏi | Trả lời ngắn |
|---|---|
| Vì sao trích đặc trưng cả ảnh test mà không rò rỉ? | CLIP đóng băng, không học gì. Rò rỉ chỉ xảy ra nếu ảnh test được dùng để huấn luyện, chọn tham số hoặc xây cạnh thống kê |
| Vì sao chỉ lấy 5 region mỗi ảnh? | Toàn bộ 5,4 triệu region tốn khoảng 11 GB. Số 5 tương ứng với 5 caption của Flickr30k |
| Sao biết 5 region không được chọn để có lợi cho mô hình? | Chọn ngẫu nhiên với seed cố định, không lọc theo nội dung, trước khi có kết quả truy vấn |
| Biết embedding do mô hình nào tạo bằng cách nào? | Mở `meta.json` |
| Vì sao dùng `.npz` thay vì `.npy` + JSON? | Embedding và id nằm trong một file nên không thể lệch nhau. Đổi lại không dùng được `mmap_mode` |
