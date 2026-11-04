# Thiết kế: chuẩn bị embedding CLIP cho Visual Genome và Flickr30k

- **Ngày:** 2026-11-04
- **Trạng thái:** Đã duyệt ngày 2026-11-04. Kế hoạch: [`docs/superpowers/plans/2026-11-04-clip-embeddings.md`](../plans/2026-11-04-clip-embeddings.md)
- **Hướng dẫn chạy lại:** [`docs/guides/chuan-bi-embedding-clip.md`](../../guides/chuan-bi-embedding-clip.md)

## 1. Mục đích

**Embedding** là một dãy số biểu diễn một ảnh hoặc một câu. CLIP đưa ảnh và câu vào cùng một không gian,
nên có thể so sánh trực tiếp hai loại này.

Mục tiêu: trích embedding **một lần** bằng CLIP đóng băng cho hai bộ dữ liệu công khai (PDF §4.1), rồi
đăng lên Kaggle. Nhờ vậy cả nhóm dùng **đúng một bộ đặc trưng**, không ai phải có GPU hay tải 15 GB ảnh chỉ
để bắt đầu. Hai bộ dữ liệu cũng đáp ứng yêu cầu "ít nhất 2 tập công khai" của cấp độ 2 (PDF §5).

**Thành công nghĩa là:** một thành viên tải dữ liệu từ Kaggle, đọc được các file `.npz`, kiểm tra được
`meta.json`, và dựng được FAISS index mà không cần ảnh gốc hay GPU.

## 2. Phạm vi

| Trong phạm vi | Ngoài phạm vi (làm ở task khác) |
|---|---|
| Đọc danh sách ảnh và văn bản của 2 bộ dữ liệu | Tạo file split (`scripts/make_splits.py`) |
| Trích embedding ảnh và văn bản | FAISS index, metric, baseline |
| Lưu `.npz` + `meta.json`, kiểm tra kết quả | Dựng đồ thị, liên kết ConceptNet |
| 1 notebook Kaggle, đăng 2 Kaggle dataset | Bộ ≥ 20 truy vấn trong `queries/` |
| Data card, hướng dẫn chạy lại, ADR | Đăng ảnh gốc lên Kaggle |

**Không được xảy ra:**

- Huấn luyện hay fine-tune CLIP (PDF §4.1).
- Chọn tham số dựa trên điểm truy vấn (PDF §6).
- Dữ liệu, đặc trưng hoặc token Kaggle bị commit vào git.
- Hai bộ đặc trưng tạo bởi các phiên bản code/mô hình khác nhau mà không ai biết.

## 3. Quyết định đã chốt

| Quyết định | Nội dung | Người quyết định |
|---|---|---|
| Bộ dữ liệu | Visual Genome và Flickr30k (đồ thị của Flickr30k lấy từ ConceptNet ở task sau) | Người dùng |
| Văn bản của Visual Genome | **5 region description mỗi ảnh**, chọn ngẫu nhiên có seed cố định | Người dùng |
| Định dạng | `image_features.npz` + `text_features.npz` + `meta.json` | Người dùng |
| Cách chạy | **1 notebook Kaggle gồm config, code, smoke, dataset 1, dataset 2.** Chỉ đổi một dòng `DATASET` rồi Run All. Có thể chạy 2 phiên song song, mỗi phiên một giá trị | Người dùng |
| Nơi chia sẻ | Kaggle, công khai (xem điều kiện ở §8) | Người dùng |
| Vị trí code | Theo ADR-0003, không tạo thư mục `prepare_dataset_with_clip/` | Claude (người dùng giao) |
| 2 Kaggle dataset riêng | Sửa một bộ không ảnh hưởng bộ kia | Claude |
| Code trên Kaggle | Không cho Kaggle truy cập repo. `src/mmxai/` vẫn là bản gốc có test; notebook chứa bản sao của code | Người dùng |
| Không nén `.npz` | Dùng `np.savez`, vì embedding gần như ngẫu nhiên nên nén được ít | Claude |
| Phạm vi Flickr30k | Chỉ các ảnh có trong `dataset_flickr30k.json` (31.014 ảnh); ảnh thừa trong thư mục bị bỏ qua | Claude |
| Id ảnh Flickr30k | Số trong tên file (`1000092795.jpg` → `1000092795`), không dùng `imgid` của Karpathy (0–31013), vì tên file khớp được với các nguồn Flickr30k khác | Claude |

## 4. Đầu vào và đầu ra

Số đã đếm trực tiếp từ file chú thích gốc ngày 2026-11-04. Số ảnh đọc được có thể ít hơn nếu có ảnh lỗi; số cuối cùng ghi vào data card.

| Bộ dữ liệu | Đầu vào | Mảng embedding | Dung lượng |
|---|---|---|---|
| Visual Genome | 108.077 ảnh | `(108077, 512)` | ~220 MB |
| Visual Genome | ~540.383 region description | Khoảng `(540383, 512)`, có 1 ảnh chỉ có 3 region | ~1,1 GB |
| Flickr30k | 31.014 ảnh | `(31014, 512)` | ~64 MB |
| Flickr30k | 155.070 caption | `(155070, 512)` | ~318 MB |

### Cấu trúc đầu ra

```text
<output_root>/<dataset>/<model>__<pretrained>/
├── image_features.npz
│     embeddings   float32, (N_images, 512), đã chuẩn hoá L2
│     image_ids    int64,   (N_images,), tăng dần
├── text_features.npz
│     embeddings   float32, (N_texts, 512), đã chuẩn hoá L2
│     image_ids    int64,   (N_texts,), ảnh mà văn bản thuộc về
│     text_ids       int64, (N_texts,), chỉ Visual Genome, là region_id
│     caption_index  int64, (N_texts,), chỉ Flickr30k, từ 0 đến 4
└── meta.json
```

- Hàng `i` của mọi mảng trong cùng một file nói về cùng một đối tượng.
- Thứ tự hàng trong `text_features.npz`: tăng dần theo `image_ids`, rồi theo `text_ids`/`caption_index`.
- Không lưu mảng kiểu object, để đọc được với `allow_pickle=False`.
- File split lưu id dạng chuỗi (`splits/README.md`), nên khi so khớp phải đổi bằng `str(image_id)`.
- Lưu ý: mở `.npz` sẽ nạp cả mảng vào RAM, không dùng được `mmap_mode`. Mảng lớn nhất ~1,1 GB nên chấp nhận được.

### `meta.json`

```json
{
  "dataset": "visual_genome",
  "dataset_version": "1.2",
  "model": "ViT-B-32",
  "pretrained": "laion2b_s34b_b79k",
  "embedding_dim": 512,
  "normalized": true,
  "num_images": 108077,
  "num_texts": 540383,
  "text_source": "region_descriptions",
  "regions_per_image": 5,
  "sampling_seed": 0,
  "failed_image_ids": [],
  "is_partial": false,
  "open_clip_version": "ghi lúc chạy",
  "torch_version": "ghi lúc chạy",
  "code_hash": "sha256 của các module đã chạy, ghi lúc chạy",
  "created_at": "ghi lúc chạy"
}
```

## 5. Cấu trúc code

Theo ADR-0003: logic nằm trong `src/mmxai/`, script chỉ gọi hàm. Notebook chứa bản sao code, có test chống lệch (§9).

```text
src/mmxai/
├── utils/config.py              # đọc YAML vào dataclass, khoá lạ thì báo lỗi (ADR-0007)
├── data/
│   ├── records.py               # ImageRecord, TextRecord dùng chung
│   ├── visual_genome.py         # load_image_records, load_region_texts, sample_regions
│   └── flickr30k.py             # load_image_records, load_captions
└── features/
    ├── clip_encoder.py          # load_frozen_clip, encode_images, encode_texts
    ├── cache.py                 # save_features, load_features, check_meta_matches_config
    ├── verify.py                # verify_features: kiểm tra kết quả sau khi chạy
    └── pipeline.py              # run_extraction: nối các bước, không import torch ở đầu file

scripts/extract_features.py      # đọc tham số, gọi các hàm trên
notebooks/
└── 10_prepare_clip_embeddings.ipynb
configs/
├── features_visual_genome.yaml
└── features_flickr30k.yaml
data_cards/
├── visual_genome.md
└── flickr30k.md
tests/
├── test_config.py
├── test_data_visual_genome.py
├── test_data_flickr30k.py
├── test_feature_cache.py        # chỉ cần numpy
├── test_feature_verify.py       # chỉ cần numpy
├── test_clip_encoder.py         # bỏ qua nếu chưa cài torch
└── test_notebooks_match_src.py  # bản sao trong notebook phải giống hệt src/mmxai/
```

**Vì sao tách như vậy:**

- Loader chỉ trả về đường dẫn ảnh, id và văn bản, nên encoder không cần biết định dạng từng bộ dữ liệu.
- `cache.py` và `verify.py` không import torch, nên test chạy ngay sau `uv sync` (ADR-0004).
- Laptop và Kaggle chạy cùng một code, chỉ khác config và đường dẫn.

### Config

Hai file chỉ khác nhau ở phần dữ liệu. Ví dụ Visual Genome:

```yaml
dataset: visual_genome
dataset_version: "1.2"
data_dir: data/visual_genome
output_root: artifacts/features

model: ViT-B-32
pretrained: laion2b_s34b_b79k
device: auto            # cuda nếu có, ngược lại cpu
image_batch_size: 256
text_batch_size: 1024
num_workers: 4
max_failed_image_fraction: 0.001

regions_per_image: 5
sampling_seed: 0
```

Với Flickr30k, bỏ `regions_per_image` và `sampling_seed`.

### Tham số dòng lệnh

| Tham số | Bắt buộc | Ý nghĩa |
|---|---|---|
| `--config` | Có | File config |
| `--data-dir`, `--images-dir`, `--output-root` | Không | Ghi đè đường dẫn trong config, dùng trên Kaggle (thư mục ảnh được gắn riêng, chỉ đọc) |
| `--limit-images`, `--limit-texts` | Không | Chạy thử một phần. `meta.json` ghi `is_partial: true` |

Bản `is_partial: true` ghi vào thư mục `<dataset>_partial/` và **không được đăng**. Nếu thư mục đích đã tồn tại, script báo lỗi thay vì ghi đè.

## 6. Hành vi chi tiết

### Chọn 5 region (Visual Genome)

1. Sắp xếp ảnh theo `image_id`, và region của mỗi ảnh theo `region_id`.
2. Tạo một `np.random.default_rng(sampling_seed)` rồi lần lượt xử lý từng ảnh theo thứ tự đó.
3. Ảnh có từ 5 region trở lên: chọn 5 không lặp. Ảnh có ít hơn 5: lấy hết. Ảnh không có region: chỉ có
   embedding ảnh.

Trước khi chọn, bỏ các `phrase` rỗng (7 trong 5.408.689 region), vì chuỗi rỗng không phải truy vấn. Ngoài ra việc chọn không lọc theo nội dung và không xem điểm truy vấn. Các `region_id` được chọn nằm sẵn trong
`text_ids`, nên kiểm tra lại được. Chạy lại với cùng seed phải ra đúng các id này.

### Mã hoá

- `model.eval()`, `torch.no_grad()`, và tất cả tham số đặt `requires_grad=False`.
- Dùng đúng `preprocess` và tokenizer của open_clip. Văn bản dài hơn 77 token bị cắt theo mặc định.
- Dùng float32, không dùng autocast, để code đơn giản và kết quả ổn định hơn.
- Chuẩn hoá L2 sau khi mã hoá.

### Ảnh lỗi

Ảnh không mở được thì bỏ qua, và id được ghi vào `failed_image_ids`. Nếu quá **0,1%** số ảnh lỗi, script dừng
và báo lỗi, vì khi đó nhiều khả năng dữ liệu tải về có vấn đề. Văn bản của ảnh lỗi cũng bị bỏ, để mọi
`image_ids` trong `text_features.npz` đều có trong `image_features.npz`.

### Kiểm tra sau khi chạy (`verify_features`)

Script tự chạy các kiểm tra này. Có lỗi thì báo thất bại:

| Kiểm tra | Điều kiện đạt |
|---|---|
| Kích thước | Số hàng của mọi mảng khớp nhau và khớp `meta.json` |
| Giá trị | Không có NaN/Inf, độ dài mỗi vector xấp xỉ 1 |
| Id | `image_ids` ảnh không trùng; mỗi `image_ids` của văn bản đều có trong tập ảnh |
| Số văn bản mỗi ảnh | Visual Genome tối đa 5; Flickr30k đúng 5 |
| Hợp lý về ngữ nghĩa | Trên 1.000 cặp lấy ngẫu nhiên, cosine trung bình của cặp đúng lớn hơn cặp bị xáo trộn |

Kiểm tra cuối chỉ để phát hiện lệch hàng giữa văn bản và ảnh. Kết quả không được dùng để chọn tham số nào.

## 7. Chống rò rỉ dữ liệu (PDF §6)

- Trích đặc trưng cho **toàn bộ** ảnh, kể cả test, là an toàn: CLIP đóng băng, không học gì từ dữ liệu.
- Bước này không tạo cạnh thống kê và không chọn siêu tham số.
- Kaggle **không** chứa split. Split chính thức nằm trong `splits/` để không có hai phiên bản.
- Văn bản thuộc về split của ảnh tương ứng. Code huấn luyện sau này phải lọc theo `image_ids`.
- 5 region được chọn trước khi có bất kỳ kết quả truy vấn nào.
- Nên nhờ subagent `leakage-auditor` kiểm tra trước khi mở PR.

## 8. Giấy phép và điều kiện đăng công khai

| Dữ liệu | Điều đã biết | Hệ quả |
|---|---|---|
| Visual Genome | Dataset card ghi CC BY 4.0 | Đăng được embedding, cần ghi nguồn |
| Ảnh Flickr30k | Trang chính thức: chỉ cho nghiên cứu/giáo dục phi thương mại, tuân thủ Flickr Terms of Use, tác giả không sở hữu bản quyền ảnh | Không đăng ảnh |
| Embedding Flickr30k | Điều khoản **không nói rõ** về dữ liệu dẫn xuất | Xem điều kiện dưới đây |

**Điều kiện:** không ai trong nhóm hay Claude có thể tự kết luận về mặt pháp lý rằng embedding Flickr30k được
đăng công khai. Vì vậy Kaggle dataset của Flickr30k mặc định là **riêng tư, chia sẻ cho nhóm**, và chỉ chuyển
sang công khai sau khi đã đọc điều khoản nhận được khi đăng ký hoặc hỏi giảng viên. Visual Genome đăng công
khai được ngay sau khi ghi nguồn. Mọi điều kiện được ghi vào data card.

## 9. Kaggle

| Nội dung | Thiết kế |
|---|---|
| Thứ tự trong notebook | (1) Config: `DATASET = "visual_genome"` hoặc `"flickr30k"`, là giá trị duy nhất cần đổi. (2) Code. (3) Smoke. (4) Dataset 1: Visual Genome. (5) Dataset 2: Flickr30k. Mỗi lần chỉ chạy phần của `DATASET`, vì 2 bộ được đăng riêng và có chế độ hiển thị khác nhau |
| Output smoke | Ghi vào `/kaggle/temp`, không vào `/kaggle/working`, để không bị đăng nhầm cùng kết quả thật |
| Notebook | Tự chứa code, không clone repo. Mỗi ô code sao chép đúng một module, bắt đầu bằng `%%writefile <đường dẫn tạm>/mmxai/...`; notebook ghi các file đó ra rồi import |
| Chống lệch code | `test_notebooks_match_src.py` so nội dung từng ô (bỏ dòng `%%writefile`) với file trong `src/mmxai/` |
| Theo dõi phiên bản | `code_hash` = sha256 nội dung các module đã chạy, theo thứ tự cố định, tính lúc chạy. Hai bộ dữ liệu phải có cùng `code_hash` |
| Mạng | Cần bật Internet trên Kaggle để cài `open-clip-torch`, tải trọng số và tải `dataset_flickr30k.json` |
| Thư viện | Dùng torch có sẵn trên Kaggle, cài `open-clip-torch==3.3.0` (bản trong `uv.lock`), ghi phiên bản vào `meta.json` |
| Smoke | Chạy thử 1.000 ảnh + 10.000 văn bản rồi `verify_features`. Lỗi thì dừng, không chạy phần đầy đủ |
| Ảnh Flickr30k | Gắn [`eeshawn/flickr30k`](https://www.kaggle.com/datasets/eeshawn/flickr30k), thư mục `flickr30k_images/`. Danh sách file đã đối chiếu 2026-11-04: 31.783 ảnh không trùng tên, có đủ 31.014 ảnh của Karpathy. Nội dung ảnh sẽ được kiểm tra khi chạy (`failed_image_ids`). Caption vẫn lấy từ `dataset_flickr30k.json`, không dùng `captions.txt` của bản sao |
| Ảnh Visual Genome | Gắn [`lyte69/visualgenome-part1-part2`](https://www.kaggle.com/datasets/lyte69/visualgenome-part1-part2). Danh sách file đã đối chiếu 2026-11-04: có đủ 108.077 ảnh; 170 file rỗng đều không nằm trong danh sách cần dùng. Bản này để `VG_100K` và `VG_100K_2` ở hai thư mục khác nhau nên notebook tạo symlink. Không gắn thì notebook tự tải |
| Xuất bản | Dùng Save Version → Save & Run All để giữ output. Kiểm tra chạy trước khi thư mục có tên cuối cùng, nên kết quả lỗi không xuất hiện trong output |
| Kaggle dataset | `clip-features-visual-genome`, `clip-features-flickr30k` |
| Bí mật | Không cần token GitHub. Không lưu token Kaggle trong repo hay notebook. Người đăng tự thao tác bằng tài khoản của họ |

**Thời gian ước tính** (chưa đo; giả định 1 GPU T4/P100, khoảng 4 CPU):

| `DATASET` | Có sẵn ảnh trên Kaggle | Phải tải ảnh |
|---|---|---|
| Visual Genome | 20–40 phút | Tới ~75 phút |
| Flickr30k | 10–25 phút | Phụ thuộc nguồn ảnh |

Kết quả chạy thử trong notebook sẽ thay các số này bằng số đo thật.

## 10. Thứ tự thực hiện

Mỗi bước là một thay đổi nhỏ kèm test, rồi dừng lại cho người duyệt (`CLAUDE.md`):

1. ADR về chọn dữ liệu và định dạng/nơi chia sẻ (thay phần liên quan của ADR-0008).
2. `utils/config.py` + test.
3. `features/cache.py` và `features/verify.py` + test.
4. `features/clip_encoder.py` + test.
5. `data/flickr30k.py` + test, rồi `data/visual_genome.py` + test.
6. `scripts/extract_features.py`, 2 config, chạy thử trên máy bằng `--limit-*`.
7. Notebook Kaggle + `test_notebooks_match_src.py`, chạy thật, data card, đăng dataset.

Cần `uv sync` chạy được trước bước 2. Hiện máy này chưa cài `uv`.

## 11. Ảnh hưởng tới tài liệu hiện có

| Tài liệu | Cần làm |
|---|---|
| ADR-0008 | ADR mới thay phần định dạng cache và nơi chia sẻ. Không sửa nội dung ADR cũ |
| ADR-0012 | ADR mới nêu ngoại lệ: notebook `1x` = chuẩn bị dữ liệu, chứa bản sao code thay vì chỉ `import mmxai` |
| `CLAUDE.md` | Sửa dòng ghi cache đặc trưng ở `data/`; ADR-0008 ghi `artifacts/` |
| `data/README.md`, `data_cards/README.md` | Thêm Flickr30k, ghi phiên bản Visual Genome |
| `scripts/README.md`, `configs/README.md` | Thêm script và config mới |

## 12. Câu hỏi còn mở

| Câu hỏi | Trạng thái |
|---|---|
| Giới hạn số phiên GPU chạy song song của Kaggle | Chưa xác nhận. Nếu không cho phép thì chạy lần lượt, không phải sửa code |

## 13. Phần để người xây dựng tự quyết

Tên hàm phụ, cách chia batch, thông báo log và mẫu dữ liệu giả trong test, miễn là giữ đúng các hành vi và định
dạng ở trên.
