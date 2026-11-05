# ADR-0042: Chia Visual Genome theo ảnh thành train / val 5.000 / test 5.000, gom ảnh trùng vào cùng tập

- **Trạng thái:** Đề xuất
- **Ngày:** 2026-11-05

## Bối cảnh

**Split** là danh sách cho biết ảnh nào thuộc tập **train** (huấn luyện GNN, xây cạnh thống kê), **val** (chọn `alpha` và siêu tham số) và **test** (báo cáo số cuối, chạy một lần). Visual Genome không có split chính thức (ADR-0040), nên nhóm phải tự tạo.

Đề yêu cầu (PDF §6, §8):

- Ảnh ứng viên và truy vấn của test không dùng để huấn luyện, chọn siêu tham số hay xây cạnh thống kê.
- Mọi mô hình so sánh dùng **cùng một tệp chia**; nộp tệp chia hoặc seed và mã tạo split.

Đề **không** quy định tỉ lệ chia hay cách dựng truy vấn. ADR-0009 đã chốt định dạng file (`splits/<dataset>/split_seed<k>.json`). ADR-0011 đã chốt: split cố định, còn 3 seed `[0, 1, 2]` chỉ thay đổi phần huấn luyện.

Dữ liệu sau khi trích embedding (data card `data_cards/visual_genome.md`): 108.077 ảnh, mỗi ảnh có 5 region description (một ảnh có 3). Data card ghi có **5 vector ảnh trùng hệt một ảnh khác**, nhiều khả năng là cùng một bức ảnh lưu hai lần. Nếu một bản nằm ở train và bản kia ở test thì mô hình đã "thấy" ảnh test lúc huấn luyện, tức là rò rỉ.

## Quyết định

1. **Đơn vị chia là ảnh.** Văn bản (region description) thuộc tập của ảnh tương ứng, lọc theo `image_ids` trong `text_features.npz`.
2. **Kích thước:** test 5.000 ảnh, val 5.000 ảnh, train là phần còn lại (khoảng 98.077 ảnh).
3. **Cách đánh giá dự kiến:** truy vấn là region description của ảnh test, kho tìm kiếm là toàn bộ ảnh test, đáp án đúng là ảnh gốc của câu. Val dùng cùng cách với ảnh val.
4. **Ảnh trùng:** hai ảnh có cosine giữa hai embedding ≥ `0.99` được coi là cùng một bức ảnh (có thể đã đổi kích thước hoặc nén lại) và vào cùng một **nhóm**; nhóm nối theo chuỗi (A gần B, B gần C thì A, B, C cùng nhóm). Cả nhóm luôn nằm cùng một tập. Ngưỡng nằm trong config (`duplicate_threshold`) và được chốt ngày 2026-11-05, trước khi có bất kỳ kết quả truy vấn nào. Lý do chọn ngưỡng rộng: gom thừa vô hại (chỉ giữ vài ảnh giống nhau ở cùng một tập), còn gom thiếu thì gây rò rỉ. Số đếm trên toàn bộ 108.077 ảnh ngày 2026-11-05: 5 cặp trùng hệt từng số, 16 cặp có cosine ≥ 0,999, 247 cặp ≥ 0,99; ví dụ `285982` và `2375578` có cosine 0,99992 dù không trùng hệt.
5. **Thuật toán:** sắp các nhóm theo `image_id` nhỏ nhất, xáo bằng `np.random.default_rng(split_seed)`, rồi lần lượt đặt từng nhóm vào test nếu test còn đủ chỗ cho cả nhóm, nếu không thì vào val nếu val còn chỗ, còn lại vào train.
6. **Một file split duy nhất:** `splits/visual_genome/split_seed0.json` với `split_seed: 0`. Seed huấn luyện tách riêng (ADR-0011).
7. **Nguồn id:** `image_features.npz` của bộ đặc trưng `ViT-B-32__laion2b_s34b_b79k` (ADR-0041). Tham số nằm trong `configs/splits_visual_genome.yaml`. Code: `src/mmxai/data/splits.py`, `scripts/make_splits.py`.

## Các phương án đã cân nhắc

| Phương án | Ưu | Nhược | Lý do không chọn |
|---|---|---|---|
| Test và val cố định 5.000 ảnh, train phần còn lại | Theo quy ước benchmark MS-COCO 5k; kho test nhỏ nên FAISS nhanh; train lớn cho GNN | Test chỉ chiếm ~4,6% | **Được chọn** |
| Tỉ lệ 80 / 10 / 10 | Phổ biến, dễ giải thích | Kho test ~10.800 ảnh; Recall khó so với các bài báo dùng 1k/5k | Không có lợi ích rõ cho đồ án |
| Test 1.000 ảnh (như Flickr30k Karpathy) | Recall cao, dễ đọc | Ít truy vấn (~5.000), std giữa các seed lớn hơn | Kho quá nhỏ với 108.077 ảnh sẵn có |
| 3 file split (seed 0, 1, 2) | CLIP thuần cũng có std | Mỗi split cần huấn luyện GNN riêng; trái ADR-0011 | ADR-0011 đã chốt split cố định |
| Chia ngẫu nhiên, bỏ qua ảnh trùng | Code đơn giản hơn | Có thể có cùng ảnh ở cả train và test | Vi phạm PDF §6 |
| Chỉ gom ảnh có embedding trùng hệt từng số | Đơn giản (`np.unique`) | Bỏ sót bản sao đã đổi kích thước/nén lại (ít nhất 11 cặp có cosine ≥ 0,999) | `leakage-auditor` phát hiện, số đếm xác nhận |
| Gom theo hash file ảnh | Chính xác cho bản sao byte-giống-byte | Không có ảnh gốc trên máy; vẫn bỏ sót ảnh nén lại | Không bắt được trường hợp chính |

## Hệ quả

- **Rò rỉ dữ liệu:** ảnh trùng không thể nằm ở hai tập khác nhau. Code đồ thị và huấn luyện phải lọc ảnh và văn bản theo split trước khi dùng (ADR-0009).
- **File split nặng ~1,4 MB** (108.077 id dạng chuỗi, mỗi id một dòng theo `splits/README.md`), vượt giới hạn 1 MB của `check-added-large-files`. Thư mục `splits/` được loại khỏi kiểm tra này trong `.pre-commit-config.yaml`, vì file split là sản phẩm bắt buộc phải nộp (PDF §8), không phải dữ liệu hay artifact.
- **Nguồn sự thật là file đã commit**, không phải lần chạy lại script. Numpy không cam kết giữ nguyên luồng ngẫu nhiên giữa các phiên bản, nên chạy lại bằng numpy khác có thể ra split khác. Code luôn đọc file trong `splits/` và lọc theo danh sách tường minh của từng tập (không suy "không thuộc train thì là test").
- **CLIP thuần có std = 0:** split cố định và CLIP đóng băng là tất định, nên std qua 3 seed chỉ phản ánh nhiễu khi huấn luyện GNN. Báo cáo cần nói rõ điều này.
- **Còn thiếu test `queries/` ⊆ test** (ADR-0009): viết khi định dạng `queries/*.jsonl` được chốt.
- **Split phụ thuộc bộ đặc trưng:** nếu trích lại embedding bằng model khác thì nhóm ảnh trùng có thể khác, nhưng split đã commit vẫn giữ nguyên và vẫn dùng được, vì nó chỉ chứa `image_id`.
- **Câu trùng chưa xử lý:** data card ghi 26% vector văn bản trùng vector khác (câu ngắn như "a man" xuất hiện ở nhiều ảnh). Một câu như vậy có thể khớp nhiều ảnh test, làm Recall@K thấp không công bằng. Cách xử lý (bỏ truy vấn trùng, hay coi mọi ảnh có cùng câu là đáp án đúng) thuộc phần đánh giá và cần ADR riêng.
- **Việc tiếp theo:** tải embedding từ Kaggle, chạy `scripts/make_splits.py`, commit file split; sau đó dựng FAISS index và metric trên tập test.

## Nguồn tham khảo

- Đề bài, §6 (chia dữ liệu và tránh rò rỉ), §7, §8.
- ADR-0009, ADR-0011, ADR-0040, ADR-0041.
- Data card `data_cards/visual_genome.md` (mục Hạn chế và rủi ro).
