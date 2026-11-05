# ADR-0043: Đánh giá truy vấn văn bản → ảnh trên Visual Genome bằng region description, coi mọi ảnh có cùng câu là đáp án đúng

- **Trạng thái:** Đề xuất
- **Ngày:** 2026-11-05

## Bối cảnh

Nhiệm vụ 1 (PDF §1, §7) yêu cầu Recall@1, Recall@5, Recall@10 và MRR cho chiều văn bản → ảnh, và so sánh CLIP thuần với CLIP + Graph **trên cùng cách chia**. Visual Genome không có bộ truy vấn chuẩn, nên nhóm phải tự định nghĩa: truy vấn là gì, tìm trong kho nào, ảnh nào là đáp án đúng.

Khái niệm cho người mới:

- **Recall@K**: tỉ lệ truy vấn có ít nhất một ảnh đúng nằm trong K kết quả đầu.
- **MRR** (Mean Reciprocal Rank): trung bình của `1 / hạng của ảnh đúng đầu tiên`. Ảnh đúng ở hạng 1 được 1 điểm, ở hạng 4 được 0,25.

Dữ liệu có sẵn (ADR-0041, ADR-0042): mỗi ảnh có 5 region description đã mã hoá bằng CLIP; split chia theo ảnh, test có 5.000 ảnh. Data card ghi **26% vector văn bản trùng một vector khác**: câu ngắn như "a man" xuất hiện ở nhiều ảnh. Nếu chỉ coi ảnh gốc của câu là đáp án, CLIP trả về một ảnh khác cũng có đúng câu đó vẫn bị tính là sai.

## Quyết định

| Thành phần | Quy định |
|---|---|
| Truy vấn | Mọi region description (đã chọn 5 mỗi ảnh) của ảnh thuộc tập đánh giá |
| Kho tìm kiếm | Mọi ảnh thuộc tập đánh giá (test: 5.000 ảnh) |
| Đáp án đúng | Mọi ảnh trong kho có ít nhất một region description **cùng câu** với truy vấn, gồm cả ảnh gốc |
| "Cùng câu" | Cosine giữa hai vector văn bản ≥ `0.9999`. File `.npz` không chứa chuỗi văn bản, nên so vector thay cho so chuỗi. Ngưỡng này chỉ để bù sai số số học khi GPU tính theo batch (cỡ 1e-6), **không** được chỉnh theo kết quả |
| Tìm kiếm | FAISS `IndexFlatIP` (tìm chính xác). Embedding đã chuẩn hoá L2 nên inner product = cosine |
| Số ứng viên | `top_k = 100`, giống số ứng viên mà CLIP + Graph sẽ xếp hạng lại |
| Metric | Recall@1/5/10 và MRR@100 (ảnh đúng không nằm trong 100 kết quả đầu thì tính 0). Ghi thêm số truy vấn và tỉ lệ truy vấn có nhiều hơn một đáp án |
| Tập dùng | `val` được chạy tuỳ ý khi phát triển; `test` chạy **một lần** cho cấu hình đã chốt (ADR-0009) |
| Seed | CLIP đóng băng và FAISS Flat là tất định: chạy 3 seed sẽ ra cùng một số, std = 0 (ADR-0011) |

Code: `src/mmxai/eval/` (metric, đáp án đúng), `src/mmxai/index/` (FAISS), `scripts/evaluate_clip.py`, `configs/clip_vg.yaml`.

## Các phương án đã cân nhắc

| Phương án | Ưu | Nhược | Lý do không chọn |
|---|---|---|---|
| Mọi ảnh có cùng câu đều đúng | Không phạt CLIP khi nó trả về một ảnh khác cũng đúng | Cần thêm bước tìm câu trùng | **Được chọn** |
| Chỉ ảnh gốc là đúng | Đơn giản nhất | Câu trùng làm Recall thấp giả tạo, nhiều nhiễu | Không công bằng với 26% truy vấn |
| Bỏ các truy vấn có câu trùng | Mỗi truy vấn đúng một đáp án | Bỏ phần lớn câu ngắn, bộ truy vấn lệch về câu dài | Thay đổi phân bố truy vấn |
| So chuỗi văn bản thay vì vector | Chính xác tuyệt đối | Phải tải thêm `region_descriptions.json` (~GB) | Vector đủ dùng với ngưỡng bù sai số số học |
| MRR trên toàn bộ thứ hạng | Không bị cắt ở 100 | Không cùng điều kiện với CLIP + Graph (chỉ xếp hạng lại top 100) | So sánh hai mô hình sẽ không công bằng |

## Hệ quả

- **So sánh công bằng:** CLIP thuần và CLIP + Graph dùng cùng truy vấn, cùng kho, cùng định nghĩa đáp án và cùng 100 ứng viên.
- **Rò rỉ dữ liệu:** bước tìm câu trùng chỉ so các câu trong cùng tập đánh giá, không dùng nhãn hay kết quả của mô hình. Ngưỡng `0.9999` được chốt trước khi chạy và không được đổi sau khi xem kết quả test.
- **Giới hạn:** hai câu khác chữ nhưng cùng nghĩa ("a man" và "man") vẫn là hai câu khác nhau. Ảnh test trùng nhau (ADR-0042) không được tính là đáp án của nhau; theo data card chỉ có 5 cặp trên toàn bộ dữ liệu nên ảnh hưởng không đáng kể.
- **MRR bị cắt ở 100** nên hơi thấp hơn MRR trên toàn bộ thứ hạng; báo cáo phải ghi là MRR@100.
- **Việc tiếp theo:** chạy trên val để kiểm pipeline, rồi trên test một lần; ghi bảng vào `results/tables/main.csv`.

## Nguồn tham khảo

- Đề bài, §1, §4.1, §6, §7.
- ADR-0009, ADR-0010, ADR-0011, ADR-0041, ADR-0042.
- Data card `data_cards/visual_genome.md` (mục Hạn chế và rủi ro).
