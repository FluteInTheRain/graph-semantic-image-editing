# ADR-0009: Chống rò rỉ dữ liệu bằng split cố định được commit và test tự động

- **Trạng thái:** Đã chấp nhận
- **Ngày:** 2026-11-01

## Bối cảnh

**Rò rỉ dữ liệu (data leakage)** = thông tin của tập test "lọt" vào lúc huấn luyện hoặc lúc chọn tham số → kết quả đẹp giả tạo. Đề (§6) có 6 quy định riêng và giảng viên sẽ hỏi về nó (§9):

| # | Quy định của đề | Cách repo đảm bảo |
|---|---|---|
| 1 | Ảnh/truy vấn test không dùng huấn luyện, chọn siêu tham số, xây cạnh thống kê | Code nhận split, hàm huấn luyện/xây cạnh chỉ nhận id `train` |
| 2 | Cạnh thống kê (đồng xuất hiện) xây từ train | `graph/` nhận danh sách id train làm tham số bắt buộc |
| 3 | `alpha` và siêu tham số chọn trên val | Hàm chọn `alpha` chỉ nhận dữ liệu val; test chỉ được chạy sau khi chốt |
| 4 | Bộ 20 truy vấn chỉ để đánh giá | `queries/` không được đọc bởi code huấn luyện/tuning |
| 5 | Mọi mô hình dùng cùng file chia | File trong `splits/` được commit; config trỏ tới file |
| 6 | ≥ 3 seed, báo cáo mean ± std | Xem ADR-0011 |

## Quyết định

1. **Split là file, được commit**: `splits/<dataset>/split_seed<k>.json` với khoá `train`, `val`, `test` (định dạng: `splits/README.md`).
2. Split sinh bằng script có seed (`scripts/make_splits.py`, sẽ viết), không chỉnh tay. Bộ có split chính thức (FB15k-237-IMG, MKG-W/Y) giữ split gốc.
3. Mọi config ghi rõ `split_file`; kết quả lưu kèm đường dẫn + hash của file split.
4. Truy vấn đánh giá (`queries/*.jsonl`) chỉ trỏ tới ảnh thuộc `test`.
5. Test pytest bắt buộc (viết cùng code `data/`):
   - ba tập không giao nhau, cho mọi file trong `splits/`;
   - id trong `queries/` thuộc tập test;
   - hàm xây cạnh thống kê gọi với id ngoài train → lỗi;
   - hàm chọn `alpha` không bao giờ nhận dữ liệu test.
6. Quy trình báo cáo: chốt config trên val → chạy test **một lần** cho mỗi seed → ghi số vào `results/tables/`.

## Các phương án đã cân nhắc

| Phương án | Ưu | Nhược | Lý do không chọn |
|---|---|---|---|
| Commit file split + test tự động | Ai cũng dùng đúng một split, máy kiểm tra hộ | Thêm file vào repo | **Được chọn** |
| Chỉ commit seed + script | File nhỏ | Kết quả phụ thuộc phiên bản thư viện/thứ tự đọc file | Đề cho phép, nhưng kém an toàn; vẫn giữ script để sinh lại |
| Chia ngẫu nhiên mỗi lần chạy | Dễ | Mỗi mô hình một split, so sánh vô nghĩa | Vi phạm §6 |
| Kiểm tra bằng mắt khi review | Không tốn code | Dễ sót | Không đáng tin |

## Hệ quả

- So sánh CLIP vs CLIP+Graph và các ablation luôn công bằng.
- Khi vấn đáp, chỉ cần mở test để chứng minh không rò rỉ.
- Đổi split = file mới (`split_seed1.json`), không ghi đè file cũ đã có kết quả.

## Nguồn tham khảo

- Kapoor & Narayanan, "Leakage and the Reproducibility Crisis in ML-based Science", Patterns (2023): https://doi.org/10.1016/j.patter.2023.100804
- Đề bài, §6 (chia dữ liệu và tránh rò rỉ), §8.
