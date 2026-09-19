# queries/ — bộ truy vấn đánh giá (CÓ commit)

**Là gì?** Bộ **≥ 20 truy vấn văn bản** tự xây (yêu cầu cấp độ 2), mỗi truy vấn có ảnh đúng (ground truth). Chia hai nhóm:

| Nhóm | Ý nghĩa | Ví dụ |
|---|---|---|
| `direct` (trực tiếp) | Từ trong truy vấn nhìn thấy được ngay trong ảnh | "a red bus on the street" |
| `indirect` (gián tiếp) | Phải suy qua quan hệ trong đồ thị mới nối được | "a child playing football" khi ảnh chỉ có `ball`, `field` |

Kết quả phải báo cáo **tách riêng** hai nhóm để chứng minh vai trò của đồ thị.

## Định dạng: JSON Lines (`queries/<tên>.jsonl`, mỗi dòng một truy vấn)

```
{"qid": "q001", "text": "a child playing football", "type": "indirect", "dataset": "visual_genome", "relevant": ["2345"], "path_hint": "child - plays - football", "author": "<tên thành viên>"}
```

| Trường | Bắt buộc | Ý nghĩa |
|---|---|---|
| `qid` | có | Mã duy nhất |
| `text` | có | Câu truy vấn |
| `type` | có | `direct` hoặc `indirect` |
| `dataset` | có | Bộ dữ liệu chứa ảnh đúng |
| `relevant` | có | Danh sách id ảnh đúng (≥ 1), phải nằm trong tập **test** |
| `path_hint` | không | Đường đi quan hệ người viết kỳ vọng (chỉ để phân tích, **không** đưa vào mô hình) |
| `author` | không | Người tạo, để kiểm tra chéo |

## Quy tắc

1. Bộ này **chỉ để đánh giá**: không tinh chỉnh mô hình, không chọn `alpha` trên nó (§6 của đề).
2. Mỗi truy vấn được một thành viên khác kiểm tra lại ảnh đúng; ghi cách xây dựng và kiểm tra vào báo cáo.
3. Sẽ có test pytest kiểm tra schema (đủ trường, `type` hợp lệ, `qid` không trùng).
