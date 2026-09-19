# splits/ — file chia train/val/test (CÓ commit)

**Split là gì?** Danh sách id cho biết ảnh/truy vấn nào thuộc tập **train** (huấn luyện), **val** (chọn siêu tham số, chọn `alpha`), **test** (chỉ dùng để báo cáo kết quả cuối).

**Vì sao commit?** Đề (§6) yêu cầu mọi mô hình so sánh dùng **cùng một file chia** và nộp file chia hoặc mã tạo split. Commit file giúp ai chạy lại cũng ra đúng cùng tập. Chi tiết: `docs/decisions/0009-chong-ro-ri-du-lieu-va-split-co-dinh.md`.

## Định dạng

Một file JSON cho mỗi (bộ dữ liệu, seed chia):

```
splits/<dataset>/split_seed<k>.json
{"dataset": "visual_genome", "seed": 0, "train": ["1", "2", ...], "val": [...], "test": [...]}
```

- Id là chuỗi, sắp xếp tăng dần để `git diff` dễ đọc.
- Tạo bằng `scripts/make_splits.py` (sẽ viết), không sửa tay.
- Với KG có sẵn split chính thức (FB15k-237-IMG, MKG-W/Y): giữ nguyên split gốc, chỉ chuyển sang định dạng này.

## Quy tắc chống rò rỉ

| Việc | Được dùng |
|---|---|
| Huấn luyện GNN, lớp chiếu | chỉ `train` |
| Xây cạnh thống kê (đồng xuất hiện, tần suất) | chỉ `train` |
| Chọn `alpha`, siêu tham số, early stopping | chỉ `val` |
| Báo cáo số cuối | `test`, chạy **một lần** sau khi đã chốt cấu hình |

Sẽ có test pytest kiểm tra: ba tập không giao nhau, và mọi file split commit đều hợp lệ.
