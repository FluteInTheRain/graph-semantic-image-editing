# ADR-0008: Lưu dữ liệu và artifact lớn ngoài git; cache đặc trưng dạng .npy + metadata JSON

- **Trạng thái:** Đã chấp nhận
- **Ngày:** 2026-11-01

## Bối cảnh

**Artifact** = file sinh ra từ pipeline: đặc trưng CLIP, chỉ mục FAISS, đồ thị đã dựng, checkpoint GNN. Chúng lớn (Visual Genome ~108k ảnh × 512 chiều float32 ≈ 220 MB đặc trưng), git không hợp để lưu.

Đề yêu cầu: đặc trưng CLIP "trích sẵn một lần và lưu lại" (§4.1); nộp qua Moodle, nếu quá dung lượng thì "nộp lên cloud, chia sẻ link" (§9).

## Quyết định

Ba vùng lưu trữ:

| Vùng | Đường dẫn | Nội dung | Git |
|---|---|---|---|
| Dữ liệu thô | `data/<dataset>/` | File tải về, **không sửa** | Bỏ qua (`/data/*`), chỉ commit `data/README.md` |
| Artifact | `artifacts/` | Cache đặc trưng, FAISS, đồ thị, checkpoint | Bỏ qua |
| Nhỏ, cần tái lập | `splits/`, `queries/`, `mmkg/*.csv`, `results/tables/` | File văn bản nhỏ | **Commit** |

Quy ước cache đặc trưng (một bộ cho mỗi model + dataset):

```
artifacts/features/<dataset>/<model>__<pretrained>/
    image_emb.npy      # float32, shape (N, D), đã chuẩn hoá L2
    image_ids.json     # danh sách id, cùng thứ tự với hàng của .npy
    meta.json          # model, pretrained, open_clip version, ngày tạo, số ảnh, git commit
```

Quy ước khác:
- FAISS: `artifacts/index/<dataset>/<model>.faiss`.
- Checkpoint GNN: `artifacts/checkpoints/<experiment>/seed<k>.pt`.
- Code kiểm tra `meta.json` khớp config trước khi dùng cache; không khớp → báo lỗi, không dùng nhầm.
- Bản nộp: nén `artifacts/` cần cho demo, đưa lên Google Drive của nhóm, dán link vào `README.md`.
- `.gitignore` chặn `*.npy`, `*.npz`, `*.faiss`, `*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`; pre-commit chặn file > 1 MB.

## Các phương án đã cân nhắc

| Phương án | Ưu | Nhược | Lý do không chọn |
|---|---|---|---|
| Ngoài git + link cloud + `meta.json` | Đơn giản, đúng §9 | Phải tự giữ link cập nhật | **Được chọn** |
| DVC | Version dữ liệu chuẩn, pipeline | Thêm công cụ, cần remote storage, học thêm | Quá nặng cho đồ án 1 học kỳ |
| Git LFS | Tích hợp git | Quota GitHub nhỏ (1 GB), tốn băng thông | Dễ hết quota với ảnh VG |
| `.npy` cho đặc trưng | Đọc nhanh, `np.load(mmap_mode="r")`, không cần torch | — | **Được chọn** |
| `.pt` (torch.save) | Tiện trong torch | Cần torch để đọc, rủi ro pickle | Muốn đọc được không cần torch |
| HDF5 / Parquet | Gom nhiều mảng | Thêm phụ thuộc | Không cần ở quy mô này |

## Hệ quả

- Ai clone repo cũng phải tải dữ liệu theo `data/README.md`; demo dùng artifact tải từ link.
- `meta.json` giúp trả lời vấn đáp "đặc trưng này từ model nào, tạo lúc nào".
- `.gitignore` gốc dùng `data/` đã được đổi thành `/data/*`: bản cũ vô tình bỏ qua cả `src/mmxai/data/`.

## Nguồn tham khảo

- NumPy `.npy` format: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
- DVC: https://dvc.org/doc
- Đề bài, §4.1 và §9.
