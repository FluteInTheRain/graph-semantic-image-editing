# ADR-0041: Lưu embedding dạng .npz + meta.json và chia sẻ qua Kaggle

- **Trạng thái:** Đề xuất
- **Ngày:** 2026-11-04

## Bối cảnh

**Embedding** là một dãy số biểu diễn một ảnh hoặc một câu. Đề yêu cầu trích đặc trưng một lần rồi lưu lại (PDF §4.1). Máy cá nhân trong nhóm không đủ mạnh để trích trên toàn bộ dữ liệu, và mọi người cần dùng đúng một bộ đặc trưng.

ADR-0008 quy định `image_emb.npy` + `image_ids.json` và chia sẻ qua Google Drive, nhưng chưa nói về embedding văn bản. ADR-0012 quy định notebook chỉ `import mmxai`, trong khi Kaggle không truy cập được repo riêng tư nếu không có token.

## Quyết định

1. **Định dạng** (thay phần định dạng cache của ADR-0008):

   ```text
   <output_root>/<dataset>/<model>__<pretrained>/
   ├── image_features.npz   # embeddings, image_ids
   ├── text_features.npz    # embeddings, image_ids, text_ids hoặc caption_index
   └── meta.json
   ```

   Hàng `i` của mọi mảng trong cùng file nói về cùng một đối tượng. Dùng `np.savez`, không nén.
2. **Văn bản:** Flickr30k lấy toàn bộ caption. Visual Genome lấy **5 region mỗi ảnh**, chọn ngẫu nhiên với `sampling_seed: 0`, sau khi bỏ `phrase` rỗng.
3. **Chia sẻ** (thay phần "Bản nộp" của ADR-0008 cho embedding): mỗi bộ dữ liệu một Kaggle dataset. Visual Genome công khai. Flickr30k **riêng tư, chia sẻ cho nhóm**, cho tới khi xác nhận điều khoản cho phép đăng công khai. Không đăng ảnh gốc.
4. **Notebook** (ngoại lệ của ADR-0012): `notebooks/1x_*` là notebook chuẩn bị dữ liệu, chứa bản sao của các module. `src/mmxai/` vẫn là bản gốc có test. `tests/test_notebooks_match_src.py` báo lỗi nếu bản sao bị lệch.
5. **Theo dõi phiên bản:** `meta.json` ghi `code_hash`, là sha256 của các module đã chạy, tính lúc chạy.

## Các phương án đã cân nhắc

| Phương án | Ưu | Nhược | Lý do không chọn |
|---|---|---|---|
| `.npz` + `meta.json` | Embedding và id không thể lệch nhau, ít file | Không dùng được `mmap_mode` | **Được chọn** |
| `.npy` + JSON (ADR-0008) | Đọc được từng phần | Nhiều file, id tách khỏi embedding | Mảng lớn nhất ~1,1 GB nên không cần `mmap_mode` |
| Toàn bộ 5,4 triệu region | Đủ dữ liệu | Khoảng 11 GB | Quá nặng để cả nhóm tải |
| Kaggle | Cả nhóm tải dễ, chạy được trên GPU của Kaggle | Thêm một nền tảng | **Được chọn** |
| Google Drive (ADR-0008) | Đơn giản | Không có GPU, tải chậm hơn khi dùng trên Kaggle | Không giải quyết được việc thiếu GPU |
| Notebook có bản sao code | Không cần token, Kaggle không cần truy cập repo | Notebook dài, cần test chống lệch | **Được chọn** |
| Clone repo bằng token trong Kaggle Secrets | Notebook ngắn, có `git_commit` | Phải tạo và quản lý token | Người dùng chọn không cấp quyền truy cập repo |
| Chỉ có code trong notebook | Đơn giản nhất | Không test được, không import được | Trái quy ước `CLAUDE.md` |

## Hệ quả

- Không còn `git_commit` trên Kaggle. Hai bộ dữ liệu phải có cùng `code_hash`.
- Sửa code xong phải chạy `scripts/sync_notebook_code.py`, nếu không test sẽ báo lỗi.
- **Rò rỉ dữ liệu:** trích đặc trưng cho cả ảnh test là an toàn vì CLIP đóng băng. Region được chọn trước khi có kết quả truy vấn. Split không nằm trên Kaggle mà nằm trong `splits/`.
- Bộ ≥ 20 truy vấn trong `queries/` được mã hoá riêng, không thuộc phạm vi này.

## Nguồn tham khảo

- Đề bài, §4.1, §6, §9.
- ADR-0008, ADR-0012.
- Thiết kế: `docs/superpowers/specs/2026-11-04-clip-embeddings-design.md`.
- NumPy `savez`: https://numpy.org/doc/stable/reference/generated/numpy.savez.html
