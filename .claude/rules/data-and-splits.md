---
paths:
  - "splits/**"
  - "queries/**"
  - "src/mmxai/data/**"
  - "src/mmxai/graph/**"
  - "scripts/make_splits.py"
  - "scripts/build_graph.py"
  - "mmkg/**"
  - "data_cards/**"
---

# Quy tắc cho dữ liệu, split và đồ thị

- File trong `splits/` và `queries/` đã commit là **đóng băng**: hook `guard-protected-files.sh` chặn
  sửa. Muốn đổi: người dùng tự bật `MMXAI_ALLOW_PROTECTED_EDIT=1`, viết ADR, chạy lại mọi mô hình.
- Split sinh bằng script có seed cố định (`scripts/make_splits.py`), deterministic: chạy lại ra
  đúng file cũ. Định dạng `splits/<dataset>/split_seed<k>.json` (khoá `train`, `val`, `test`), chỉ
  lưu id. Truy vấn trong `queries/*.jsonl` chỉ trỏ tới ảnh thuộc `test` (ADR-0009).
- Hàm dựng cạnh thống kê (đồng xuất hiện, PMI, tần suất) nhận **danh sách id train** làm tham số bắt
  buộc – không có tham số mặc định "dùng hết". Có test chứng minh ảnh test không góp vào cạnh.
- Cạnh lấy từ KG ngoài (ConceptNet/Wikidata) không phụ thuộc split, nhưng ghi rõ nguồn + phiên bản.
- Không tải dữ liệu vào repo: dữ liệu ở `data/` (gitignored). Ghi nguồn, giấy phép vào `data_cards/`.
- MMKG tự xây (cấp độ 3): ghi data card (nguồn ảnh/văn bản, cách tạo triple, điều kiện sử dụng),
  không thu thập dữ liệu cá nhân nhạy cảm.
