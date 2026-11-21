# ADR-0058: Cho phép một thành viên duyệt bộ truy vấn KG

- **Trạng thái:** Đề xuất
- **Ngày:** 2026-11-21

## Bối cảnh

ADR-0057 sinh 28 cặp truy vấn trực tiếp / gián tiếp trên test Visual Genome. Quy trình cũ, lấy từ ADR-0044 và mục 4 của `docs/query_benchmark.md`, cần **hai người**:

- **Người duyệt A** xem ảnh, giữ ảnh đúng và chuyển cặp sang `reviewed`.
- **Người kiểm B**, khác A, xem lại và chuyển sang `verified`.

`build_verified_benchmark` (`src/mmxai/queries/validator.py`) từ chối dòng `verified` nếu thiếu `checker`, hoặc `checker` trùng `reviewer`.

Đề bài (PDF §5) chỉ yêu cầu bộ ≥ 20 truy vấn, có nhóm gián tiếp, và báo cáo tách riêng trực tiếp / gián tiếp. Đề **không** bắt buộc hai người duyệt. Hai người là quy tắc nhóm tự đặt để giảm sai sót. Trong thực tế chỉ một thành viên có thời gian duyệt: Nguyễn Thiện Thuật đã duyệt cả 28 cặp ngày 2026-11-21 trên trang duyệt có ảnh. Kết quả là 19 cặp được giữ và 9 cặp bị bỏ, có ghi lý do.

## Quyết định

- **Bộ truy vấn KG (ADR-0057) chỉ cần một người duyệt.**
  - Một dòng `verified` cần có `reviewer`, `verified_relevant` không rỗng, và chỉ chứa ảnh test.
  - `checker` có thể để trống.
- Config `configs/query_benchmark_vg_kg.yaml` có `required_reviewers: 1`. Mặc định của code vẫn là `2`, nên bộ v1 (ADR-0044) giữ nguyên quy tắc hai người.
- Manifest chuẩn hoá theo quy tắc mới:
  - cặp giữ lại có `verification_status = verified`, `reviewer = Nguyễn Thiện Thuật`, `checker` rỗng;
  - cặp bỏ đi giữ `rejected` và lý do trong `notes`.
- File `queries/query_review_manifest_kg.csv` do **người** chép vào repo. Hook `guard-protected-files.sh` khoá file đã commit trong `queries/` với AI.
- Báo cáo phải ghi rõ: bộ truy vấn được **một thành viên** kiểm tra ảnh đúng.

## Các phương án đã cân nhắc

| Phương án | Ưu | Nhược | Lý do không chọn |
|---|---|---|---|
| Một người duyệt, ghi rõ trong báo cáo | Làm được ngay; đáp án vẫn do người xem ảnh | Không có người thứ hai bắt lỗi; có thể sót ảnh đúng hoặc giữ nhầm ảnh sai | **Được chọn** |
| Giữ hai người, chờ thành viên khác | Chặt chẽ hơn | Chặn toàn bộ bước build và đánh giá; nhóm không có người rảnh | Không kịp tiến độ |
| Để AI làm người kiểm thứ hai | Nhanh | AI không xem được ảnh trên máy này; ghi tên người vào cột `checker` là sai sự thật (PDF §9) | Không trung thực |

## Hệ quả

- **Dễ hơn:** build được bộ chính thức ngay, với 19 cặp, tức 38 truy vấn, trên mức tối thiểu 10 cặp.
- **Khó hơn / rủi ro:** chỉ một người đánh giá ảnh nên độ tin cậy của đáp án thấp hơn. Bản báo cáo và bảng kết quả phải nêu giới hạn này.
- **Rò rỉ (PDF §6):** không đổi. Người duyệt chỉ nhìn câu và ảnh, không xem kết quả của mô hình nào. Bộ truy vấn vẫn chỉ dùng để đánh giá.
- **Việc tiếp theo:**
  - người dùng chép manifest đã chuẩn hoá vào `queries/`;
  - chạy `scripts/build_kg_query_benchmark.py`, rồi đánh giá các mô hình theo `docs/query_benchmark.md`.

## Nguồn tham khảo

- ADR-0044, ADR-0057; `docs/query_benchmark.md`, mục 4.
- `src/mmxai/queries/validator.py`, các hàm `check_verified_row` và `build_verified_benchmark`.
- Đề bài `Project03_Multimodal_XAI.pdf`, §5, §6, §9.
