# ADR-0054: Đánh giá giải thích định lượng trên MMKG tiếng Việt, dùng nguyên lựa chọn của Visual Genome

- **Trạng thái:** Đề xuất
- **Ngày:** 2026-11-18

## Bối cảnh

Cấp độ 3 (PDF §5) yêu cầu đánh giá giải thích **định lượng**: fidelity, validity, sparsity trên ít nhất 10 truy vấn. Phần này đã làm trên Visual Genome (VG, PR #19, ADR-0016): explainer chính là **Integrated Gradients (IG)** với **m = 10** cạnh, chọn trên val VG theo characterization, test chạy một lần (`results/tables/xai_level3.csv`, báo cáo mục 7.3).

Một **giải thích** ở đây là tập m cạnh `contains` (khái niệm → ảnh) mà IG cho là đóng góp nhiều nhất vào điểm đồ thị của ảnh được xếp hạng 1. **Fidelity+** đo điểm giảm bao nhiêu khi bỏ đúng các cạnh đó. Muốn tin giải thích thì phải giảm nhiều hơn khi bỏ cùng số cạnh **ngẫu nhiên** (đối chứng). **Validity** kiểm cạnh có thật trong đồ thị đã chấm. **Sparsity** đo giải thích dùng bao nhiêu phần trong số cạnh của ảnh.

ADR-0053 đã chạy mô hình VG trên MMKG tiếng Việt mà không huấn luyện lại. Câu hỏi của ADR này: giải thích của mô hình đó trên MMKG có **trung thực** không, tức các thực thể MMKG được chọn làm giải thích có thực sự tạo ra điểm hay không.

## Quyết định

Do qphulong duyệt ngày 2026-11-18 (Claude Code đề xuất), trước khi chạy.

1. **Mô hình:** GNN một lớp `clip_graph_vg_lr1e4`, trọng số và `alpha` của từng seed 0/1/2 như ADR-0053. Đây là mô hình được giải thích ở cấp độ 3 trên VG, và là mô hình duy nhất mà `src/mmxai/explain/` hỗ trợ (`sage_scoring.image_scorer`). Đồ thị là đồ thị MMKG của ADR-0053: 119 thực thể, cạnh `contains` CLIP top-16.
2. **Explainer và kích thước: IG, m = 10**, lấy nguyên từ lựa chọn trên val VG (`configs/explain_vg.yaml`). Không chọn lại gì trên MMKG. Mỗi ảnh có 16 cạnh nên giải thích bỏ ngoài 6 cạnh.
3. **Truy vấn: toàn bộ 318 câu** của `queries/mmkg_vi/queries.jsonl` (không lấy mẫu), nên không có bước chọn truy vấn nào; kết quả tách theo nhóm trực tiếp/gián tiếp × Anh/Việt như ADR-0053.
4. **Ảnh được giải thích:** ảnh hạng 1 của từng truy vấn dưới CLIP + Graph (cùng quy tắc VG).
5. **Thước đo và tham số:** giữ nguyên `configs/explain_vg.yaml`: fidelity+/−, tụt hạng, đối chứng ngẫu nhiên cùng kích thước (20 lần), robust fidelity, sparsity, validity; kiểm định Wilcoxon một phía "giải thích hơn ngẫu nhiên" theo truy vấn; mean ± std qua 3 seed.
6. **Kiểm tái lập trước khi giải thích:** với từng seed, xếp hạng lại toàn bộ truy vấn phải cho đúng MRR@100 của run chuyển giao đã có (`results/runs/*_clip_graph_vg_lr1e4_mmkg_vi_seed<k>/metrics.json`); lệch thì dừng.
7. **Chạy một lần** (như test): thư mục `results/runs/<thời gian>_explain_mmkg_vi_test/`; lần chạy thứ hai bị từ chối.

## Các phương án đã cân nhắc

| Phương án | Ưu | Nhược | Lý do không chọn |
|---|---|---|---|
| Dùng nguyên IG, m = 10 của VG, toàn bộ 318 câu | Không chọn gì trên MMKG; đủ truy vấn để tách nhóm | Lựa chọn của VG có thể không tối ưu cho MMKG | **Được chọn** |
| Chọn lại explainer và m trên MMKG | Có thể tốt hơn | MMKG không có val; chọn trên chính tập đánh giá là rò rỉ | Vi phạm PDF §6 |
| Lấy mẫu 20 câu như VG | Giống hệt VG | Ít câu, không tách nhóm được; thêm một bước chọn | Chạy toàn bộ rẻ (380 ảnh) |
| Giải thích mô hình hai lớp (có triple) | Triple xuất hiện trong giải thích | `explain/` chưa hỗ trợ hai lớp; phải viết và kiểm explainer mới | Ngoài phạm vi; ghi là việc sau |

## Hệ quả

- **Diễn giải:** trên MMKG, mô hình một lớp không hơn đối chứng không đồ thị (ADR-0053). Fidelity ở đây đo giải thích có trung thực với **mô hình** hay không, không nói đồ thị giúp xếp hạng. Với `alpha` ≈ 0, điểm cuối gần như chỉ còn điểm đồ thị, nên bỏ cạnh có thể đổi thứ hạng.
- **Rò rỉ:** không huấn luyện, không chọn tham số nào trên MMKG. Cạnh không đọc `entity_id`.
- Triple MMKG không xuất hiện trong giải thích của mô hình một lớp (chỉ có cạnh `contains`).
- Việc tiếp: bảng `results/tables/xai_mmkg.csv`, một tiểu mục trong báo cáo mục 8.

## Nguồn tham khảo

- Đề bài §5 (cấp độ 3), §6 (rò rỉ).
- ADR-0016 (explainer tính trên chính mô hình, chọn trên val), ADR-0053 (chuyển giao sang MMKG), `configs/explain_vg.yaml`, báo cáo mục 7.3.

## Kết quả (2026-11-18)

Run `results/runs/20261118-2258_explain_mmkg_vi_test/`, bảng `results/tables/xai_mmkg.csv`. Xếp hạng lại khớp MRR@100 của run chuyển giao ở cả 3 seed. 954 giải thích (318 truy vấn × 3 seed): Fid+ 0,0094 ± 0,0003 so với ngẫu nhiên 0,0001; tụt hạng 0,82 ± 0,04 bậc so với 0,10 (Wilcoxon p ≈ 1e-59); validity 100 %; sparsity 0,375. Đúng ở cả bốn nhóm truy vấn. Characterization không dùng được (giá trị chuẩn hoá bị cắt ở mọi truy vấn, ≈ 0 với tiếng Việt). Phân tích: báo cáo mục 8.4.
