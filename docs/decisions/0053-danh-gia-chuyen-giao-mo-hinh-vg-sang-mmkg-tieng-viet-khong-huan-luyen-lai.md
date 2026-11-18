# ADR-0053: Đánh giá chuyển giao mô hình Visual Genome sang MMKG tiếng Việt, không huấn luyện lại

- **Trạng thái:** Đề xuất
- **Ngày:** 2026-11-18

## Bối cảnh

Cấp độ 3 (PDF §5) yêu cầu hai việc trên MMKG tự xây (`mmkg/`, ADR-0050, ADR-0052): **chạy phương pháp tốt nhất** trên dữ liệu này và **phân tích khả năng chuyển giao**.

**Phương pháp tốt nhất** là hai mô hình GNN của cấp độ 1, huấn luyện trên Visual Genome (VG). Kết quả test VG (`results/tables/main.csv`, 3 seed, MRR@100):

| Mô hình | Config | MRR@100 |
|---|---|---|
| CLIP thuần | `configs/clip_vg.yaml` | 0,1076 |
| GNN một lớp (ADR-0046) | `configs/clip_graph_vg_lr1e4.yaml` | 0,1284 ± 0,0006 |
| GNN hai lớp, cạnh đồng xuất hiện (ADR-0048) | `configs/clip_cooc_graph_vg_lr1e4.yaml` | 0,1280 ± 0,0006 |

G-Retriever bậc 1 (ADR-0051) chọn `alpha = 1,0` trên val, tức trùng CLIP thuần, nên không phải ứng viên.

**Chuyển giao** nghĩa là: lấy mô hình đã học ở một nơi (VG: ảnh đời thường, khái niệm như "man", "tree", câu tiếng Anh) rồi dùng ở nơi khác (MMKG: di tích và hiện vật Việt Nam, 119 thực thể, câu tiếng Việt) mà không học lại. Làm được vì trọng số của hai mô hình chỉ là các ma trận 512 × 512 nhân lên vector CLIP đóng băng (`src/mmxai/models/sage.py`), không gắn với danh sách khái niệm nào. Đổi tập khái niệm thì chỉ đổi đồ thị, không đổi trọng số.

MMKG chỉ dùng để **đánh giá**: 380 ảnh, 80 hiện vật có ảnh, 39 khái niệm, 303 triple. Không có tập train hay val trên MMKG. Vì vậy **mọi lựa chọn phải chốt trước khi có kết quả**, giống luật dùng bộ truy vấn đánh giá (CLAUDE.md, ADR-0009). Phương án bàn giao đã thống nhất giữa Khang và qphulong ngày 2026-11-17 (mô tả PR #21).

## Quyết định

Các quyết định 1–5 do qphulong duyệt ngày 2026-11-18 (Claude Code đề xuất), trước khi chạy bất kỳ mô hình nào trên MMKG.

1. **Mô hình và `alpha`: dùng nguyên của VG.** Với mỗi mô hình (một lớp, hai lớp) và mỗi seed 0/1/2: nạp `artifacts/models/visual_genome/<config>/seed<k>/model.pt` và `alpha` trong `selection.json` cùng thư mục (chọn trên val VG). Không huấn luyện, không chọn lại gì trên MMKG. So với CLIP thuần, cùng ứng viên.

2. **Đỉnh khái niệm: toàn bộ 119 thực thể MMKG** (80 hiện vật + 39 khái niệm). Vector mỗi đỉnh = vector văn bản CLIP của `"a photo of a {name_en}"`, như VG (`concept_prompt` của `configs/graph_vg_clip_topk.yaml`). Dùng tên tiếng Anh cho **cả** truy vấn tiếng Anh lẫn tiếng Việt: đồ thị giống hệt nhau, nên so sánh hai ngôn ngữ chỉ đo tác động của ngôn ngữ **truy vấn**.

3. **Cạnh `contains` ảnh → thực thể: CLIP top-k, `k = 16`**, như ADR-0046 trên VG. Mỗi ảnh nối với 16 thực thể có cosine CLIP cao nhất. **Không** đọc cột `entity_id` của `mmkg/images.csv` khi dựng cạnh: cột đó là đáp án (data card mục 9). Lưu ý: MMKG chỉ có 119 thực thể, nên 16 cạnh chiếm khoảng 13% số thực thể, trong khi trên VG chỉ là một phần nhỏ của bộ từ vựng. Không chỉnh `k` trên MMKG.

4. **Cạnh khái niệm → khái niệm (chỉ mô hình hai lớp): triple MMKG**, mỗi triple `(head, relation, tail)` thành cạnh theo **hai chiều**, bỏ qua tên quan hệ, giống cạnh đồng xuất hiện đối xứng của ADR-0048. Triple lấy từ Wikidata, không phụ thuộc split, nên không có rủi ro rò rỉ kiểu cạnh thống kê. Khác biệt về nghĩa (VG học trên đồng xuất hiện, MMKG đưa vào quan hệ tri thức) là một phần của phân tích chuyển giao.

5. **Bộ truy vấn: 318 câu**, sinh tự động, ghi vào `queries/mmkg_vi/` (file mới, sau khi commit thì đóng băng như mọi file trong `queries/`). Với mỗi hiện vật trong 80 hiện vật có ảnh:
   - **Trực tiếp** (có tên): tiếng Anh `name_en`, tiếng Việt `name_vi`.
   - **Gián tiếp** (giấu tên): mô tả bằng **mọi** triple có hiện vật làm `head`, mỗi quan hệ một mẫu câu cố định (tiếng Anh và tiếng Việt, cùng nội dung), dùng tên của `tail`. Dùng mọi quan hệ chứ không chỉ 6 quan hệ mô tả (`instance_of`, `made_from`, `culture`, `kept_at`, `located_in`, `time_period`), vì 22 hiện vật không có quan hệ nào trong 6 quan hệ đó.
   - Mẫu câu viết trong `src/mmxai/data/mmkg_queries.py` (`CLAUSES`), có test, chốt trước khi chạy. Sinh bằng `scripts/make_mmkg_queries.py --config configs/mmkg_vi_transfer.yaml`.
   - Dự kiến 80 × 2 × 2 = 320 câu; thực tế **318**: hiện vật `Q910146` ("Ngưu Lang Chức Nữ / The Weaver Girl and the Cowherd", 5 ảnh) không có triple nào, nên không có câu gián tiếp (mô tả rỗng khớp mọi hiện vật, mọi ảnh sẽ là đáp án). Hiện vật này là một truyền thuyết chứ không phải di tích; giữ nguyên trong MMKG vì đã qua duyệt, chỉ ghi nhận ở đây.

6. **Đáp án đúng: mọi ảnh của mọi hiện vật khớp câu truy vấn.**
   - Trực tiếp: hiện vật có cùng tên (không có hai hiện vật trùng `name_en` hay `name_vi`, nên thực tế là ảnh của chính hiện vật đó).
   - Gián tiếp: hiện vật có **đủ** các cặp `(relation, tail)` được mô tả. Một mô tả chung chung (vd. chỉ "a National Treasure of Vietnam") có nhiều hiện vật đúng; như vậy không phạt mô hình khi trả về một hiện vật đúng mô tả nhưng không phải hiện vật gốc.

7. **Mỗi câu tính một lần.** Pipeline VG gắn mỗi câu với một ảnh và tìm đáp án bằng câu trùng (ADR-0043); dùng cách đó thì hiện vật 5 ảnh bị tính 5 lần. Trên MMKG, mỗi câu là **một dòng truy vấn với tập ảnh đúng ghi sẵn**.

8. **Metric và ứng viên:** như ADR-0043, ADR-0047: xếp hạng lại 100 ứng viên đầu của CLIP (trên 380 ảnh), R@1, R@5, R@10, MRR@100, 3 seed, mean ± std (`ddof = 1`). Báo cáo tổng và tách theo 4 nhóm: {trực tiếp, gián tiếp} × {tiếng Anh, tiếng Việt}. Câu gián tiếp tách thêm theo **độ cụ thể**: *cụ thể* (đúng 1 hiện vật khớp, 35 câu mỗi ngôn ngữ) và *chung chung* (> 1 hiện vật, 44 câu mỗi ngôn ngữ; câu chung nhất "a heritage of Vietnam, listed as National Treasures of Vietnam" có 47 hiện vật đúng, nên gần như luôn trúng). Cách tách này chốt sau khi sinh câu nhưng **trước khi chạy mô hình** (2026-11-18), chỉ đổi cách trình bày, không đổi câu hay đáp án. Mô hình CLIP: `ViT-B-32` / `laion2b_s34b_b79k`, như VG.

### Bổ sung 2026-11-18: đối chứng không đồ thị

Chốt **trước khi chạy** (qphulong yêu cầu, sau review của `experiment-reviewer`; chưa xem kết quả của thí nghiệm này):

9. **Thêm `clip_nograph_vg_mmkg_vi`:** trọng số và `alpha` của `artifacts/models/visual_genome/clip_nograph_vg/seed<k>/` (đối chứng không đồ thị của ADR-0046, lr `1e-3`, đúng đối chứng đã báo cáo trên VG: \mrr{} 0,1266 trong `results/tables/main.csv`). Mô hình này có cùng phép chiếu học được nhưng `W_nb = 0` (đã kiểm: cả 3 seed, giá trị lớn nhất của `conv.lin_l.weight` bằng 0), nên cạnh không ảnh hưởng điểm. So với hai mô hình có đồ thị, nó cho biết mức tăng trên MMKG đến từ cấu trúc ảnh – thực thể hay chỉ từ phép chiếu. Chọn bản lr `1e-3` chứ không phải `clip_nograph_vg_lr1e4` vì chỉ bản này có kết quả test VG đã báo cáo. Cùng ứng viên, cùng truy vấn, cùng metric, 3 seed.

## Các phương án đã cân nhắc

| Phương án | Ưu | Nhược | Lý do không chọn |
|---|---|---|---|
| Dùng nguyên mô hình VG, không huấn luyện lại | Đo đúng "chuyển giao"; không cần split trên MMKG nhỏ | Khác miền lớn, có thể không giúp gì | **Được chọn** |
| Tinh chỉnh (fine-tune) trên một phần MMKG | Có thể tốt hơn | 380 ảnh quá ít để có train/val/test tin được; phải làm split và kiểm rò rỉ (data card mục 9) | Không đủ dữ liệu, ra ngoài phạm vi |
| Chọn lại `k` hoặc `alpha` trên MMKG | Điểm có thể cao hơn | Không có val trên MMKG: chọn trên chính tập đánh giá là rò rỉ | Vi phạm PDF §6 |
| Vector đỉnh từ tên tiếng Việt cho truy vấn tiếng Việt | Đồng bộ ngôn ngữ | Hai đồ thị khác nhau, không tách được ngôn ngữ truy vấn khỏi ngôn ngữ đồ thị | Lẫn hai yếu tố |
| Gián tiếp chỉ dùng 6 quan hệ mô tả | Câu tự nhiên hơn | 22 hiện vật không có câu | Thiếu hiện vật |
| Đáp án chỉ là hiện vật gốc | Đơn giản | Phạt mô hình khi trả về hiện vật khác đúng mô tả | Đánh giá sai |
| Tái dùng cách tìm đáp án bằng câu trùng (ADR-0043) | Không cần code mới | Hiện vật nhiều ảnh bị tính nhiều lần | Lệch trọng số |

## Hệ quả

- **Rò rỉ (PDF §6):** không có huấn luyện hay chọn tham số nào trên MMKG; `k`, `alpha`, mẫu câu chốt trước. Cạnh không đọc `entity_id` (đáp án). Bộ truy vấn MMKG chỉ để đánh giá, như `queries/` của VG.
- **Giải thích:** mô hình hai lớp dùng triple MMKG làm cạnh, nên triple ảnh hưởng điểm và có thể xuất hiện trong giải thích (Integrated Gradients, PR #19). Đánh giá giải thích trên MMKG là việc tuỳ chọn, chưa làm trong ADR này.
- **Diễn giải kết quả:** CLIP `laion2b` học chủ yếu trên tiếng Anh, nên dự kiến truy vấn tiếng Việt yếu hơn nhiều; đó là kết quả cần báo cáo, không phải lỗi. Trên VG, phần lớn mức tăng đến từ phép chiếu học được chứ không từ cạnh (`clip_nograph` 0,1266); phân tích chuyển giao cần nhắc điều này.
- **Thế giới đóng:** đỉnh khái niệm gồm cả 80 hiện vật, tức chính tập tên của đáp án. Ảnh không nối với hiện vật của nó qua nhãn mà qua CLIP, nên không rò nhãn, nhưng đồ thị biết trước danh sách tên (trên VG, đỉnh là từ vựng đối tượng chung). 2/158 câu gián tiếp vẫn chứa tên hiện vật qua tên của `tail` (vd. `has_part` của chùa Tây Phương); không sửa file truy vấn đã đóng băng, chỉ ghi nhận (leakage-auditor, 2026-11-18).
- **Ngoài giao thức:** bảng `results/tables/mmkg_transfer_query_spread.csv` (độ phân tán câu truy vấn theo CLIP) được thêm **sau** khi có kết quả, là phân tích khám phá; không đổi số liệu hay cấu hình nào. Nhãn `split = test` trong `metrics.json` chỉ nghĩa là "ảnh đánh giá": MMKG không có train/val.
- **Chưa làm** (nếu làm phải chốt trước trong ADR mới): bỏ/xáo triple cho mô hình hai lớp, xáo cạnh `contains`, kiểm định cặp theo truy vấn.
- Không có seed nào trong CLIP thuần và trong việc dựng đồ thị (tất định); khác biệt giữa seed chỉ đến từ trọng số VG.
- Việc tiếp: sinh truy vấn (`queries/mmkg_vi/`), trích đặc trưng CLIP, dựng đồ thị, chạy, ghi bảng `results/tables/`, viết phần chuyển giao trong báo cáo.

## Nguồn tham khảo

- Đề bài `Project03_Multimodal_XAI.pdf` §5 (cấp độ 3), §6 (rò rỉ dữ liệu).
- ADR-0043 (giao thức đánh giá), ADR-0046 (GNN một lớp, cạnh CLIP top-k), ADR-0047 (chọn `alpha`), ADR-0048 (GNN hai lớp), ADR-0051 (G-Retriever bậc 1), ADR-0050 và ADR-0052 (MMKG).
- `data_cards/mmkg_vi.md` mục 9 (rủi ro rò rỉ của `entity_id`); mô tả PR #21 (phương án bàn giao).
