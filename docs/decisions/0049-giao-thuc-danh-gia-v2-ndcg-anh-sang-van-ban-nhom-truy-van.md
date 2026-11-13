# ADR-0049: Thêm nDCG@10, chiều ảnh → văn bản và đánh giá theo nhóm truy vấn; chạy lại 5 experiment dưới tên `_eval2`

- **Trạng thái:** Đề xuất
- **Ngày:** 2026-11-13

## Bối cảnh

PDF §7 có một bảng "Đánh giá kết quả". Hàng **Nhiệm vụ 1 – Truy vấn**, cột **Mở rộng**, gồm ba mục: **nDCG**, **chiều truy vấn thứ hai (ảnh → văn bản)** và **đánh giá theo thể loại hoặc nhóm truy vấn**. Nhóm muốn làm cả ba cho cấp độ 1, với cả CLIP thuần lẫn CLIP + Graph.

Một số khái niệm cho người mới:

- **nDCG@10** (normalized Discounted Cumulative Gain) chấm cả *thứ tự* của 10 kết quả đầu: kết quả đúng nằm càng cao thì càng được nhiều điểm, rồi chia cho điểm của thứ tự lý tưởng. Kết quả nằm trong khoảng 0–1. Khác với MRR, nDCG tính **mọi** đáp án đúng chứ không chỉ đáp án đầu tiên.
- **Ảnh → văn bản:** đưa một ảnh vào và tìm các câu mô tả đúng nó. Hiện repo chỉ đánh giá chiều văn bản → ảnh.
- **Nhóm truy vấn:** tách kết quả theo loại câu truy vấn, để xem đồ thị giúp ở đâu.

**Ràng buộc.** PDF §6 và ADR-0009 yêu cầu mọi lựa chọn làm trên val. Nhóm còn tự đặt một quy tắc chặt hơn: mỗi experiment chỉ chạy test **một lần**. Mọi mô hình văn bản → ảnh đã chạy test. Muốn có nDCG và số theo nhóm trên test thì phải chấm test lần nữa. Đó là một ngoại lệ của quy tắc nhóm, nên cần ghi lại ở đây.

Thiết kế chi tiết, gồm các phương án và các quyết định của người làm: `docs/superpowers/specs/2026-11-13-cap-do-1-mo-rong-design.md`.

## Quyết định

**1. nDCG@10.** Relevance nhị phân, dùng định nghĩa đáp án đúng của ADR-0043.
- Công thức: `DCG@10 = Σ_{p ≤ 10, đúng} 1/log2(p+1)` và `IDCG@10 = Σ_{j ≤ min(#đúng, 10)} 1/log2(j+1)`.
- Tính trên **cùng** danh sách đã xếp hạng mà R@K và MRR dùng (`ranked_metrics` trong `src/mmxai/eval/metrics.py`).

**2. Ảnh → văn bản.** Không huấn luyện thêm.
- Một mô hình chấm điểm một cặp (câu t, ảnh i): `α·cos(t, x_i) + (1−α)·cos(t, h_i)`.
- Truy vấn là mọi ảnh của tập đánh giá. Kho tìm kiếm là mọi câu đã lấy mẫu (5 câu mỗi ảnh) của các ảnh đó.
- Câu đúng: các câu của chính ảnh, cộng mọi câu giống hệt một trong các câu đó (ADR-0043).
- FAISS lấy top 100 câu, rồi xếp hạng lại; hoà thì giữ thứ tự của CLIP.
- `α` của chiều này được chọn **trên val**, sau khi đã chọn epoch và `α` của chiều văn bản → ảnh như cũ. Cùng lưới, cùng luật: MRR@100 cao nhất, hoà thì `α` lớn hơn. Lưu ở `selection.json` với các khoá `alpha_image_to_text` và `alpha_sweep_image_to_text`.
- CLIP thuần dùng `α = 1`.
- Code: `src/mmxai/eval/image_to_text.py`, `src/mmxai/eval/extended.py`.

**3. Nhóm truy vấn** (chỉ cho chiều văn bản → ảnh). Có hai cách chia độc lập. Code: `src/mmxai/eval/query_groups.py`.
- **Cách 1, câu riêng / câu chung:** câu có đúng 1 ảnh đúng là `specific`, câu có từ 2 ảnh đúng trở lên là `generic`.
- **Cách 2, quan hệ:** chuẩn hoá câu (chữ thường, chỉ giữ chữ cái), rồi khớp vị từ theo dãy từ liền nhau.
  - Danh sách vị từ đọc từ `configs/query_benchmark_vg.yaml` (ADR-0044), không sửa file đó.
  - Gán nhóm theo thứ tự ưu tiên: `action` nếu câu chứa một vị từ trong `action_predicates` hoặc alias của nó; nếu không thì `spatial` nếu chứa một vị từ khác trong danh sách; còn lại là `none`.
- **Kiểm tra trên val trước khi chốt** (`scripts/inspect_query_groups.py`, 2026-11-13):
  - Cách 2: `action` 8,5%, `spatial` 39,5%, `none` 52,0%. Cách 1: `specific` 88,4%, `generic` 11,6%.
  - Đã đọc 30 câu ngẫu nhiên mỗi nhóm. **Luật không đổi**, nên các danh sách `extra_action_predicates`, `extra_spatial_predicates`, `excluded_predicates` đều rỗng.
  - Sai sót nhỏ được chấp nhận: "hanging **down** from" không khớp vị từ "hanging from"; động từ ngoài danh sách như "putting … in" rơi vào nhóm `spatial`.
  - Kết quả kiểm tra: `results/records/20261113_cap-do-1-mo-rong/pre_lock_checks/`.

**4. Chạy lại 5 experiment** dưới tên mới có hậu tố `_eval2`:
- `clip_vg_eval2`, `clip_nograph_vg_eval2`, `clip_graph_vg_lr1e4_eval2`, `clip_cooc_graph_vg_lr1e4_eval2`, `clip_graph_vg_lr1e4_shuffled_eval2`.
- Đây là ngoại lệ có điều kiện của quy tắc "test chạy một lần", nên phải thoả cả năm điều kiện sau:
  1. Config giữ nguyên so với bản gốc (split, siêu tham số, seed 0/1/2). Chỉ khác tên, `model_dir` và các trường mở rộng; có test kiểm (`tests/test_graph_rerank_pipeline.py`, `tests/test_clip_baseline.py`).
  2. ADR này và quy tắc báo cáo trong `results/tables/README.md` được commit **trước** mọi lần huấn luyện và chạy test `_eval2`. Mọi số mới đều được báo cáo, dù tốt hay xấu.
  3. Không xoá hay sửa dòng cũ trong `results/tables/main.csv`. Số mới nằm trong `results/tables/eval2_text_to_image.csv`, `eval2_image_to_text.csv` và `eval2_groups.csv`.
  4. Mỗi experiment `_eval2` chạy test **đúng một lần** cho mỗi seed. CLIP thuần giờ cũng bị chặn chạy lại bằng code (`test_run_exists`).
  5. Chạy toàn bộ, gồm cả huấn luyện, trên cùng một máy và cùng một commit. File run được chép vào `results/records/<ngày>_cap-do-1-mo-rong/`.
- Không chạy lại `clip_graph_vg` (`lr = 1e-3`), vì đây không phải cấu hình chính.
- Kiểm tra trước khi chạy: code mới phải tái tạo **đúng** số val cũ của CLIP thuần. Đã đạt ngày 2026-11-13: R@1, R@5, R@10, MRR@100 khớp tới chữ số cuối với `results/runs/20261105-1029_clip_vg_seed0`.

Người quyết định: FluteInTheRain, 2026-11-13 (chạy lại, làm cả 3 mục, chia nhóm theo cách 1 và 2, cách làm A). Các chi tiết còn lại do Claude Code đề xuất trong bản thiết kế; nhóm chưa duyệt.

## Các phương án đã cân nhắc

| Phương án | Ưu | Nhược | Lý do không chọn |
|---|---|---|---|
| Chạy lại 5 experiment dưới tên `_eval2`, theo 5 điều kiện | Mọi mục mở rộng có số test; hai chiều dùng cùng một mô hình; mọi file run truy vết được | Ngoại lệ của quy tắc test chạy một lần; số mới có thể lệch nhẹ so với `main.csv` | **Được chọn** |
| Không chạy lại: nDCG và nhóm của chiều văn bản → ảnh chỉ báo trên val | Giữ nguyên quy tắc; nhanh | Bảng mở rộng lẫn val và test; hai chiều đến từ hai lần huấn luyện khác nhau | Kém nhất quán, số val hơi lạc quan |
| Chấm lại test bằng checkpoint của máy này cho các dòng đã có | Không phải huấn luyện lại | Dòng test cũ đến từ checkpoint của máy khác; R@K/MRR có thể không khớp `main.csv` mà không giải thích được | Không truy vết được |
| Script đánh giá riêng, tách khỏi pipeline | Không đụng code cũ | Thành hai đường chấm test song song, dễ lệch nhau | Chọn mở rộng pipeline hiện có |
| Chia nhóm theo thể loại đối tượng (WordNet) | Đúng chữ "thể loại" trong đề | Phải đoán đối tượng chính của câu; nhiều heuristic | Khó bảo vệ khi vấn đáp |

## Hệ quả

- **Dễ hơn:** báo cáo có nDCG, chiều ảnh → văn bản và số theo nhóm trên test, cho cả CLIP thuần và CLIP + Graph. Config cũ vẫn đi qua đúng đoạn code cũ. Test kiểm tập khoá của `metrics.json` và `selection.json`; giá trị thì chưa có test riêng, nhưng được bảo đảm vì đó chính là đoạn code cũ.
- **Khó hơn:** có hai bộ số cho cùng một cấu hình (`main.csv` và `eval2_*`). Báo cáo phải nói rõ bộ nào dùng cho mục nào, và ghi mức lệch giữa hai bộ.
- **Rò rỉ dữ liệu:** test được nhìn thêm một lần cho các cấu hình đã có. Không lựa chọn nào dựa trên lần nhìn này: `α` của chiều ảnh → văn bản và luật chia nhóm đều chốt trên val trước khi chạy. Mọi lựa chọn mới đều chỉ dùng val.
- **Việc tiếp theo:** dựng cạnh đồng xuất hiện trên máy chạy; huấn luyện 4 mô hình đồ thị × 3 seed; chạy test một lần; sinh 3 bảng mới; viết hồ sơ phiên và cập nhật guide.

## Nguồn tham khảo

- `Project03_Multimodal_XAI.pdf`: §6 (chia dữ liệu, tránh rò rỉ), §7 (bảng Đánh giá kết quả, cột Mở rộng).
- ADR-0009 (rò rỉ dữ liệu), ADR-0010 (theo dõi thí nghiệm bằng file), ADR-0043 (giao thức đánh giá, đáp án đúng), ADR-0044 (danh sách vị từ), ADR-0047 (chọn `α` trên val).
- `docs/superpowers/specs/2026-11-13-cap-do-1-mo-rong-design.md`, `docs/superpowers/plans/2026-11-13-cap-do-1-mo-rong.md`.
