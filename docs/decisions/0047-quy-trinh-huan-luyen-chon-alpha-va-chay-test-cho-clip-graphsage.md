# ADR-0047: Huấn luyện trên train, chọn epoch và `alpha` trên val, chạy test một lần bằng script riêng cho CLIP + GraphSAGE

- **Trạng thái:** Đề xuất
- **Ngày:** 2026-11-12

## Bối cảnh

ADR-0046 chốt **mô hình** (GraphSAGE một lớp, khái niệm → ảnh) và **mục tiêu học** (InfoNCE với hard negative từ CLIP top 100). ADR đó chưa nói rõ mấy việc mà code phải quyết định:

- Dừng huấn luyện ở epoch nào, và đo "tốt hơn" bằng gì.
- Chọn `alpha` thế nào khi hai giá trị cho cùng một điểm.
- Làm sao cho tập test **không thể** ảnh hưởng tới lựa chọn nào (PDF §6, ADR-0009).
- Laptop chỉ có CPU, 4 luồng và khoảng 9 GB RAM trống.

Khái niệm cho người mới:

- **Early stopping (dừng sớm):** sau mỗi epoch, chấm mô hình trên val. Giữ trọng số của epoch tốt nhất. Dừng khi `patience` epoch liền không tốt hơn.
- **Đồ thị con của một split:** chỉ gồm ảnh của split đó, mọi khái niệm, và cạnh của các ảnh đó. Với mô hình một lớp, điểm của một ảnh chỉ phụ thuộc vào cạnh của chính nó, nên chấm trên đồ thị con cho **đúng** kết quả như chấm trên đồ thị đầy đủ.

## Quyết định

| Việc | Quy định | Code |
|---|---|---|
| Câu dùng để huấn luyện | **Chỉ** các câu train mà ảnh gốc nằm trong top 100 của CLIP (tìm trong ảnh train). Lý do ở mục "Lần chạy đầu" bên dưới | `texts_with_positive_in_candidates` |
| Tín hiệu early stopping | MRR@100 trên **val** của điểm đồ thị **một mình** (`alpha = 0`). Epoch 0 (chưa học, bằng CLIP) cũng được chấm; nếu không epoch nào hơn thì giữ mô hình chưa học | `graph_only_val_score`, `fit` |
| Giới hạn | `max_epochs = 10`, `patience = 2`, `batch_size = 512` | `configs/clip_graph_vg.yaml` |
| Chọn `alpha` | Sau khi có trọng số tốt nhất: thử lưới `0.0 … 1.0` trên **val**, chọn MRR@100 cao nhất. **Hoà thì chọn `alpha` lớn hơn** (gần CLIP hơn) | `sweep_alpha`, `choose_alpha` |
| Ứng viên | CLIP thuần và CLIP + Graph gọi **cùng một hàm** `retrieve_candidates` (cùng 100 ứng viên, cùng ảnh đúng) | `rerank/fusion.py`, `eval/clip_baseline.py` |
| Chấm một split | Trên đồ thị con của split đó (`split_subgraph`), bằng `SAGEConv` thật | `build_split_graph`, `image_vectors` |
| Tăng tốc lúc huấn luyện | Tính trước trung bình vector khái niệm của mỗi ảnh, rồi `h = lin_l(mean) + lin_r(x)`. Chính xác vì mô hình một lớp, dùng mean, không có hàm kích hoạt. Test chứng minh kết quả trùng với `SAGEConv` | `forward_from_neighbor_mean` |
| Tách test | Hai script. `train_graph_rerank.py`: chỉ train + val, lưu `model.pt` và `selection.json` (`alpha`, epoch). `evaluate_graph_rerank.py`: nạp mô hình, chấm test với `alpha` đã lưu, **không chọn gì**; **từ chối** chạy lần hai cùng experiment và seed | `train_and_select`, `evaluate_test` |
| Đối chứng không đồ thị | Cùng code, cùng huấn luyện, `use_graph: false`: `W_nb` cố định bằng 0 (không học), `W_self` và bias vẫn học | `configs/clip_nograph_vg.yaml` |
| Seed | `[0, 1, 2]`, một hàm `set_seed` (ADR-0011). Seed đổi thứ tự batch và hard negative được lấy mẫu | `utils/seed.py` |
| Máy | CPU laptop. Phần tìm hard negative (khoảng 10 phút) làm một lần cho mọi seed | `prepare` |

### Lần chạy đầu: vì sao phải lọc câu huấn luyện

Bản đầu lấy **mọi** câu train. Ảnh đúng luôn có mặt trong danh sách ứng viên, kể cả khi CLIP xếp nó ngoài top 100; còn 15 ảnh sai thì luôn nằm trong top 100. Mô hình học được một "mẹo": ứng viên mà CLIP thích **ít nhất** thường là ảnh đúng, tức là đảo ngược CLIP. Trên val, cả 100 ứng viên đều từ top 100 của CLIP, nên mẹo này phá thứ hạng: MRR@100 của điểm đồ thị trên val tụt từ 0,106 (epoch 0, bằng CLIP) xuống **0,038** sau một epoch, trong khi loss train giảm còn 1,23 (2026-11-12, run bị dừng, không lưu kết quả).

Lọc như trên làm ảnh đúng và ảnh sai đến từ **cùng một tập**: đúng bài toán xếp hạng lại. Ảnh đúng nằm ngoài top 100 thì đằng nào cũng không thể được đẩy lên.

Quyết định bởi: Claude Code đề xuất ngày 2026-11-12, **nhóm chưa duyệt**. Các giá trị lấy từ ADR-0046 (`k = 16`, `n_hard = 15`, `τ = 0.05`, `lr = 1e-3`) vẫn là đề xuất của ADR-0046.

## Các phương án đã cân nhắc

| Phương án | Ưu | Nhược | Lý do không chọn |
|---|---|---|---|
| Huấn luyện trên mọi câu train, ảnh đúng có thể ngoài top 100 | Nhiều dữ liệu hơn | Mô hình học đảo ngược CLIP (đã thấy ở lần chạy đầu) | Sai bài toán |
| Chỉ câu có ảnh đúng trong top 100 | Đúng bài toán xếp hạng lại; ảnh đúng và ảnh sai cùng một tập | Bỏ bớt câu khó | **Được chọn** |
| Thêm ảnh sai ngẫu nhiên ngoài top 100 | Giữ mọi câu | Vẫn lệch với lúc chấm (chỉ có top 100); thêm một siêu tham số | Phức tạp hơn |
| Early stop bằng MRR của điểm đồ thị một mình | Đúng thứ loss đang học; tách lựa chọn epoch khỏi lựa chọn `alpha` | Epoch tốt nhất cho điểm đồ thị chưa chắc tốt nhất cho điểm kết hợp | **Được chọn** |
| Early stop bằng MRR tốt nhất qua mọi `alpha` | Nhắm thẳng vào kết quả cuối | Hai lựa chọn trộn vào nhau, khó giải thích khi vấn đáp | Phức tạp hơn mà lợi ích không rõ |
| Early stop bằng loss trên val | Không cần xếp hạng | Loss không phải metric được báo cáo | Lệch với mục tiêu |
| Hoà `alpha` → chọn lớn hơn | Không rời baseline khi không có bằng chứng | Có thể bỏ qua một `alpha` nhỏ cũng tốt như vậy | **Được chọn** |
| Một script làm cả val lẫn test | Gọn | Dễ chạy lại test sau khi xem số, tức là "debug đến khi đẹp" | Vi phạm tinh thần ADR-0009 |
| Chạy `SAGEConv` trên đồ thị train đầy đủ mỗi bước | Một đường code duy nhất | PyG tạo một bản sao 512 số cho mỗi cạnh: 1,6 triệu cạnh ≈ 3,2 GB mỗi lần, quá RAM laptop | Dùng shortcut có test chứng minh tương đương |
| Kaggle GPU | Nhanh hơn | Phải upload đồ thị, bản sao code trong notebook | Đo trên laptop: khoảng 220 GFLOP/s, đủ dùng |

## Hệ quả

- **Rò rỉ dữ liệu (PDF §6, ADR-0009):**
  - Huấn luyện chỉ dùng ảnh và câu của train. Hard negative chỉ tìm trong ảnh train. Có test: `test_training_never_sees_val_or_test_images`.
  - Epoch và `alpha` chọn trên val. Script test không có lưới `alpha`, và từ chối chạy lần hai cho cùng seed.
  - `queries/` không được dùng ở đâu trong quy trình này.
  - Đồ thị được kiểm là dựng trên đúng file split (so `split_sha256`).
- **Giải thích (ADR-0016):** `explain/` dùng `ConceptToImageSage.forward` (SAGEConv thật) trên `edge_index` khái niệm → ảnh, `num_layers = 1`. Shortcut chỉ dùng lúc huấn luyện.
- **Tái lập:** `model.pt` và `selection.json` nằm trong `artifacts/models/` (không commit). Run test ghi `model_sha256` và commit của mô hình. `summarize_runs.py` bỏ run có `git_dirty`, nên phải commit code trước khi chạy.
- **Hạn chế:**
  - Chọn `alpha` sau khi chọn epoch là chọn tham lam (greedy), không tối ưu chung.
  - Tên thư mục run tính theo phút: hai run cùng experiment và seed trong cùng một phút sẽ báo lỗi.
- **Việc tiếp theo:** chạy 3 seed cho `clip_graph_vg` và `clip_nograph_vg`; chạy test; cập nhật `results/tables/main.csv`. Sau đó: ablation (xáo cạnh, khái niệm ngẫu nhiên, cạnh annotation) và nối với `explain/`.

## Nguồn tham khảo

- Đề bài, §4.2, §6, §7.
- ADR-0009, ADR-0010, ADR-0011, ADR-0016, ADR-0043, ADR-0046.
- Mã nguồn `SAGEConv` của PyTorch Geometric 2.8.1 đã cài (`torch_geometric/nn/conv/sage_conv.py`): `lin_l` áp lên trung bình hàng xóm (có bias), `lin_r` áp lên chính đỉnh (không bias).
