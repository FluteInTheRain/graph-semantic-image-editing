# configs/ — cấu hình thí nghiệm

**Là gì?** Mỗi file YAML mô tả **một thí nghiệm**: dữ liệu nào, file split nào, model CLIP nào, GNN nào, lưới `alpha`.

| File | Vai trò |
|---|---|
| `configs/clip_graph_vg.yaml` | **CLIP + Graph**: GraphSAGE một lớp xếp hạng lại top 100 của CLIP; `alpha` và số epoch chọn trên `val` (ADR-0046, ADR-0047) |
| `configs/clip_nograph_vg.yaml` | Đối chứng **không đồ thị**: huấn luyện y hệt nhưng cố định `W_nb = 0` (ADR-0046 mục 6) |
| `configs/clip_cooc_graph_vg.yaml` | CLIP + Graph **có cạnh đồng xuất hiện**: GraphSAGE hai lớp, khái niệm nghe khái niệm đi cùng nó trong annotation train (ADR-0048) |
| `configs/concept_edges_vg_cooccurrence.yaml` | Dựng cạnh khái niệm → khái niệm bằng PMI trên annotation **train** (ADR-0048) |
| `configs/*_lr1e4.yaml` | Ba thí nghiệm trên với `lr = 1e-4` thay vì `1e-3` (val tốt nhất ngay epoch 1 rồi giảm); so với bản `1e-3` **chỉ trên val** |
| `configs/features_visual_genome.yaml` | Trích embedding CLIP cho Visual Genome (5 region mỗi ảnh) |
| `configs/features_flickr30k.yaml` | Trích embedding CLIP cho Flickr30k |
| `configs/clip_vg.yaml` | Baseline CLIP thuần trên Visual Genome; mặc định chạy `val`, `test` chỉ chạy một lần (ADR-0043) |
| `configs/vector_store_vg.yaml` | Vector store CLIP thuần của Visual Genome, một index cho `val` và một cho `test` |
| `configs/splits_visual_genome.yaml` | Tạo split Visual Genome: test 5.000, val 5.000, train phần còn lại (ADR-0042) |
| `configs/query_benchmark_vg.yaml` | Sinh cặp truy vấn trực tiếp/gián tiếp từ scene graph ảnh **test** và dựng bộ chính thức từ manifest đã kiểm tra (ADR-0044) |
| `configs/query_benchmark_vg_pilot.yaml` | Pilot 5 cặp trên **val** để chốt quy tắc; ra `results/runs/` (ADR-0044) |
| `configs/eval_query_benchmark_vg.yaml` | Đánh giá CLIP thuần trên bộ truy vấn đã kiểm tra, tách direct/indirect (ADR-0044) |
| `configs/graph_vg_clip_topk.yaml` | Đồ thị ảnh – khái niệm của **mô hình chính**: cạnh `contains` lấy từ CLIP top-k cho mọi split (ADR-0046) |
| `configs/graph_vg.yaml` | Đồ thị ảnh – khái niệm với cạnh `contains` lấy từ annotation (ADR-0045); chỉ làm ablation cận trên (ADR-0046) |

Quy tắc (lý do: `docs/decisions/0007-cau-hinh-yaml-va-dataclass.md`):
- YAML thuần. Code sau này sẽ đọc bằng `yaml.safe_load` rồi nạp vào `dataclass` (dự kiến đặt ở `src/mmxai/utils/`).
- Khoá lạ (gõ sai tên) phải gây lỗi ngay, không bị bỏ qua âm thầm.
- Một file = một thí nghiệm. Không dùng kế thừa/ghép config nhiều tầng.
- Không bao giờ ghi giá trị đã chọn trên tập test vào config.
