# configs/ — cấu hình thí nghiệm

**Là gì?** Mỗi file YAML mô tả **một thí nghiệm**: dữ liệu nào, file split nào, model CLIP nào, GNN nào, lưới `alpha`.

| File | Vai trò |
|---|---|
| `configs/clip_graph_vg.yaml` | Ví dụ mẫu (chưa dùng bởi code): CLIP + GCN trên Visual Genome. Copy để tạo thí nghiệm mới. |
| `configs/features_visual_genome.yaml` | Trích embedding CLIP cho Visual Genome (5 region mỗi ảnh) |
| `configs/features_flickr30k.yaml` | Trích embedding CLIP cho Flickr30k |
| `configs/splits_visual_genome.yaml` | Tạo split Visual Genome: test 5.000, val 5.000, train phần còn lại (ADR-0042) |

Quy tắc (lý do: `docs/decisions/0007-cau-hinh-yaml-va-dataclass.md`):
- YAML thuần. Code sau này sẽ đọc bằng `yaml.safe_load` rồi nạp vào `dataclass` (dự kiến đặt ở `src/mmxai/utils/`).
- Khoá lạ (gõ sai tên) phải gây lỗi ngay, không bị bỏ qua âm thầm.
- Một file = một thí nghiệm. Không dùng kế thừa/ghép config nhiều tầng.
- Không bao giờ ghi giá trị đã chọn trên tập test vào config.
