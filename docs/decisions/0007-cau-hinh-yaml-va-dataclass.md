# ADR-0007: Cấu hình thí nghiệm bằng YAML thuần + dataclass

- **Trạng thái:** Đã chấp nhận
- **Ngày:** 2026-11-01

## Bối cảnh

**Config** = file ghi các lựa chọn của một lần chạy (dữ liệu, model CLIP, loại GNN, `top_k`, lưới `alpha`...). Tách config khỏi code để:
- chạy nhiều thí nghiệm (CLIP vs CLIP+Graph, ablation, 3 seed) mà không sửa code;
- lưu lại đúng config đã dùng cạnh kết quả → tái lập.

Ràng buộc quan trọng: ở buổi vấn đáp, **mỗi thành viên** phải giải thích được config được đọc thế nào. Công cụ càng "phép màu" càng khó giải thích.

## Quyết định

1. Mỗi thí nghiệm là một file YAML trong `configs/` (ví dụ `configs/clip_graph_vg.yaml`). Không có kế thừa/ghép nhiều tầng.
2. Đọc bằng `yaml.safe_load` → dict → nạp vào `dataclass` có kiểu rõ ràng (code sẽ đặt ở `src/mmxai/utils/`).
3. Khoá không có trong dataclass → báo lỗi ngay (chống gõ sai tên khoá âm thầm).
4. Ghi đè nhanh trên dòng lệnh chỉ cho vài tham số thường đổi (`--seed`, `--alpha`), qua `argparse`.
5. Mỗi lần chạy copy config đã dùng (sau ghi đè) vào thư mục kết quả (ADR-0010).

Khung config:

| Khối | Ví dụ khoá |
|---|---|
| chung | `experiment`, `dataset`, `split_file`, `seeds`, `output_dir` |
| `features` | `model`, `pretrained`, `cache_dir` |
| `retrieval` | `top_k` |
| `graph` | `edges`, `gnn`, `hidden_dim`, `num_layers` |
| `rerank` | `alpha_grid` (chọn trên val) |

## Các phương án đã cân nhắc

| Phương án | Ưu | Nhược | Lý do không chọn |
|---|---|---|---|
| YAML + dataclass | ~30 dòng code tự viết, ai cũng đọc hiểu, có kiểu | Không có sweep tự động | **Được chọn** |
| argparse thuần | Không cần file | Hàng chục cờ, khó lưu/so sánh | Không lưu được cấu hình đầy đủ gọn gàng |
| OmegaConf | Nội suy biến, merge | Thêm khái niệm (interpolation, structured config) | Lợi ích nhỏ so với chi phí giải thích |
| Hydra | Mạnh: sweep, compose, tự tạo thư mục output | Đổi thư mục làm việc, nhiều "phép màu", khó debug | Khó giải thích khi vấn đáp |
| Python file làm config | Linh hoạt | Config có thể chứa logic, khó kiểm soát | Lẫn code với dữ liệu cấu hình |

## Hệ quả

- Lưới `alpha` và sweep seed viết bằng vòng `for` trong `scripts/evaluate.py` — rõ ràng, dễ trình bày.
- Muốn thêm tham số: thêm field vào dataclass + YAML, trong cùng một PR.
- Nếu sau này cần sweep lớn, có thể chuyển sang Hydra bằng ADR mới.

## Nguồn tham khảo

- Python `dataclasses`: https://docs.python.org/3/library/dataclasses.html
- PyYAML `safe_load`: https://pyyaml.org/wiki/PyYAMLDocumentation
- Hydra: https://hydra.cc/docs/intro/
