# app/ — giao diện nhỏ (tuỳ chọn)

**Là gì?** Chỗ đặt một UI Gradio nhỏ (`app/gradio_app.py`) nếu nhóm muốn demo đẹp hơn notebook.

- Bắt buộc theo đề chỉ là "notebook **hoặc** giao diện nhỏ". Notebook demo là bản chính; UI là phần thêm.
- UI chỉ gọi hàm trong `src/mmxai/`, không chứa logic riêng.
- Lý do chọn Gradio: `docs/decisions/0013-demo-notebook-va-gradio.md`.

## Chạy demo đường đi giải thích

`app/gradio_app.py`: nhập một câu truy vấn, xem top-k ảnh **val** và, với mỗi ảnh, đường đi
`truy vấn → khái niệm → contains → ảnh` đã tạo ra điểm đồ thị của ảnh đó.

```bash
uv sync --group ml --group graph --group demo
uv run python app/gradio_app.py --config configs/clip_graph_vg_lr1e4.yaml --seed 0 \
    --vg-data-dir data/visual_genome   # tuỳ chọn, chỉ để hiện thumbnail
```

Cần có trước: checkpoint `artifacts/models/visual_genome/clip_graph_vg_lr1e4/seed<k>/`
(`model.pt`, `selection.json`), đặc trưng và đồ thị trong `artifacts/`. Nạp mất khoảng 10 giây, mỗi
truy vấn khoảng 0.1 giây trên CPU laptop.

**Đọc bảng của mỗi ảnh như thế nào?** Mô hình một lớp là tuyến tính, nên điểm đồ thị `g` tách được
**chính xác** thành tổng các phần:

| Dòng | Ý nghĩa |
|---|---|
| `query → image itself` | phần của chính vectơ ảnh (`W_self x_i`), thường là phần lớn nhất |
| `query → bias` | phần của bias `b` |
| `query → <khái niệm> → contains → image` | phần đi qua một cạnh (3 dương + 3 âm mạnh nhất) |
| `N other edges` | tổng các cạnh không hiện |
| `total = g` | cộng cả cột lại ra đúng `g` |

Cột **Occlusion** là mức `g` giảm thật khi xoá riêng cạnh đó. Hai số khác nhau vì xoá cạnh còn đổi mẫu
số của phép trung bình và `‖h_i‖`.

- Chỉ tìm trên **val** (ADR-0009); `alpha` là giá trị huấn luyện đã chọn trên val, UI không cho đổi.
- Logic nằm ở `src/mmxai/explain/score_path.py` (tách điểm), `query_path.py` (tìm + giải thích),
  `path_view.py` (bảng); app chỉ gọi các hàm này.
