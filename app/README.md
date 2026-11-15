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
    --vg-data-dir data/visual_genome   # cần image_data.json để hiện ảnh
```

Cần có trước: checkpoint `artifacts/models/visual_genome/clip_graph_vg_lr1e4/seed<k>/`
(`model.pt`, `selection.json`), đặc trưng và đồ thị trong `artifacts/`. Nạp mất khoảng 10 giây, mỗi
truy vấn khoảng 0.1 giây trên CPU laptop.

**Màn hình gồm gì?**

| Phần | Nội dung |
|---|---|
| Gallery | top-k ảnh val. Cần `data/visual_genome/image_data.json`; có ảnh trong `images/` thì dùng file trên máy, không có thì dùng URL gốc (cần mạng) |
| Đồ thị (phần chính) | 3 cột: truy vấn → node khái niệm → node ảnh. Cạnh `contains` được tô là cạnh Occlusion chọn: **xanh** = xoá thì `g` giảm (cạnh đẩy ảnh lên), **đỏ** = xoá thì `g` tăng (kéo xuống), dày hơn = tác động lớn hơn. Cung xám từ truy vấn tới ảnh là phần "ảnh tự đi" (`W_self x_i`), số ghi trong ô ảnh. Đường chấm truy vấn → khái niệm không phải cạnh đồ thị |
| "Numbers behind the graph" | bảng số của từng ảnh (mục thu gọn), giải thích ở dưới |

Logic vẽ nằm ở `src/mmxai/explain/path_graph.py` (`build_path_graph` chọn node/cạnh, `draw_path_graph` vẽ);
không có bước tìm đường nào, mọi cạnh được tô đều là cạnh Occlusion đã chọn (ADR-0016).

**Đọc bảng của mỗi ảnh như thế nào?** Bảng có hai cột số, trả lời hai câu hỏi khác nhau:

| Cột | Câu hỏi | Cách tính |
|---|---|---|
| **Occlusion** | Xoá riêng cạnh này thì `g` giảm bao nhiêu? | Xoá cạnh, chạy lại mô hình (can thiệp thật). `+` = cạnh đẩy ảnh lên, `−` = kéo xuống |
| **part of g** | Khi giữ **mọi** cạnh, cạnh này chiếm bao nhiêu trong `g`? | Mô hình một lớp tuyến tính nên `g` tách **chính xác** thành tổng các phần (`score_path.py`) |

| Dòng | Ý nghĩa |
|---|---|
| `image itself (W_self x_i)` | phần của chính vectơ ảnh, thường là phần lớn nhất |
| `bias b` | phần của bias |
| `<khái niệm> → contains → image` | một cạnh của đồ thị; hiện 3 cạnh Occlusion dương lớn nhất + 3 âm mạnh nhất, giống phân tích 5+5 cấp độ 1 |
| `N other edges` | tổng "part of g" của các cạnh không hiện |
| `total = g` | cộng cột "part of g" lại ra đúng `g` |

Lưu ý khi đọc:

- Hai cột có thể **khác nhau, kể cả khác dấu**: xoá một cạnh còn đổi mẫu số của phép trung bình và
  `‖h_i‖`. Muốn biết cạnh "giúp" hay "hại" ảnh thì xem cột Occlusion.
- Chỉ cạnh `contains` (khái niệm → ảnh) là cạnh thật của đồ thị. Truy vấn **không** phải một node: nó đi
  vào `g` qua tích vô hướng với `q`, nên không có "cạnh truy vấn → khái niệm".

- Chỉ tìm trên **val** (ADR-0009); `alpha` là giá trị huấn luyện đã chọn trên val, UI không cho đổi.
- Logic nằm ở `src/mmxai/explain/score_path.py` (tách điểm), `query_path.py` (tìm + giải thích),
  `path_view.py` (bảng); app chỉ gọi các hàm này.
