# app/ — giao diện nhỏ (tuỳ chọn)

**Là gì?** Chỗ đặt một UI Gradio nhỏ (`app/gradio_app.py`) nếu nhóm muốn demo đẹp hơn notebook.

- Bắt buộc theo đề chỉ là "notebook **hoặc** giao diện nhỏ". Notebook demo là bản chính; UI là phần thêm.
- UI chỉ gọi hàm trong `src/mmxai/`, không chứa logic riêng.
- Lý do chọn Gradio: `docs/decisions/0013-demo-notebook-va-gradio.md`.

## Chạy demo đường đi giải thích

`app/gradio_app.py`: nhập một câu truy vấn, xem top-k ảnh **val** và giải thích của từng ảnh.
Ba nhóm nút chọn cho biết đang demo phần nào của đề (PDF §5):

| Nút | Lựa chọn | Hiện khi |
|---|---|---|
| **Level** | 1, 2, 3 | luôn hiện |
| **Method** | `Plain CLIP` (baseline) · `CLIP + Graph (GraphSAGE, ours)` · `G-Retriever (PCST, stage 2)` | G-Retriever từ cấp 2 |
| **Explanation** | `Basic: paths (Occlusion)` · `Quantitative: IG (m = 10) + fidelity` | ẩn với CLIP thuần; định lượng chỉ ở cấp 3 với CLIP + Graph |

| Kết hợp | Màn hình hiện gì | Code |
|---|---|---|
| CLIP thuần | top-k theo cosine, không có giải thích (không có đồ thị) | `query_path.search_clip` |
| CLIP + Graph, cơ bản | đồ thị đường đi `truy vấn → khái niệm → contains → ảnh` + bảng tách điểm (giữ như trước) | `query_path`, `path_graph`, `path_view` |
| CLIP + Graph, định lượng | với mỗi ảnh: m cạnh của explainer chính (IG, m = 10, **đọc từ `configs/explain_vg.yaml`**, đã chọn trên val), rồi Fid+ và tụt hạng khi xoá chúng so với xoá ngẫu nhiên, Fid−, sparsity, validity; dòng đầu là trung bình k ảnh | `query_quantitative`, `quantitative_view` (dùng lại `order_relations`, `evaluate_explanation` của `scripts/evaluate_explanations.py`) |
| G-Retriever, cơ bản | đồ thị con `S_q` của truy vấn (khái niệm được thưởng + cầu nối PCST), rồi với mỗi ảnh: các khái niệm của ảnh nằm trong `S_q` (đúng tập điểm đồ thị dùng) và điểm mất khi bỏ từng khái niệm | `retriever_path`, `retriever_view` (dùng lại `PcstSageRanker`, `explain_candidate`) |

Không có lựa chọn nào đổi `alpha`, explainer hay m: tất cả đã chốt trên val.

### Setup trên máy mới (vừa clone repo)

Cần: Linux/macOS/WSL, khoảng 8 GB RAM, khoảng 10 GB đĩa trống (thư viện ~6 GB, embedding 1,3 GB, dữ liệu ~0,4 GB), có mạng. **Không cần GPU.** Mọi lệnh chạy từ thư
mục gốc của repo.

**Bước 1. Cài `uv` và thư viện** (khoảng 5–10 phút, torch nặng nhất)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh        # cài uv, mở terminal mới sau đó
uv sync --group ml --group graph --group demo           # torch, open_clip, faiss, PyG, gradio
```

Mạng công ty báo lỗi chứng chỉ (`invalid peer certificate: UnknownIssuer`) thì thêm `--system-certs`
vào lệnh `uv sync`.

**Bước 2. Dữ liệu Visual Genome** (`data/`, gitignored; khoảng 370 MB)

```bash
mkdir -p data/visual_genome && cd data/visual_genome
curl -L -O https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip
curl -L -O https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/objects.json.zip
unzip -q image_data.json.zip && unzip -q objects.json.zip && rm ./*.zip
cd ../..
```

| File | Dùng để |
|---|---|
| `image_data.json` | URL của từng ảnh, để gallery hiện ảnh |
| `objects.json` | đếm từ vựng khái niệm trên ảnh train khi dựng đồ thị (bước 4) |

Không cần tải 15 GB ảnh: gallery dùng URL gốc ở `cs.stanford.edu` (trình duyệt cần mạng). Muốn chạy
offline thì giải nén ảnh vào `data/visual_genome/images/VG_100K/` và `VG_100K_2/`
(`docs/guides/chuan-bi-embedding-clip.md`); app tự dùng file trên máy khi có.

**Bước 3. Embedding CLIP** (tải sẵn, khoảng 1,3 GB)

Tải 3 file `image_features.npz`, `text_features.npz`, `meta.json` của phiên bản **visual_genome** trong
tab Output của notebook Kaggle (hướng dẫn chi tiết: `docs/guides/chuan-bi-embedding-clip.md` §2), đặt vào:

```text
artifacts/features/visual_genome/ViT-B-32__laion2b_s34b_b79k/
```

**Bước 4. Dựng đồ thị** (khoảng 3–4 phút CPU, tối đa ~4,3 GB RAM; lần đầu tải trọng số CLIP ~600 MB)

```bash
uv run python scripts/build_graph.py --config configs/graph_vg_clip_topk.yaml
# → artifacts/graphs/visual_genome/contains_clip_top16_v1/
```

**Bước 5. Checkpoint mô hình đồ thị** — chọn một trong hai:

- **Lấy từ nhóm** (nhanh): xin thư mục `artifacts/models/visual_genome/clip_graph_vg_lr1e4/` đã huấn
  luyện, đặt đúng đường dẫn đó. Mỗi seed cần `seed<k>/model.pt` và `seed<k>/selection.json`.
- **Tự huấn luyện** (CPU được, chỉ dùng train/val: khoảng 8 phút chuẩn bị + khoảng 10 phút mỗi seed, 4–5 GB RAM):

  ```bash
  uv run python scripts/train_graph_rerank.py --config configs/clip_graph_vg_lr1e4.yaml --seeds 0
  ```

  Mô hình tự huấn luyện có thể khác chút với bản nhóm báo cáo; demo vẫn đúng vì nó kiểm checkpoint khớp
  với `selection.json` của chính nó.

**Bước 6. Chạy demo**

```bash
./demo.sh
```

`demo.sh` gói sẵn lệnh đầy đủ (`HF_HUB_OFFLINE=1 uv run --system-certs --all-groups python
app/gradio_app.py --config configs/clip_graph_vg_lr1e4.yaml --seed 0 --vg-data-dir data/visual_genome`).
Tham số thêm vào sẽ ghi đè mặc định, ví dụ `./demo.sh --seed 1`.

Mở <http://127.0.0.1:7860>, gõ truy vấn (ví dụ `man riding a horse`), bấm **Search**. Nạp mất khoảng
10 giây, mỗi truy vấn khoảng 0,1 giây trên CPU laptop.

Mạng công ty: nếu app đứng im lúc nạp (treo khi gọi Hugging Face qua proxy) mà máy đã có trọng số
CLIP trong cache, thêm `HF_HUB_OFFLINE=1` trước lệnh.

**Bước 7 (tuỳ chọn). G-Retriever cho cấp 2.** App chỉ nạp G-Retriever lần đầu bạn chọn nó; thiếu
file thì app báo lỗi trong ô kết quả, các chế độ khác vẫn chạy. Cần hai thứ của
`configs/clip_pcst_sage_vg.yaml` (đổi bằng `--retriever-config`):

- Cạnh đồng xuất hiện PMI (khoảng 1 phút):
  `uv run python scripts/build_concept_edges.py --config configs/concept_edges_vg_cooccurrence.yaml`
- Checkpoint giai đoạn 2 `artifacts/models/visual_genome/clip_pcst_sage_vg/seed<k>/` (`model.pt`,
  `selection.json`): xin từ nhóm, hoặc tự chạy `scripts/prepare_pcst_cache.py` rồi
  `scripts/train_pcst_sage.py` (xem docstring hai script; PCST trên train mất khoảng một giờ).

**Kiểm tra nhanh** (thiếu gì thì app báo lỗi ngay lúc nạp):

```bash
ls data/visual_genome/                                                   # image_data.json  objects.json
ls artifacts/features/visual_genome/ViT-B-32__laion2b_s34b_b79k/         # 3 file
ls artifacts/graphs/visual_genome/contains_clip_top16_v1/                # graph.npz  meta.json
ls artifacts/models/visual_genome/clip_graph_vg_lr1e4/seed0/             # model.pt  selection.json
```

| Lỗi khi nạp | Nguyên nhân |
|---|---|
| `FileNotFoundError ... selection.json` | thiếu bước 5 |
| `... was built with a different split file` | đồ thị dựng với split khác: xoá thư mục đồ thị, làm lại bước 4 |
| `the analysis does not reproduce training` | checkpoint và đồ thị/embedding không cùng bộ: dựng lại đồ thị hoặc huấn luyện lại |
| Gallery trống, có dòng "No thumbnails" | thiếu `data/visual_genome/image_data.json` hoặc quên `--vg-data-dir` |

**Màn hình gồm gì?** (chế độ CLIP + Graph, giải thích cơ bản)

| Phần | Nội dung |
|---|---|
| Gallery | top-k ảnh val. Cần `data/visual_genome/image_data.json`; có ảnh trong `images/` thì dùng file trên máy, không có thì dùng URL gốc (cần mạng) |
| Đồ thị (phần chính) | 3 cột: truy vấn → node khái niệm → node ảnh. Cạnh `contains` được tô là cạnh Occlusion chọn: **xanh** = xoá thì `g` giảm (cạnh đẩy ảnh lên), **đỏ** = xoá thì `g` tăng (kéo xuống), dày hơn = tác động lớn hơn. Cung xám từ truy vấn tới ảnh là phần "ảnh tự đi" (`W_self x_i`), số ghi trong ô ảnh. Đường chấm truy vấn → khái niệm không phải cạnh đồ thị |
| "Numbers behind the result" | bảng số của từng ảnh (mục thu gọn), giải thích ở dưới cho chế độ cơ bản của CLIP + Graph; các chế độ khác có đoạn "How to read" ở đầu bảng |

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
