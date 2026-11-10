# Hướng dẫn: chạy toàn bộ pipeline (CLIP thuần và CLIP + Graph) kèm giải thích

> **Trạng thái 2026-11-10:** code đã có test trên dữ liệu tổng hợp nhỏ (`tests/test_models_training.py`, `tests/test_explain_demo.py`), **chưa chạy trên dữ liệu thật**. Thiết kế mô hình: [ADR-0046](../decisions/0046-gnn-mot-lop-huan-luyen-cho-truy-van-anh-voi-canh-clip-top-k.md) (Đề xuất). Giải thích: [ADR-0016](../decisions/0016-giai-thich-trung-thuc-tu-co-che-xep-hang.md) (Occlusion trên chính mô hình).

## 1. Ý tưởng trong 1 phút

```
câu truy vấn ─► CLIP text ─► FAISS top 100 ảnh ─┬─► clip:        xếp theo cosine
                                                └─► clip_graph:  alpha·cosine + (1−alpha)·điểm đồ thị
                                                                  (GNN một lớp đã huấn luyện)
                                                         │
                                                         ▼
                                          ảnh top-1 + giải thích: cạnh khái niệm nào đẩy ảnh lên / kéo xuống
```

Chỉ có **hai script**, chạy cùng một kiểu lệnh:

| Script | Khi nào chạy | Làm gì | Thời gian |
|---|---|---|---|
| `scripts/setup_and_train.py` | **Một lần** | Kiểm tra dữ liệu → dựng đồ thị → huấn luyện GNN (mỗi seed) → chọn `alpha` trên val | Lâu (nên có GPU) |
| `scripts/run_pipeline.py` | **Bao nhiêu lần cũng được** | Truy vấn vài câu, in ảnh tìm được + giải thích ảnh top-1, mode `clip` hoặc `clip_graph` | Vài giây tới một phút (chủ yếu là nạp đặc trưng) |

`uv run --group ml --group graph` tự cài torch, open_clip, faiss, PyG ở lần đầu (đã khai báo trong `pyproject.toml`, ADR-0006). Sau lưng proxy có chứng chỉ riêng thì thêm `UV_NATIVE_TLS=1` ở đầu lệnh.

## 2. Chuẩn bị dữ liệu

`scripts/setup_and_train.py` **tự tải** các file annotation Visual Genome còn thiếu (danh sách, URL và sha256 ở `configs/data_vg.yaml`, code ở `mmxai.data.download.download_file`). Chỉ embedding CLIP là phải tải tay.

| Cần gì | Dùng cho | Ai tải | Đặt ở |
|---|---|---|---|
| Embedding CLIP của Visual Genome | Cả hai mode | **Tải tay**: `docs/guides/chuan-bi-embedding-clip.md` §2 (Kaggle) | `artifacts/features/visual_genome/ViT-B-32__laion2b_s34b_b79k/` |
| `objects.json` (khoảng 55 MB zip, 349 MB sau giải nén) | Dựng đồ thị (từ vựng khái niệm đếm trên train) | **Script tự tải**, kiểm sha256; sai hash thì xoá và dừng | `data/visual_genome/objects.json` |
| `region_descriptions.json` (khoảng 127 MB zip) | Để `run_pipeline.py` in **nội dung câu** truy vấn | **Script tự tải**; tải lỗi thì chỉ cảnh báo, vẫn train tiếp | `data/visual_genome/region_descriptions.json` |

File đã có thì không tải lại. Tải dở chừng không để lại file "trông như hoàn chỉnh" (tải vào tên tạm `*.part`).

## 3. Bước 1 – cài, dựng đồ thị, huấn luyện (một lần)

```bash
uv run --group ml --group graph python scripts/setup_and_train.py              # seed 0, 1, 2
uv run --group ml --group graph python scripts/setup_and_train.py --seeds 0    # chỉ 1 seed, nhanh hơn
```

**Thiết bị:** CPU là đủ. Đo trên CPU 14 core (không GPU): một bước train 0,14 giây, **khoảng 9 phút một epoch**, ước khoảng 45–90 phút một seed (early stopping thường dừng sau 5–10 epoch). Có GPU thì nhanh hơn nhiều; code tự chọn (`device: auto`). Lúc train cần khoảng **3–4 GB RAM** trống (riêng vector câu mô tả đã 1,1 GB).

Script chạy lần lượt, bước nào đã có kết quả thì **bỏ qua**, nên chạy lại sau khi bị ngắt là an toàn:

| Bước | Làm gì | Code | Kết quả |
|---|---|---|---|
| 1 | Tải `objects.json`, `region_descriptions.json` nếu thiếu; kiểm tra có embedding (thiếu thì dừng và chỉ tài liệu tải) | `mmxai.data.download.download_file` | `data/visual_genome/*.json` |
| 2 | Dựng đồ thị cạnh CLIP top-16 (`configs/graph_vg_clip_topk.yaml`) | `mmxai.graph.pipeline.run_build_graph` | `artifacts/graphs/visual_genome/contains_clip_top16_v1/` |
| 3 | Huấn luyện GNN **chỉ trên ảnh train**: mỗi câu mô tả so với ảnh đúng + 15 ảnh sai lấy từ top 100 của CLIP (InfoNCE), early stopping theo MRR@100 trên val | `mmxai.models.training.train_gnn` | `artifacts/checkpoints/gnn_vg_seed<k>/model.pt`, `meta.json` |
| 4 | Xếp hạng lại val với mọi `alpha` trong `alpha_grid`, chọn `alpha` có MRR cao nhất | `mmxai.rerank.evaluate.run_rerank_evaluation` | `artifacts/checkpoints/gnn_vg_seed<k>/chosen_alpha.json`, `results/runs/<thời gian>_gnn_vg_seed<k>/` |

### Xem log huấn luyện trên Weights & Biases (tuỳ chọn, ADR-0047)

```bash
uv run --group tracking wandb login                     # một lần; hoặc đặt WANDB_API_KEY
uv run --group ml --group graph --group tracking python scripts/setup_and_train.py
```

- Mỗi seed là một run tên `gnn_vg_seed<k>` trong project `mmxai-gnn` (đổi ở `wandb_project` trong `configs/gnn_vg.yaml`; `WANDB_ENTITY` chọn team).
- Log: `train/loss` và `val/MRR@100` mỗi epoch, `train/batch_loss` mỗi 50 bước, `best_epoch` ở cuối.
- Không thêm `--group tracking`, đặt `WANDB_MODE=disabled`, hoặc mất mạng: train vẫn chạy bình thường, chỉ in một dòng cảnh báo.
- Số trong báo cáo vẫn lấy từ `results/` và `meta.json`, không lấy từ W&B.

Siêu tham số (`n_hard`, `temperature`, `learning_rate`, ...) nằm ở `configs/gnn_vg.yaml`. Chỉ chỉnh dựa trên **val**.

## 4. Bước 2 – chạy pipeline và xem giải thích (nhiều lần)

```bash
# 5 câu mô tả ngẫu nhiên của ảnh test, mỗi mode
uv run --group ml --group graph python scripts/run_pipeline.py --mode clip
uv run --group ml --group graph python scripts/run_pipeline.py --mode clip_graph

# Nhiều câu hơn, mẫu khác
uv run --group ml --group graph python scripts/run_pipeline.py --mode clip_graph --num-queries 10 --sample-seed 1

# Câu tự gõ (dữ liệu thật; không có đáp án nên không in thứ hạng đúng)
uv run --group ml --group graph python scripts/run_pipeline.py --mode clip_graph --text "a dog on a sofa"
```

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--mode` | (bắt buộc) | `clip`: chỉ cosine. `clip_graph`: cosine + điểm đồ thị với `alpha` đã chọn trên val |
| `--num-queries` | `5` | Số câu mô tả lấy ngẫu nhiên từ split làm truy vấn |
| `--sample-seed` | `0` | Đổi số này để lấy mẫu câu khác |
| `--text` | – | Câu tự gõ; lặp lại để truyền nhiều câu. Có `--text` thì không lấy mẫu từ split |
| `--split` | `test` | `val` hoặc `test`: kho ảnh để tìm |
| `--seed` | `0` | Dùng mô hình đã huấn luyện với seed nào |
| `--show-top` | `5` | Số ảnh in ra cho mỗi truy vấn |
| `--top-m` | `3` | Số quan hệ tối đa mỗi phía (đẩy lên / kéo xuống) trong giải thích |

**Đọc kết quả** (ví dụ dạng in ra, số chỉ minh hoạ):

```
=== Query 1 (clip_graph): a brown dog lying on a couch
  1. image 2345678  score 0.3120  <- correct
  2. image 2399001  score 0.3011
  ...
  First correct image at rank 1
  Why image 2345678 is first:
    Pushed the image up:
      1. dog -[in]-> image 2345678 (+0.02)
      2. sofa -[in]-> image 2345678 (+0.01)
    Pulled the image down:
      (none)
```

- `<- correct`: ảnh có cùng câu mô tả với truy vấn (quy tắc đáp án của ADR-0043). `FAILED` = không có ảnh đúng trong top 100.
- **Giải thích ở mode `clip_graph`** (Occlusion, `mmxai.explain.demo.explain_image`): bỏ **từng** cạnh khái niệm → ảnh của ảnh top-1, chạy lại chính mô hình, đo điểm đồ thị giảm bao nhiêu. Số trong ngoặc là mức ảnh hưởng lên **điểm cuối**, đã nhân `(1 − alpha)`.
- **Giải thích ở mode `clip`**: in `The graph was not used (alpha = 1)`. CLIP thuần không có đồ thị nên không có quan hệ nào để chỉ ra.
- Đây là giải thích cấp độ 1–2. Fidelity (Fid+/Fid−) là cấp độ 3, chưa đưa vào script.

## 5. Lưu ý chống rò rỉ (PDF §6, ADR-0009)

- Mặc định `--split test` chỉ để **xem** kết quả cho báo cáo. Sau khi xem test thì **không đổi** config, `alpha` hay mô hình. Đang còn chỉnh thì dùng `--split val`.
- Muốn có **số liệu** test (Recall/MRR, 3 seed) cho báo cáo thì chạy `scripts/evaluate_rerank.py --split test`, mỗi seed một lần (xem docstring của script).
- Mô hình chỉ học trên ảnh train; test `test_training_graph_has_no_val_or_test_node` kiểm điều này.

## 6. Lỗi thường gặp

| Thông báo | Nguyên nhân | Cách sửa |
|---|---|---|
| `Missing CLIP features in ...` | Chưa tải embedding | Mục 2, dòng 1 |
| `objects.json has sha256 ...` | File tải về khác bản đã ghi trong `data_cards/visual_genome.md` | Chạy lại; vẫn sai thì báo nhóm (nguồn có thể đã đổi) |
| `URLError` / lỗi chứng chỉ khi tải | Mạng hoặc proxy | Thử lại; hoặc tải tay theo `docs/guides/xay-do-thi.md` §2 rồi chạy lại |
| `... model.pt not found` | Chưa huấn luyện seed đó | Chạy bước 1 với `--seeds <k>` |
| `... chosen_alpha.json not found` | Chưa chọn `alpha` trên val | Chạy lại bước 1 (bước 4 sẽ chạy bù) |
| `invalid peer certificate: UnknownIssuer` | Proxy có chứng chỉ riêng | Thêm `UV_NATIVE_TLS=1` trước lệnh |
