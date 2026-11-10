# Hướng dẫn: dựng đồ thị ảnh – khái niệm cho Visual Genome

> **Trạng thái 2026-11-07:** bản đầu, chỉ có cạnh `contains`. Đã chạy thật một lần trên toàn bộ Visual Genome (số liệu ở mục 4). Thiết kế: [ADR-0045](../decisions/0045-do-thi-anh-khai-niem-canh-contains-cho-visual-genome.md) (Đề xuất).

## 1. Ý tưởng trong 1 phút

Đồ thị có **hai loại đỉnh** và **một loại cạnh**:

```
[ảnh 1] ──contains──► dog.n.01 ◄──contains── [ảnh 2]
   │                                            │
   └──contains──► grass.n.01      sofa.n.01 ◄───┘
```

- **Đỉnh ảnh**: mọi ảnh của split (train, val, test). Vector của đỉnh là embedding CLIP của ảnh, lấy từ cache đã có.
- **Đỉnh khái niệm**: một synset WordNet mà Visual Genome gắn sẵn vào đồ vật. Ví dụ `dog.n.01` gom cả "dog" lẫn "puppy". Vector của đỉnh là CLIP text của câu `"a photo of a dog"`.
- **Cạnh `contains`**: ảnh có ít nhất một đồ vật thuộc khái niệm đó.

Mỗi khái niệm chỉ có **một** đỉnh, nên hai ảnh cùng có chó được nối với nhau qua `dog.n.01`. GNN sẽ truyền thông tin theo những đường như vậy.

**Hai quy tắc chống rò rỉ dữ liệu (PDF §6):**

1. Danh sách khái niệm được **đếm trên train**: giữ synset xuất hiện trong ≥ 5 ảnh train. Synset chỉ có ở ảnh test thì không có đỉnh (test: `test_concepts_are_counted_on_train_images_only`).
1b. Đồ vật **thiếu synset** (VG để trường `synsets` rỗng) được tra synset theo tên, bằng **từ điển học trên train**: tên "couch" ở train hầu hết được gắn `sofa.n.01`, nên "couch" thiếu synset cũng thành `sofa.n.01`. Từ điển chỉ đếm trên train; ảnh val/test chỉ tra, không góp vào (test: `test_name_dictionary_is_learned_on_train_images_only`).
2. Không có cạnh quan hệ (`on`, `holding`...), vì bộ truy vấn ADR-0044 được sinh từ chính quan hệ của ảnh test.

## 2. Chuẩn bị

| Cần gì | Lấy ở đâu | Đặt ở |
|---|---|---|
| Thư viện | `uv sync --group ml --group graph` (torch, open_clip, PyG; đã khai báo sẵn trong `pyproject.toml`) | `.venv/` |
| `objects.json` (v1.4, 349 MB sau giải nén) | [Trang tải Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/api.html), file `objects.json.zip` | `data/visual_genome/objects.json` |
| Embedding CLIP của ảnh VG | `docs/guides/chuan-bi-embedding-clip.md` §2 | `artifacts/features/visual_genome/ViT-B-32__laion2b_s34b_b79k/` |
| Split | Đã commit | `splits/visual_genome/split_seed0.json` |

Tải `objects.json` bằng lệnh:

```bash
mkdir -p data/visual_genome
curl -L -o data/visual_genome/objects.json.zip \
  https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/objects.json.zip
unzip data/visual_genome/objects.json.zip -d data/visual_genome/
sha256sum data/visual_genome/objects.json
# phải ra 94238b6187c0ea1f97acc1fdaec106bd3dee12bb0f81b23dc8b07e9173514985
```

Embedding của ảnh **chưa cần** để dựng đồ thị. Nó chỉ cần khi đổi đồ thị sang `Data` của PyG (mục 5).

## 3. Chạy

```bash
uv run python scripts/build_graph.py --config configs/graph_vg.yaml
```

- Lần đầu chạy, open_clip tự tải trọng số CLIP từ Hugging Face (~600 MB).
- **Nếu gặp lỗi `401 Unauthorized` từ Hugging Face**: token HF lưu trên máy bị hỏng hoặc hết hạn. Chạy lại ở chế độ ẩn danh (model này công khai):
  ```bash
  HF_HUB_DISABLE_IMPLICIT_TOKEN=1 uv run python scripts/build_graph.py --config configs/graph_vg.yaml
  ```
- Không cần GPU: trên CPU mất khoảng **3–4 phút** và tối đa **~4,3 GB RAM** (đo ngày 2026-11-07, laptop 11 GB RAM).
- Script **không ghi đè**. Muốn dựng lại thì tự xoá `artifacts/graphs/visual_genome/contains_v1/`.

Output (gitignored, không commit):

| File | Nội dung |
|---|---|
| `artifacts/graphs/visual_genome/contains_v1/graph.npz` | `image_ids`, `concept_keys` (synset), `concept_names` (chữ đưa vào CLIP), `concept_features` (số khái niệm × 512), `edge_image`, `edge_concept` |
| `artifacts/graphs/visual_genome/contains_v1/meta.json` | Số đếm, sha256 của `objects.json` và split, commit git, config đã chạy |

Cạnh lưu dạng **số dòng**: cạnh thứ k nối `image_ids[edge_image[k]]` với `concept_keys[edge_concept[k]]`.

## 4. Kết quả lần chạy 2026-11-07

| Mục | Số lượng |
|---|---|
| Đỉnh ảnh | 108.077 |
| Đỉnh khái niệm (synset, ≥ 5 ảnh train) | 3.880 |
| Cạnh `contains` | 1.719.645 (trung vị 16 cạnh mỗi ảnh, nhiều nhất 50) |
| Synset của đồ vật lấy từ đâu | 80,4% có sẵn trong annotation · 15,1% tra theo tên · 4,5% không tra được (bỏ) |
| Từ điển tên → synset (học trên train) | 40.195 tên; trên train, synset được chọn khớp synset thật ở 99,8% đồ vật |
| Bản ghi hỏng bị bỏ | 18 (tên rỗng) |
| Khái niệm nhiều cạnh nhất | shirt, man, tree, wall, sky |
| Ảnh không có cạnh nào | train 1.643 · val 92 · test 70 (~1,7%) |

**Vì sao vẫn còn ảnh không có cạnh:** 1.793 ảnh trong Visual Genome không có đồ vật nào được gán nhãn, nên không có gì để nối. Với những ảnh này, điểm chỉ còn đến từ CLIP.

**Vì sao cần bước tra theo tên:** 13.320 ảnh có đồ vật bình thường ("cabinet", "couch", "table") nhưng Visual Genome để trường `synsets` rỗng cho **mọi** đồ vật trong ảnh. Lần chạy đầu không tra theo tên, nên ~14% ảnh không có cạnh nào (trên val: 717/5.000). Có bước tra, con số này giảm còn ~1,7% (val: 92/5.000).

Lưu ý nhỏ: 134 khái niệm có tên trùng với khái niệm khác (ví dụ hai nghĩa của "bank"), nên có cùng vector CLIP text.

## 5. Dùng đồ thị trong code (cho người viết GNN)

```python
from mmxai.features.cache import load_image_features, load_meta
from mmxai.graph.contains import load_contains_graph
from mmxai.graph.pyg import to_pyg_data

graph, concept_names, concept_features, graph_meta = load_contains_graph(
    "artifacts/graphs/visual_genome/contains_v1"
)
features_dir = "artifacts/features/visual_genome/ViT-B-32__laion2b_s34b_b79k"
image_features, feature_image_ids = load_image_features(features_dir)
data = to_pyg_data(
    graph, concept_features, image_features, feature_image_ids,
    graph_meta, load_meta(features_dir),
)
# data.x          (108077 + 3880) × 512: ảnh trước, khái niệm sau
# data.edge_index 2 × (2 × 1719645): mỗi cạnh contains lưu cả hai chiều
# data.node_type  0 = ảnh, 1 = khái niệm
```

Đỉnh ảnh thứ i trong `data` ứng với ảnh `graph.image_ids[i]`. Đỉnh khái niệm thứ j ứng với dòng `num_images + j`.

`to_pyg_data` bắt buộc nhận hai `meta.json` (của đồ thị và của cache ảnh) để kiểm hai bên dùng **cùng một model CLIP**. Hai model khác nhau có thể cùng 512 chiều nhưng khác không gian, nên nếu không kiểm thì code vẫn chạy mà kết quả vô nghĩa. Id ảnh bị trùng trong cache cũng báo lỗi.

⚠️ **Hai điều bắt buộc cho người viết GNN** (chi tiết: ADR-0045, mục Hệ quả):

1. `data` chứa **cả ảnh val và test**. Khi huấn luyện, chỉ được dùng đồ thị con gồm ảnh **train** và các khái niệm. Ảnh val/test chỉ được gắn vào lúc suy luận. Nếu không, GNN đã "nhìn" ảnh test trong lúc học (PDF §6).
2. Cạnh `contains` của ảnh test lấy từ annotation, mà nhãn đúng của bộ truy vấn cũng lấy từ annotation đó. Vì vậy phần thực nghiệm phải có ablation thay/xáo/bỏ cạnh của ảnh test.

## 6. Code nằm ở đâu

| File | Vai trò | Riêng / chung |
|---|---|---|
| `src/mmxai/graph/visual_genome.py` | `build_name_to_synset`: từ điển tên → synset học trên train; `image_concepts_from_scene_graphs`: ảnh → tập synset; `concept_text`: synset → chữ | **Riêng VG** |
| `src/mmxai/graph/contains.py` | `select_concepts` (đếm trên train), `build_contains_graph`, lưu/nạp `graph.npz` | Chung |
| `src/mmxai/graph/pipeline.py` | `run_build_graph`: nối các bước, mã hoá khái niệm bằng CLIP, ghi `meta.json` | Chung |
| `src/mmxai/graph/pyg.py` | `to_pyg_data`: đổi sang `torch_geometric.data.Data` | Chung |
| `scripts/build_graph.py`, `configs/graph_vg.yaml` | CLI và cấu hình | |
| `tests/test_graph_*.py` | Test trên dữ liệu giả nhỏ, không cần `data/` | |

Đọc dữ liệu thô dùng lại `build_scene_graphs` trong `src/mmxai/data/scene_graph.py` (ADR-0044), chỉ truyền `objects.json`.

Thêm Flickr30k sau này: viết `src/mmxai/graph/flickr30k.py` trả về cùng dạng `{image_id: tập khái niệm}`, thêm một nhánh trong `run_build_graph` và một file config. GNN và rerank không phải sửa.
