# Chạy lại thực nghiệm

File này chỉ ghi cần cài gì và chạy lệnh gì cho từng cấp độ. Mọi lệnh chạy từ thư mục gốc của repo.

**Máy cần:** Linux / macOS / WSL, khoảng 8 GB RAM, khoảng 10 GB đĩa trống, có mạng. **Không cần GPU.**

| Cấp độ | Cài | Dữ liệu |
|---|---|---|
| 1 | `uv sync --group ml --group graph` | Embedding Visual Genome, `objects.json`, `image_data.json` |
| 2 | như cấp độ 1 | thêm embedding Flickr30k |
| 3 | như cấp độ 1 | thêm ảnh MMKG (script tự tải) |
| Demo | `uv sync --group ml --group graph --group demo` | như cấp độ 1 |

Các cấp độ cộng dồn: cấp độ 2 và 3 cần các bước của cấp độ 1 chạy trước.

> Test của mỗi experiment chỉ được chạy **một lần** cho mỗi seed. Các script chấm test và đánh giá
> từ chối chạy lại nếu đã có kết quả. Số đã báo cáo nằm sẵn trong `results/tables/`.

---

## 0. Cài đặt chung

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # cài uv (macOS: brew install uv), mở terminal mới
uv sync --all-groups                                 # hoặc chỉ các group của cấp độ cần chạy (bảng trên)
uv run pytest                                        # kiểm tra cài đặt: phải pass hết
```

Mạng công ty báo lỗi `invalid peer certificate: UnknownIssuer`: thêm `--system-certs` vào lệnh `uv sync`.

### Embedding CLIP (tải sẵn, khoảng 1,3 GB mỗi bộ)

1. Mở <https://www.kaggle.com/code/qphulong/prepare-clip-embeddings-for-mmkg-xai/> → **View Versions**.
2. Có 2 phiên bản, mỗi phiên bản là một bộ dữ liệu (`visual_genome`, `flickr30k`). Vào tab **Output**, tải 3 file.
3. Đặt vào:

```text
artifacts/features/visual_genome/ViT-B-32__laion2b_s34b_b79k/{image_features.npz, text_features.npz, meta.json}
artifacts/features/flickr30k/ViT-B-32__laion2b_s34b_b79k/{image_features.npz, text_features.npz, meta.json}
```

Muốn tự trích từ ảnh gốc thay vì tải: xem [`docs/guides/chuan-bi-embedding-clip.md`](docs/guides/chuan-bi-embedding-clip.md) mục 3–4.

### Nhãn Visual Genome (khoảng 370 MB)

```bash
mkdir -p data/visual_genome && cd data/visual_genome
curl -L -O https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip
curl -L -O https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/objects.json.zip
unzip -q image_data.json.zip && unzip -q objects.json.zip && rm ./*.zip
cd ../..
```

File split (`splits/`) và bộ truy vấn (`queries/`) đã có trong repo, không cần tạo.

---

## 1. Cấp độ 1: CLIP, CLIP + Graph, ablation, giải thích (Visual Genome)

```bash
# 1. Baseline CLIP thuần
for s in 0 1 2; do uv run python scripts/evaluate_clip.py --config configs/clip_vg.yaml --seed $s; done

# 2. Dựng đồ thị (khoảng 4 phút), cạnh đồng xuất hiện, đồ thị xáo cạnh cho ablation
uv run python scripts/build_graph.py --config configs/graph_vg_clip_topk.yaml
uv run python scripts/build_concept_edges.py --config configs/concept_edges_vg_cooccurrence.yaml
uv run python scripts/shuffle_graph.py --config configs/graph_vg_clip_topk_shuffled.yaml

# 3. Huấn luyện + chọn alpha trên val (khoảng 10 phút/seed), rồi chấm test
for cfg in clip_graph_vg_lr1e4 clip_nograph_vg clip_cooc_graph_vg_lr1e4 clip_graph_vg_lr1e4_shuffled; do
  uv run python scripts/train_graph_rerank.py --config configs/$cfg.yaml --seeds 0 1 2 --threads 4
  for s in 0 1 2; do uv run python scripts/evaluate_graph_rerank.py --config configs/$cfg.yaml --seed $s; done
done

# 4. Tổng hợp bảng → results/tables/main.csv
uv run python scripts/summarize_runs.py

# 5. Giải thích + phân tích 5 truy vấn thành công / 5 thất bại (trên val)
uv run python scripts/explain.py --config configs/clip_graph_vg_lr1e4.yaml --seed 0 \
    --split val --num-edges 3 --text-data-dir data/visual_genome
```

Tuỳ chọn, đánh giá mở rộng (nDCG, ảnh → văn bản, nhóm truy vấn): lặp lại bước 3 với các config `configs/*_eval2.yaml`, rồi chạy `uv run python scripts/summarize_runs.py --experiment-suffix _eval2 --kind main` (đổi `main` thành `groups`, `image_to_text` để ra các bảng còn lại).

---

## 2. Cấp độ 2: Flickr30k, G-Retriever, bộ 38 truy vấn

Cần chạy xong bước 2 của cấp độ 1 trước.

Cấp độ 2 chạy trên **2 tập công khai**: Visual Genome và Flickr30k.

| Yêu cầu | Visual Genome | Flickr30k |
|---|---|---|
| CLIP, CLIP + Graph, ablation | cấp độ 1 (mục 1) | mục 2a |
| G-Retriever + đối chứng không đồ thị | mục 2b | mục 2b |
| Bộ 38 truy vấn trực tiếp / gián tiếp | mục 2c | – |

### 2a. CLIP + Graph trên Flickr30k (tập thứ hai)

```bash
for s in 0 1 2; do uv run python scripts/evaluate_clip.py --config configs/clip_flickr30k.yaml --seed $s; done

uv run python scripts/build_graph_from_vocabulary.py --config configs/graph_flickr30k_clip_topk.yaml
uv run python scripts/shuffle_graph.py --config configs/graph_flickr30k_clip_topk_shuffled.yaml

for cfg in clip_graph_flickr30k_lr1e4 clip_nograph_flickr30k clip_graph_flickr30k_lr1e4_shuffled; do
  uv run python scripts/train_graph_rerank.py --config configs/$cfg.yaml --seeds 0 1 2 --threads 4
  for s in 0 1 2; do uv run python scripts/evaluate_graph_rerank.py --config configs/$cfg.yaml --seed $s; done
done

uv run python scripts/summarize_runs.py --split test --output results/tables/level1_flickr30k_test.csv
```

### 2b. G-Retriever (PCST + GraphSAGE) trên cả hai tập

Bước cache PCST chậm, khoảng 1 giờ CPU mỗi tập. Mỗi tập chạy G-Retriever và một đối chứng không đồ thị cùng cấu hình huấn luyện.

```bash
# Visual Genome → results/tables/g_retriever_test.csv
uv run python scripts/prepare_pcst_cache.py --config configs/clip_pcst_sage_vg.yaml --seed 0
for s in 0 1 2; do
  uv run python scripts/train_pcst_sage.py --config configs/clip_pcst_sage_vg.yaml --seed $s
  uv run python scripts/evaluate_pcst_sage.py --config configs/clip_pcst_sage_vg.yaml --seed $s
done
uv run python scripts/train_graph_rerank.py --config configs/clip_nograph_vg_lr1e4.yaml --seeds 0 1 2 --threads 4
for s in 0 1 2; do uv run python scripts/evaluate_graph_rerank.py --config configs/clip_nograph_vg_lr1e4.yaml --seed $s; done
uv run python scripts/summarize_runs.py --split test --output results/tables/g_retriever_test.csv

# Flickr30k (cần đồ thị Flickr30k ở 2a) → results/tables/g_retriever_flickr30k_test.csv
uv run python scripts/prepare_pcst_cache.py --config configs/clip_pcst_sage_flickr30k.yaml --seed 0
for s in 0 1 2; do
  uv run python scripts/train_pcst_sage.py --config configs/clip_pcst_sage_flickr30k.yaml --seed $s
  uv run python scripts/evaluate_pcst_sage.py --config configs/clip_pcst_sage_flickr30k.yaml --seed $s
done
uv run python scripts/train_graph_rerank.py --config configs/clip_nograph_flickr30k_lr1e4.yaml --seeds 0 1 2 --threads 4
for s in 0 1 2; do uv run python scripts/evaluate_graph_rerank.py --config configs/clip_nograph_flickr30k_lr1e4.yaml --seed $s; done
uv run python scripts/summarize_runs.py --split test --output results/tables/g_retriever_flickr30k_test.csv
```

`summarize_runs.py` gom mọi run trong `results/runs/`. Muốn ra đúng bảng của từng tập, thêm `--runs-dir <thư mục chỉ chứa run của tập đó>`.

### 2c. Bộ 38 truy vấn trực tiếp / gián tiếp

Bộ truy vấn `queries/visual_genome_kg_test.jsonl` đã có sẵn. Cần checkpoint của 2b.

```bash
for s in 0 1 2; do
  uv run python scripts/evaluate_query_benchmark.py --config configs/eval_query_benchmark_vg_kg.yaml --seed $s
  for m in clip_pcst_sage_vg clip_nograph_vg_lr1e4; do
    uv run python scripts/evaluate_query_benchmark_models.py \
        --config configs/eval_query_benchmark_vg_kg_$m.yaml --seed $s
  done
done

uv run python scripts/summarize_query_benchmark.py     # → results/tables/kg_query_benchmark.csv
```

---

## 3. Cấp độ 3: giải thích định lượng, MMKG tiếng Việt

Cần các checkpoint của cấp độ 1, bước 3.

```bash
# Fidelity / validity / sparsity trên Visual Genome → results/tables/xai_level3*.csv
uv run python scripts/evaluate_explanations.py --config configs/explain_vg.yaml

# MMKG tiếng Việt: tải ảnh (cần mạng), trích embedding, đánh giá
uv run python scripts/download_mmkg_images.py --collect-config configs/mmkg_vi_collect.yaml \
    --build-config configs/mmkg_vi_build.yaml
uv run python scripts/evaluate_mmkg_transfer.py features --config configs/mmkg_vi_transfer.yaml
uv run python scripts/evaluate_mmkg_transfer.py evaluate --config configs/mmkg_vi_transfer.yaml   # → mmkg_transfer*.csv
uv run python scripts/evaluate_mmkg_explanations.py --config configs/explain_mmkg_vi.yaml \
    --transfer-config configs/mmkg_vi_transfer.yaml                                              # → xai_mmkg.csv
```

Bảng MMKG (`mmkg/*.csv`) và bộ truy vấn (`queries/mmkg_vi/queries.jsonl`) đã có sẵn, không cần thu thập lại.

---

## 4. Demo

Cần: nhãn Visual Genome, embedding Visual Genome, đồ thị (cấp độ 1 bước 2) và checkpoint
`artifacts/models/visual_genome/clip_graph_vg_lr1e4/seed0/` (cấp độ 1 bước 3, hoặc xin nhóm).

```bash
uv sync --group ml --group graph --group demo
./demo.sh
```

Mở <http://127.0.0.1:7860>, gõ truy vấn (ví dụ `man riding a horse`), bấm **Search**.
Chế độ G-Retriever trong demo cần thêm checkpoint của 2b. Hướng dẫn chi tiết và lỗi thường gặp: [`app/README.md`](app/README.md).

---

## Kết quả ra đâu

| Đường dẫn | Nội dung |
|---|---|
| `artifacts/` | Embedding, đồ thị, checkpoint (`model.pt`, `selection.json`) |
| `results/runs/`, `results/query_benchmark/runs/` | Mỗi lần chạy một thư mục: `config.yaml`, `metrics.json`, `env.json` |
| `results/tables/` | Bảng mean ± std qua 3 seed, nguồn số liệu của báo cáo |

## Lỗi thường gặp

| Lỗi | Cách xử lý |
|---|---|
| `invalid peer certificate: UnknownIssuer` | thêm `--system-certs` vào `uv sync` / `uv run` |
| macOS: `OMP: Error #15` | trỏ `libomp.dylib` của faiss sang bản của torch trong `.venv`, xem [`docs/guides/huan-luyen-va-xep-hang-lai.md`](docs/guides/huan-luyen-va-xep-hang-lai.md) mục 4 |
| Script treo khi tải trọng số CLIP sau proxy | máy đã có cache thì thêm `HF_HUB_OFFLINE=1` trước lệnh |
| Script báo đã có kết quả, từ chối chạy | test đã chạy rồi; xoá thư mục run tương ứng nếu muốn chạy lại trên máy mình |
