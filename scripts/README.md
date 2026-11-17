# scripts/ — điểm chạy dòng lệnh (CLI)

**Là gì?** Các file chạy trực tiếp bằng `uv run python scripts/<tên>.py --config configs/<file>.yaml`. Script chỉ **đọc tham số và gọi hàm** trong `src/mmxai/`; logic thật nằm trong package để test được.

Danh sách dự kiến (chưa viết), theo thứ tự chạy pipeline:

| Script | Làm gì | Gọi vào |
|---|---|---|
| `scripts/make_splits.py` (đã có) | Tạo file split train/val/test cố định, ghi vào `splits/`. Cần embedding Visual Genome (ADR-0042) | `mmxai.data.splits` |
| `scripts/extract_features.py` (đã có) | Trích đặc trưng CLIP **một lần**, lưu cache. Hướng dẫn: `docs/guides/chuan-bi-embedding-clip.md` | `mmxai.features.pipeline` |
| `scripts/sync_notebook_code.py` (đã có) | Chép code và config vào notebook Kaggle sau khi sửa | (chỉ đọc file) |
| `scripts/evaluate_clip.py` (đã có) | Baseline CLIP thuần: dựng FAISS trên ảnh của tập đánh giá, đo Recall@1/5/10 + MRR@100 (ADR-0043). Cần `uv sync --group ml` | `mmxai.eval.clip_baseline` |
| `scripts/build_index.py` (đã có) | Lưu vector store CLIP thuần: một FAISS index cho mỗi tập (`val`, `test`) vào `artifacts/vector_store/` | `mmxai.index.vector_store` |
| `scripts/generate_queries.py` (đã có) | Sinh cặp truy vấn trực tiếp/gián tiếp ứng viên từ scene graph Visual Genome + manifest để người kiểm tra (ADR-0044). Hướng dẫn: `docs/query_benchmark.md` | `mmxai.queries.generator` |
| `scripts/build_query_benchmark.py` (đã có) | Dựng `queries/visual_genome_test.jsonl` chỉ từ các cặp đã `verified` trong manifest | `mmxai.queries.validator` |
| `scripts/evaluate_query_benchmark.py` (đã có) | CLIP thuần trên bộ truy vấn đã kiểm tra: Recall@1/5/10/100 + MRR@100 tách direct/indirect. Cần `uv sync --group ml` | `mmxai.eval.query_benchmark` |
| `scripts/generate_kg_queries.py` (đã có) | Sinh cặp truy vấn trực tiếp/gián tiếp ứng viên qua khái niệm ConceptNet (hoạt động, đồ vật) + manifest để người kiểm tra (ADR-0050) | `mmxai.queries.kg_generator` |
| `scripts/build_kg_query_benchmark.py` (đã có) | Dựng `queries/visual_genome_kg_test.jsonl` chỉ từ các cặp KG đã `verified` trong manifest | `mmxai.queries.validator` |
| `scripts/summarize_runs.py` (đã có) | Gom `results/runs/*` thành `results/tables/main.csv` (mean ± std, ≥3 seed, bỏ run có code chưa commit) | `mmxai.eval.summary` |
| `scripts/build_graph.py` (đã có) | Dựng đồ thị ảnh – khái niệm, chỉ cạnh `contains`; từ vựng khái niệm đếm trên train (ADR-0045). Hướng dẫn: `docs/guides/xay-do-thi.md`. Cần `uv sync --group ml --group graph` | `mmxai.graph.pipeline` |
| `scripts/shuffle_graph.py` (đã có) | Xáo cạnh `contains` của một đồ thị đã dựng, trong từng split, giữ bậc của mỗi ảnh và mỗi khái niệm; ghi thành thư mục đồ thị mới (ablation ADR-0046 mục 6) | `mmxai.graph.shuffle` |
| `scripts/build_concept_edges.py` | Dựng cạnh khái niệm → khái niệm từ đồng xuất hiện (PMI) trên annotation **train**, dùng khái niệm của đồ thị `contains` (ADR-0048) | `mmxai.graph.cooccurrence` |
| `scripts/train_graph_rerank.py` (đã có) | Huấn luyện GraphSAGE (một lớp, hoặc hai lớp có cạnh đồng xuất hiện) trên train, early stopping và chọn `alpha` trên **val**, lưu mô hình (ADR-0046, ADR-0047). Hướng dẫn: `docs/guides/huan-luyen-va-xep-hang-lai.md` | `mmxai.eval.graph_rerank`, `mmxai.models`, `mmxai.rerank` |
| `scripts/evaluate_graph_rerank.py` (đã có) | Chấm **test một lần** mỗi seed bằng mô hình và `alpha` đã lưu; không chọn gì, từ chối chạy lại cùng seed (ADR-0047) | `mmxai.eval.graph_rerank` |
| `scripts/select_pcst_rerank.py` | Chọn `prize_k`, `edge_cost`, `alpha` của bộ xếp hạng lại PCST trên **val** (không huấn luyện), lưu `selection.json` (ADR-0051). Hướng dẫn: `docs/guides/huan-luyen-va-xep-hang-lai.md` mục 9. Cần `uv sync --group ml --group graph` | `mmxai.eval.pcst_rerank`, `mmxai.graph.pcst`, `mmxai.models.subgraph_scorer` |
| `scripts/evaluate_pcst_rerank.py` | Chấm **test một lần** mỗi seed bằng lựa chọn đã lưu; không chọn gì, từ chối chạy lại cùng seed (ADR-0051, ADR-0009) | `mmxai.eval.pcst_rerank` |
| `scripts/inspect_query_groups.py` (đã có) | Kiểm tra nhóm truy vấn trên **val** (spec mục 5.3; chỉ val, từ chối `test`): in số lượng và tỷ lệ mỗi nhóm của cả hai cách chia (độ đặc hiệu, loại quan hệ) rồi 30 câu ngẫu nhiên (seed 0) mỗi nhóm quan hệ để đọc bằng mắt. Dùng config `_eval2` của CLIP (vd `configs/clip_vg_eval2.yaml`); không ghi file | `mmxai.eval.extended`, `mmxai.eval.query_groups` |
| `scripts/explain.py` (đã có) | Phân tích 5 truy vấn thành công + 5 thất bại trên **val** (chỉ val, từ chối `test`): thứ hạng ảnh đúng đầu tiên theo CLIP và CLIP + Graph (`alpha` đã lưu khi huấn luyện), các cạnh khái niệm → ảnh đẩy ảnh lên/kéo xuống tính bằng Occlusion trên chính mô hình đã huấn luyện, kiểm tra xoá top `--num-edges` cạnh hỗ trợ và kiểm tra bỏ mọi cạnh vào ảnh (điểm đồ thị, thứ hạng trước/sau). Dừng nếu MRR val tính lại khác số `selection.json` đã lưu. Ghi `analysis.json` + `analysis.md` vào `results/runs/<thời điểm>_<experiment>_explain_seed<k>/`. Cấp độ 3: xem `scripts/evaluate_explanations.py` | `mmxai.explain.query_analysis`, `mmxai.explain.ranking_model`, `mmxai.eval.query_ranks` |
| `scripts/evaluate_explanations.py` (đã có) | **Cấp độ 3**: giải thích ảnh hạng 1 của ≥ 10 truy vấn bốc ngẫu nhiên × 3 seed trên mô hình `clip_graph_vg_lr1e4` bằng Occlusion, IG, GNNExplainer; đo Fid+/Fid−, tụt hạng, robust fidelity, đối chứng ngẫu nhiên, sparsity, validity; mean ± std qua seed + Wilcoxon. Val trước (đề xuất explainer và `m`), test một lần. Ghi `results/runs/<thời điểm>_<experiment>_<split>/`. Quy trình: `docs/research/xai-method-proposal.md` mục 7d | `mmxai.explain.evaluation`, `mmxai.explain.sage_scoring`, `mmxai.explain.fidelity` |
