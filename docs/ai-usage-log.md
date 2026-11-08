# Nhật ký sử dụng công cụ AI

**Vì sao có file này?** Đề bài (PDF §9) cho phép dùng mã nguồn mở và công cụ AI, nhưng nhóm phải
**khai báo trong báo cáo**, ghi rõ **phần đã chỉnh sửa**, và **mỗi thành viên tự giải thích được toàn
bộ nội dung khi vấn đáp**. File này là nguồn để viết mục "Khai báo sử dụng AI" trong báo cáo cuối kỳ.

## Cách ghi

- Mỗi phần việc đáng kể có AI hỗ trợ (viết/sửa code, viết tài liệu, phân tích kết quả, tìm bài báo) = **một dòng**.
- Có thể dùng skill `/log-ai-usage` trong Claude Code để thêm dòng tự động.
- Cột **Đã tự kiểm tra/hiểu chưa** chỉ do **chính thành viên** điền `Rồi`, sau khi đã đọc lại, chạy
  test và tự giải thích được phần đó. AI luôn ghi `Chưa`.
- Không xoá dòng cũ; nếu sau này viết lại phần đó thì thêm dòng mới ghi rõ.

| Cột | Ý nghĩa |
|---|---|
| Ngày | Ngày làm, dạng YYYY-MM-DD |
| Thành viên | Người chịu trách nhiệm phần việc (người sẽ trả lời khi vấn đáp) |
| Công cụ | Tên công cụ + model, ví dụ `Claude Code (Opus)`, kèm subagent/skill nếu có |
| Phần việc AI hỗ trợ | AI đã làm gì, file nào (đường dẫn) |
| Phần đã chỉnh sửa | Thành viên đã tự sửa/viết lại phần nào so với bản AI đề xuất |
| Đã tự kiểm tra/hiểu chưa | `Chưa` / `Rồi` – chỉ thành viên tự điền `Rồi` |

## Nhật ký

| Ngày | Thành viên | Công cụ | Phần việc AI hỗ trợ | Phần đã chỉnh sửa | Đã tự kiểm tra/hiểu chưa |
|---|---|---|---|---|---|
| 2026-11-01 | (trưởng nhóm) | Claude Code (Opus) | Dựng cấu hình công cụ AI: `CLAUDE.md`, `.claude/` (settings, hooks, rules, agents, skills), `.mcp.json`, ADR-0020 đến ADR-0027, file này | (chưa) | Chưa |
| 2026-11-04 | qphulong | Claude Code (Opus 5.5), skill `superpowers:brainstorming`, subagent builder check | Viết bản thiết kế `docs/superpowers/specs/2026-11-04-clip-embeddings-design.md` và hướng dẫn `docs/guides/chuan-bi-embedding-clip.md`; so sánh bản sao Flickr30k trên Kaggle | (chưa) | Chưa |
| 2026-11-04 | qphulong | Claude Code (Opus 5.5), skill `superpowers:executing-plans`, subagent `leakage-auditor` và code reviewer | Viết code trích embedding CLIP và test: `src/mmxai/utils/config.py`, `src/mmxai/data/{records,flickr30k,visual_genome}.py`, `src/mmxai/features/{cache,verify,clip_encoder,pipeline}.py`, `scripts/extract_features.py`, `scripts/sync_notebook_code.py`, `notebooks/10_prepare_clip_embeddings.ipynb`; ADR-0040, ADR-0041; data card | (chưa) | Chưa |
| 2026-11-04 | qphulong | Claude Code (Opus 5.5) | Kiểm tra embedding tải về từ Kaggle (`verify_features`, CRC, `code_hash`, vector trùng, truy vấn thử trên mẫu); cập nhật kết quả lần chạy vào `docs/guides/chuan-bi-embedding-clip.md`, `data_cards/flickr30k.md`, `data_cards/visual_genome.md` | (chưa) | Chưa |
| 2026-11-05 | FluteInTheRain | Claude Code (Opus 5.5), skill `superpowers:brainstorming`, `/new-adr`, subagent `leakage-auditor` | Tạo split cố định cho Visual Genome: ADR-0042, `src/mmxai/data/splits.py`, `scripts/make_splits.py`, `configs/splits_visual_genome.yaml`, `tests/test_splits.py`; cập nhật `data_cards/visual_genome.md`, `scripts/README.md`, `configs/README.md` | (chưa) | Chưa |
| 2026-11-05 | FluteInTheRain | Claude Code (Opus 5.5), `/new-adr` | Baseline CLIP thuần: ADR-0043, `src/mmxai/eval/{metrics,relevance,clip_baseline}.py`, `src/mmxai/index/image_index.py`, `src/mmxai/utils/{yaml_config,runs}.py`, `scripts/evaluate_clip.py`, `configs/clip_vg.yaml` và test tương ứng | (chưa) | Chưa |
| 2026-11-05 | FluteInTheRain | Claude Code (Opus 5.5) | Vector store theo split và bảng tổng hợp: `src/mmxai/index/vector_store.py` (+ save/load trong `image_index.py`), `src/mmxai/eval/summary.py`, `scripts/build_index.py`, `scripts/summarize_runs.py`, `configs/vector_store_vg.yaml`, test tương ứng; tạo `results/tables/main.csv` | (chưa) | Chưa |
| 2026-11-06 | Thuat Nguyen | Claude Code (Opus 5.5), subagent general-purpose | Bộ truy vấn đánh giá Nhiệm vụ 2 từ scene graph Visual Genome: `src/mmxai/data/scene_graph.py`, `src/mmxai/queries/{schema,generator,validator}.py`, `src/mmxai/eval/query_benchmark.py`, `scripts/{generate_queries,build_query_benchmark,evaluate_query_benchmark}.py`, `configs/{query_benchmark_vg,query_benchmark_vg_pilot,eval_query_benchmark_vg}.yaml`, test tương ứng, ADR-0044 (Đề xuất), `docs/query_benchmark.md`. Chưa chạy trên dữ liệu thật (thiếu annotation) | (chưa) | Chưa |
| 2026-11-06 | Thuat Nguyen | Claude Code (Opus 5.5, brainstorm), subagent general-purpose | Sửa quy tắc sinh truy vấn của ADR-0044 sau các pilot val và cài đặt: whitelist vị từ + template (`predicate_templates`), lọc dạng chuỗi và vị từ hành động, đáp án theo synset, lọc tên chung chung/bộ phận cơ thể/tên không phải từ thường: `src/mmxai/data/scene_graph.py`, `src/mmxai/queries/{generator,validator}.py`, `configs/query_benchmark_vg{,_pilot}.yaml`, test tương ứng, ADR-0044, `docs/query_benchmark.md` | (chưa) | Chưa |
| 2026-11-07 | thiennp3 | Claude Code (Opus 5.5), subagent `paper-researcher` (web search; MCP `arxiv` không kết nối được) | Khảo sát tài liệu cho phần xAI và soạn `docs/research/xai-method-proposal.md`: 4 phần việc của module xAI theo ADR-0016, công thức fidelity/validity/sparsity (GraphFramEx, Robust Fidelity), câu hỏi nghiên cứu, baseline explainer, giao diện cần nhận và lưu ý gửi nhóm Graph | (chưa) | Chưa |
| 2026-11-09 | thiennp3 | Claude Code (Opus 5.5) | Bước 1 của module xAI: hàm xoá cạnh dùng chung cho fidelity, Occlusion và ablation, `src/mmxai/explain/intervention.py` (`mark_remaining_edges`) và `tests/test_explain_intervention.py`; sửa ADR-0016, `CLAUDE.md`, `.claude/rules/explanations.md` theo hướng "explainer tính trên chính mô hình" | (chưa) | Chưa |
| 2026-11-09 | thiennp3 | Claude Code (Opus 5.5) | Cấp độ 1 của module xAI: tìm các cạnh có thể ảnh hưởng điểm của một ảnh sau K lớp GNN, `src/mmxai/explain/subgraph.py` (`edges_reaching_node`) và `tests/test_explain_subgraph.py` | (chưa) | Chưa |
| 2026-11-09 | thiennp3 | Claude Code (Opus 5.5) | Cấp độ 1 của module xAI: explainer Occlusion (xoá từng cạnh trong đồ thị con, đo điểm giảm), `src/mmxai/explain/occlusion.py` và `tests/test_explain_occlusion.py` | (chưa) | Chưa |
