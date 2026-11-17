# G-Retriever bậc 2 trên Flickr30k: kế hoạch cài đặt

> Spec: `docs/superpowers/specs/2026-11-17-g-retriever-flickr30k-design.md`. Nhánh `feat/g-retriever-flickr30k`.
> Mỗi task một commit (Conventional Commits, kết thúc bằng `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`).
> Trước mỗi commit chạy `python -m uv run ruff check` và `python -m uv run pytest -q`.
> Có 2 test đã fail sẵn, không phải lỗi của plan: `tests/test_notebooks_match_src.py` (symlink trên Windows).

**Cách chạy:** 2 batch, chạy lần lượt.
- **B1** (T1 [M] + T2 [S]): Sonnet.
- **B2** (T3 [M] + T4 [S]): Sonnet.

Sau mỗi batch, Haiku review chỉ theo spec.

## Batch B1

### T1 [M]: cho phép dùng cạnh PMI được đếm trên split của dataset khác
- [ ] Thêm trường `concept_edges_split_file: str | None = None` vào ba dataclass. Đặt trường này **sau cùng**, vì có giá trị mặc định:
  - `GraphRerankConfig` (`src/mmxai/eval/graph_rerank.py`);
  - `PcstRerankConfig` (`src/mmxai/eval/pcst_rerank.py`);
  - `PcstSageConfig` (`src/mmxai/eval/pcst_sage_rerank.py`).
  Thêm một comment ngắn giải thích vì sao: Flickr30k mượn từ vựng và cạnh PMI của VG, theo ADR-0052.
- [ ] Sửa `load_concept_edges_for` trong `graph_rerank.py`:
  - nếu `config.concept_edges_split_file` khác `None`, so `meta["split_sha256"]` với `file_sha256(config.concept_edges_split_file)`;
  - ngược lại, so với `file_sha256(config.split_file)` như cũ;
  - giữ nguyên thông báo lỗi chứa "different split file" và giữ check `concept_keys`.
- [ ] Thêm test vào `tests/test_graph_rerank_pipeline.py`. Dùng lại helper `write_inputs` có sẵn; nếu helper chưa dựng cạnh PMI thì đọc cách các test hiện có tạo `concept_edges_dir` rồi làm theo. Viết hai test:
  1. Cạnh đếm trên split A, thí nghiệm dùng split B và đặt `concept_edges_split_file = A`: `load_inputs` nạp được và `inputs.concept_edges` khác `None`.
  2. Cùng tình huống nhưng đặt `concept_edges_split_file` là một split thứ ba: báo `ValueError` khớp "different split".
- [ ] Commit: `feat(rerank): allow concept edges counted on another dataset's split (ADR-0052)`.

### T2 [S]: config Flickr30k cho G-Retriever bậc 2 và đối chứng
- [ ] Tạo `configs/clip_pcst_sage_flickr30k.yaml` bằng cách chép `configs/clip_pcst_sage_vg.yaml`, rồi chỉ đổi những khoá sau:
  - `experiment: clip_pcst_sage_flickr30k`
  - `dataset: flickr30k`
  - `split_file: splits/flickr30k/split_seed0.json`
  - `features_dir: artifacts/features/flickr30k/ViT-B-32__laion2b_s34b_b79k`
  - `graph_dir: artifacts/graphs/flickr30k/contains_clip_top16_v1`
  - `cache_dir: artifacts/pcst_cache/clip_pcst_sage_flickr30k`
  - `model_dir: artifacts/models/flickr30k/clip_pcst_sage_flickr30k`
  - thêm `concept_edges_split_file: splits/visual_genome/split_seed0.json`, đặt ngay dưới `concept_edges_dir`. **Không** đổi `concept_edges_dir`, vì đó là cạnh PMI của VG.
  Giữ các comment quyết định của bản VG. Sửa comment đầu file thành "configs/clip_pcst_sage_vg.yaml on Flickr30k: only the dataset changes; PMI concept edges are borrowed from VG (ADR-0052)".
- [ ] Tạo `configs/clip_nograph_flickr30k_lr1e4.yaml` bằng cách chép `configs/clip_nograph_vg_lr1e4.yaml`, rồi chỉ đổi:
  - `experiment: clip_nograph_flickr30k_lr1e4`
  - `dataset: flickr30k`
  - `split_file: splits/flickr30k/split_seed0.json`
  - `features_dir: artifacts/features/flickr30k/ViT-B-32__laion2b_s34b_b79k`
  - `graph_dir: artifacts/graphs/flickr30k/contains_clip_top16_v1`
  - `model_dir: artifacts/models/flickr30k/clip_nograph_flickr30k_lr1e4`
  Comment đầu file: "configs/clip_nograph_vg_lr1e4.yaml on Flickr30k: same-config control for clip_pcst_sage_flickr30k (ADR-0052)".
- [ ] Tạo `tests/test_flickr30k_pcst_configs.py`, theo kiểu của `tests/test_flickr30k_configs.py`:
  - So hai cặp (VG, Flickr30k). Mọi khoá ngoài nhóm dataset (`experiment`, `dataset`, `split_file`, `features_dir`, `graph_dir`, `cache_dir`, `model_dir`, `output_dir`) phải bằng nhau.
  - Flickr30k chỉ được có thêm đúng một khoá: `concept_edges_split_file`, và khoá này chỉ xuất hiện ở bản pcst.
  - Các khoá đường dẫn dữ liệu chứa `flickr30k`.
  - `concept_edges_dir` của bản Flickr30k bằng của bản VG.
  - `concept_edges_split_file == "splits/visual_genome/split_seed0.json"`.
  - Cả hai config nạp được bằng `load_pcst_sage_config` và `load_graph_rerank_config`.
- [ ] Commit: `feat(configs): G-Retriever stage 2 and its same-config control on Flickr30k (ADR-0052)`.

## Batch B2

### T3 [M]: ADR-0052 và hướng dẫn chạy
- [ ] Tạo `docs/decisions/0052-g-retriever-tren-flickr30k-dung-lai-canh-pmi-cua-vg.md` theo template ADR-0001, viết tiếng Việt.
  - Trạng thái "Đề xuất", ngày 2026-11-17.
  - Nội dung lấy từ spec, mục 2–8: bối cảnh, quyết định (nhóm chọn ngày 2026-11-17), hai phương án bị loại, rò rỉ, hệ quả.
  - Ghi chú trùng số ADR-0050 (spec mục 8).
- [ ] Thêm dòng 0052 vào cuối bảng "Quyết định khi cài đặt" trong `docs/decisions/README.md`, bắt đầu bằng "(Đề xuất)".
- [ ] Thêm mục `## 11. G-Retriever bậc 2 trên Flickr30k (ADR-0052)` vào cuối `docs/guides/huan-luyen-va-xep-hang-lai.md`. Mục gồm:
  1. Ý tưởng, 3–4 câu.
  2. Bảng "cần có trước khi chạy". Mỗi dòng ghi thêm cách tạo:
     - đặc trưng Flickr30k;
     - đồ thị `artifacts/graphs/flickr30k/contains_clip_top16_v1`, dựng bằng `scripts/build_graph_from_vocabulary.py --config configs/graph_flickr30k_clip_topk.yaml`;
     - đồ thị VG `contains_clip_top16_v1`, nguồn từ vựng;
     - cạnh PMI VG `concept_cooccurrence_pmi10_top10_v1`.
  3. Bảng lệnh theo thứ tự. Lấy đúng tên script, cờ (`--seed`, `--estimate`) và cách gọi `uv run` từ bảng 10.7, rồi đổi config:
     - ước tính thời gian;
     - `prepare_pcst_cache`;
     - `train_pcst_sage` × 3 seed;
     - `train` đối chứng `clip_nograph_flickr30k_lr1e4` × 3 seed (cùng script với `clip_nograph_vg_lr1e4`; tìm script đó trong mục 4 của guide);
     - test, mỗi seed một lần.
  4. Ghi chú Colab:
     - symlink **cả** `artifacts/pcst_cache` lên Drive (bài học 2026-11-17: cache mất khi xoá runtime);
     - `drive.flush_and_unmount()` sau mỗi phiên.
  5. Thời gian ước tính (spec mục 7) và quy tắc dừng giống mục 10.8.
- [ ] Commit: `docs: ADR-0052 and guide for G-Retriever stage 2 on Flickr30k`.

### T4 [S]: nhật ký AI
- [ ] Thêm một dòng vào cuối bảng `docs/ai-usage-log.md`, giữ CRLF như các dòng khác:
  - ngày 2026-11-17, thành viên `Thuat Nguyen`;
  - công cụ: `Claude Code (Opus 5.5 lập spec/plan; Sonnet 5.5 subagent, Haiku 4.5 review)`;
  - phần việc: tóm tắt T1–T3 kèm đường dẫn file;
  - "Phần đã chỉnh sửa" ghi `(chưa)`; cột cuối ghi `Chưa`.
- [ ] Commit: `docs(ai-log): G-Retriever on Flickr30k`.
