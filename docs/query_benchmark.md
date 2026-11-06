# Bộ truy vấn đánh giá từ scene graph Visual Genome (Nhiệm vụ 2)

> **Trạng thái 2026-11-06:** code và test đã có; **chưa chạy trên dữ liệu thật** vì máy tạo code chưa có `objects.json` / `relationships.json`. Hiện có **0 truy vấn ứng viên thật, 0 truy vấn đã kiểm tra**. Quyết định thiết kế: [ADR-0044](decisions/0044-bo-truy-van-danh-gia-tu-scene-graph-visual-genome.md) (Đề xuất).

## 1. Ý tưởng trong 30 giây

Đề (PDF §5, Nhiệm vụ 2) cần ≥ 20 truy vấn, có nhóm **gián tiếp** phải khai thác quan hệ. Ta không tự bịa truy vấn, mà lấy từ **scene graph** có sẵn của Visual Genome:

1. Tìm trong một ảnh **hai quan hệ thật** nối nhau qua **một đối tượng chung** (cùng `object_id`).
2. Viết **truy vấn trực tiếp**: gọi tên mọi đối tượng.
3. Viết **truy vấn gián tiếp**: giấu tên đối tượng ở giữa (node cầu), chỉ mô tả nó qua quan hệ.
4. Tìm **mọi ảnh test** có cùng chuỗi quan hệ → danh sách đáp án ứng viên.
5. **Hai thành viên** xem ảnh, xác nhận đáp án. Chỉ cặp đã xác nhận mới vào bộ chính thức.

Ví dụ **tổng hợp** (dữ liệu giả trong `tests/scene_graph_fixtures.py`, **không phải** ảnh Visual Genome):

| Bước | Nội dung |
|---|---|
| Scene graph | `person(1) -holding-> frisbee(2)`, `dog(3) -looks at-> frisbee(2)` |
| Trực tiếp | "A person is holding a frisbee, and a dog looks at the frisbee." |
| Gián tiếp | "A person is holding the thing that a dog looks at." (giấu `frisbee`) |
| Đáp án ứng viên | ảnh giả 100 và 200 (cùng chuỗi); ảnh 300 không có (con chó nhìn quả bóng) |

Dòng JSONL thật mà generator sinh ra từ ví dụ giả này (`scripts/generate_queries.py` chạy trên fixture):

```json
{"qid": "vg-test-001-i", "text": "A person is holding the thing that a dog looks at.", "type": "indirect", "dataset": "visual_genome", "relevant": ["100", "200"], "path_hint": "person - holding - frisbee; dog - looks at - frisbee", "author": "generator-template-v1", "pair_id": "vg-test-001", "source_image_id": "100", "relation_evidence": [{"relationship_id": 1, "subject_id": 1, "subject_name": "person", "predicate": "holding", "object_id": 2, "object_name": "frisbee", "raw": ["Person", "HOLDING", "frisbee"]}, {"relationship_id": 2, "subject_id": 3, "subject_name": "dog", "predicate": "looks at", "object_id": 2, "object_name": "frisbee", "raw": ["dog", "looks_at", "frisbee"]}], "chain_object_ids": [1, 2, 3], "hop_count": 2, "split": "test", "verification_status": "candidate", "reasoning_type": "shared_object", "hidden_names": ["frisbee"]}
```

## 2. Chuẩn bị dữ liệu

Cần hai file annotation của Visual Genome (trang chính thức, xem `data/README.md`):

| File | Bản | Đặt ở |
|---|---|---|
| `objects.json` | v1.4 (hoặc v1.2, ghi rõ vào `data_cards/visual_genome.md`) | `data/visual_genome/objects.json` |
| `relationships.json` | cùng bản với `objects.json` | `data/visual_genome/relationships.json` |

- Tải gói objects và relationships ở trang tải của Visual Genome (file nén), giải nén vào `data/visual_genome/`. Tên file khác hoặc đặt chỗ khác thì sửa `objects_file`, `relationships_file`, `data_dir` trong config. Ghi bản và ngày tải vào data card.
- `relationships.json` khoảng 700 MB, nạp cần vài GB RAM.
- Không cần ảnh để **sinh** truy vấn; cần ảnh (hoặc trình xem ảnh theo `image_id`) để **kiểm tra** bằng mắt.
- Đánh giá CLIP cần thêm embedding (`docs/guides/chuan-bi-embedding-clip.md` §2) và `uv sync --group ml`.

## 3. Quy trình và lệnh

| # | Việc | Lệnh | Ra file |
|---|---|---|---|
| 1 | Pilot 5 cặp trên **val**, đọc thử câu, chốt quy tắc trong config | `uv run python scripts/generate_queries.py --config configs/query_benchmark_vg_pilot.yaml` | `results/runs/query_pilot_val/` (gitignored) |
| 2 | Sinh 40 cặp ứng viên trên **test** (quy tắc đã chốt, không đổi nữa) | `uv run python scripts/generate_queries.py --config configs/query_benchmark_vg.yaml` | `queries/visual_genome_candidates.jsonl`, `queries/query_review_manifest.csv` |
| 3 | Thành viên A kiểm tra, thành viên B kiểm lại (mục 4) | sửa CSV bằng Excel/LibreOffice | `queries/query_review_manifest.csv` |
| 4 | Dựng bộ chính thức từ các cặp `verified` (≥ 10 cặp) | `uv run python scripts/build_query_benchmark.py --config configs/query_benchmark_vg.yaml` | `queries/visual_genome_test.jsonl` |
| 5 | Commit bộ chính thức (sau đó hook đóng băng file) | `git add queries/ && git commit` | |
| 6 | Đánh giá CLIP thuần, mỗi seed một lần | `uv run python scripts/evaluate_query_benchmark.py --config configs/eval_query_benchmark_vg.yaml --seed <k>` | `results/query_benchmark/runs/<ngày>_clip_vg_query_benchmark_seed<k>/` |

Mọi script từ chối ghi đè file đã có. Muốn sinh lại thì tự xoá file (và ghi lý do trong PR).

Khi chạy bước 1–2, script in số bản ghi bị bỏ (`dropped_records`) và số chuỗi bị loại theo lý do (`rejected_chains`). Ghi các số này vào báo cáo.

## 4. Kiểm tra bởi người (manifest)

Mỗi dòng của `query_review_manifest.csv` là **một cặp** (trực tiếp + gián tiếp dùng chung đáp án).

| Cột | Ai điền | Ý nghĩa |
|---|---|---|
| `direct_text`, `indirect_text`, `path_hint`, `relationship_ids`, `hidden_names` | generator | Câu và bằng chứng |
| `candidate_relevant`, `num_candidates` | generator | Ảnh test có cùng chuỗi trong annotation |
| `verification_status` | người | `candidate` → `reviewed` → `verified`, hoặc `rejected` |
| `reviewer` | thành viên A | Tên người xem ảnh |
| `verified_relevant` | thành viên A | Id ảnh **thật sự** đúng (cách nhau bởi dấu cách), chỉ ảnh thuộc test; được bỏ ảnh sai, thêm ảnh test đúng mà annotation thiếu |
| `checker` | thành viên B | Tên người kiểm lại, **phải khác** `reviewer` |
| `edited_direct_text`, `edited_indirect_text` | tuỳ chọn | Sửa câu cho tự nhiên hơn, **không đổi nghĩa**; câu gián tiếp không được nhắc tên trong `hidden_names` |
| `notes` | tuỳ chọn | Lý do bỏ, nhận xét |

Quy tắc khi xem ảnh: câu gián tiếp phải đúng với ảnh **và** chỉ trả lời được khi lần theo quan hệ. Câu sai nghĩa, mơ hồ, hoặc annotation sai → `rejected`. Không chọn hay bỏ cặp dựa trên việc CLIP trả lời đúng hay sai (đó là "debug-đến-khi-đẹp" trên tập test).

`build_query_benchmark.py` chỉ lấy dòng `verified`; dòng `verified` mà thiếu người kiểm, trùng người, hay có ảnh ngoài test thì script **dừng và báo lỗi**, không lặng lẽ bỏ qua.

## 5. Quy tắc sinh truy vấn

Code: `src/mmxai/queries/generator.py`. Mọi ngưỡng nằm trong `configs/query_benchmark_vg.yaml`.

| Quy tắc | Vì sao |
|---|---|
| Chuỗi gồm đúng 2 quan hệ (`min_hops = max_hops = 2`), đi qua các `object_id` khác nhau | Gián tiếp thật sự cần ≥ 2 quan hệ; 3 bước cho câu lồng nhau khó kiểm tra |
| Tên mỗi đối tượng trong chuỗi là duy nhất trong ảnh | "the dog" phải chỉ đúng một con chó; đồ thị theo id khớp với điều người đọc thấy |
| Bỏ tên chung chung (`generic_names`) và vị từ tầm thường (`trivial_predicates`) | "it", "thing", "of" không kiểm được bằng mắt |
| Câu gián tiếp không chứa tên node cầu | Nếu không, đó chỉ là câu trực tiếp viết lại |
| Bỏ câu trùng, tối đa 1 cặp mỗi ảnh | Bộ truy vấn trải trên nhiều ảnh |
| Bỏ chuỗi có > 25 ảnh test khớp (`max_candidate_images`) | Quá chung, không kiểm tay được |
| Thứ tự ảnh xáo bằng `seed`, duyệt tất định | Chạy lại ra đúng cùng file |

Ba dạng chuỗi 2 bước (`reasoning_type` của câu gián tiếp), ví dụ tổng hợp:

| Dạng | Chuỗi | Câu gián tiếp |
|---|---|---|
| `shared_object` | A -p-> X <-q- B | "A person is holding the thing that a dog looks at." |
| `path` | A -p-> X -q-> B | "A man is riding the thing that is on a beach." |
| `shared_subject` | A <-p- X -q-> B | "The thing that is wearing a hat is holding an umbrella." |

## 6. Định dạng file

Giữ đúng các trường bắt buộc của `queries/README.md` (`qid`, `text`, `type`, `dataset`, `relevant`, `path_hint`, `author`) và thêm:

| Trường | Ý nghĩa |
|---|---|
| `pair_id` | Cặp trực tiếp/gián tiếp; hai truy vấn cùng cặp có cùng đáp án |
| `source_image_id` | Ảnh sinh ra chuỗi |
| `relation_evidence` | Các quan hệ thật đã dùng (id quan hệ, id đối tượng, tên chuẩn hoá, chuỗi gốc `raw`) |
| `chain_object_ids` | Thứ tự các đối tượng trong chuỗi |
| `hidden_names` | Tên bị giấu trong câu gián tiếp (rỗng với câu trực tiếp) |
| `reasoning_type`, `hop_count` | Dạng chuỗi, số quan hệ |
| `split` | Luôn `test` với bộ chính thức |
| `verification_status` | `candidate` / `reviewed` / `verified` / `rejected` |
| `reviewed_by`, `checked_by` | Chỉ có trong bộ chính thức |

`path_hint`, `relation_evidence`, `hidden_names` chỉ để phân tích và kiểm tra. **Mô hình không được đọc chúng**: giao diện `QueryRanker` (`src/mmxai/eval/query_benchmark.py`) chỉ nhận văn bản truy vấn và embedding.

## 7. Đánh giá

- Embedding truy vấn bằng đúng CLIP của embedding ảnh (`ViT-B-32`, `laion2b_s34b_b79k`, cùng tokenizer qua `encode_texts`); script kiểm `meta.json` của bộ đặc trưng.
- Kho tìm kiếm: 5.000 ảnh test, FAISS `IndexFlatIP`, top 100 (ADR-0043).
- Metric: Recall@1/5/10/100 và MRR@100, cho **all**, **direct**, **indirect**; kèm thứ hạng từng truy vấn (`per_query`) để sau này so sánh cặp với CLIP + Graph.
- CLIP + Graph: viết một lớp có hàm `rank(texts, query_embeddings, top_k)` trả về id ảnh, rồi gọi `evaluate_benchmark`.
- Thiếu bộ chính thức hoặc thiếu embedding → script dừng với thông báo rõ ràng.

## 8. Chống rò rỉ dữ liệu (PDF §6)

- Generator chỉ nạp scene graph của **một** tập; đáp án chỉ tìm trong tập đó. Có test chứng minh ảnh val/train không bao giờ là đáp án của truy vấn test.
- Pilot chạy trên **val**, file ra nằm ngoài `queries/`.
- Bộ chính thức **chỉ để đánh giá**: không chọn `alpha`, ngưỡng, template hay quy tắc lọc dựa trên kết quả của nó.
- **Lưu ý cho người làm CLIP + Graph:** truy vấn gián tiếp sinh từ chính annotation quan hệ của ảnh test. Nếu đồ thị dùng cạnh `scene_relation` lấy từ annotation của ảnh test, mô hình "nhìn thấy" chuỗi đã tạo ra truy vấn, nên kết quả nhóm gián tiếp sẽ lạc quan. Phải ghi rõ trong báo cáo và nên làm ablation bỏ/xáo cạnh đó (ADR-0044, mục Hệ quả).

## 9. Kiểm thử

```bash
uv run pytest tests/test_scene_graph.py tests/test_query_generator.py tests/test_query_validator.py tests/test_query_benchmark_eval.py tests/test_queries_committed.py
```

`tests/test_queries_committed.py` kiểm mọi file `queries/*.jsonl` đã commit: đúng schema, đáp án thuộc test; file `*_test.jsonl` phải toàn truy vấn `verified`.
