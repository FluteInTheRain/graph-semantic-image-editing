# Bộ truy vấn đánh giá từ scene graph Visual Genome (Nhiệm vụ 2)

> **Trạng thái 2026-11-06:** code, test và quy tắc sinh **v2** (sau 3 lần pilot trên **val**) đã có; **chưa sinh trên test**, nên hiện có **0 truy vấn ứng viên thật trên test, 0 truy vấn đã kiểm tra**. Quyết định thiết kế: [ADR-0044](decisions/0044-bo-truy-van-danh-gia-tu-scene-graph-visual-genome.md) (Đề xuất).

## 1. Ý tưởng trong 30 giây

Đề (PDF §5, Nhiệm vụ 2) cần ≥ 20 truy vấn, có nhóm **gián tiếp** phải khai thác quan hệ. Ta không tự bịa truy vấn, mà lấy từ **scene graph** có sẵn của Visual Genome:

1. Tìm trong một ảnh **hai quan hệ thật** nối nhau qua **một đối tượng chung** (cùng `object_id`).
2. Viết **truy vấn trực tiếp**: gọi tên mọi đối tượng.
3. Viết **truy vấn gián tiếp**: giấu tên đối tượng ở giữa (node cầu), chỉ mô tả nó qua quan hệ.
4. Tìm **mọi ảnh test** có cùng chuỗi quan hệ → danh sách đáp án ứng viên.
5. **Hai thành viên** xem ảnh, xác nhận đáp án. Chỉ cặp đã xác nhận mới vào bộ chính thức.

### Ví dụ từ pilot trên **val** (không thuộc bộ test, không dùng để đánh giá)

Hai cặp thật sinh bởi `configs/query_benchmark_vg_pilot.yaml` (file `results/runs/query_pilot_val/visual_genome_candidates_val.jsonl`, gitignored).

| Bước | Dạng `path` (A -p-> X -q-> C) | Dạng `shared_object` (A -p-> X <-q- B) |
|---|---|---|
| Scene graph | `person -holding-> kite`, `kite -in-> sky` | `horse -pulling-> carriage`, `man -sitting in-> carriage` |
| Trực tiếp | "A person is holding a kite, and the kite is in a sky." | "A horse is pulling a carriage, and a man is sitting in the carriage." |
| Gián tiếp | "A person is holding something that is in a sky." (giấu `kite`) | "A horse is pulling the thing that a man is sitting in." (giấu `carriage`) |
| Đáp án ứng viên | 1 ảnh (ảnh nguồn `2410609`) | 1 ảnh (ảnh nguồn `2373004`) |

Dòng JSONL đầy đủ của câu gián tiếp dạng `path` (in đẹp cho dễ đọc; file thật là một dòng):

```json
{
  "qid": "vg-val-005-i",
  "text": "A person is holding something that is in a sky.",
  "type": "indirect",
  "dataset": "visual_genome",
  "relevant": ["2410609"],
  "path_hint": "person - holding - kite; kite - in - sky",
  "author": "generator-template-v2",
  "pair_id": "vg-val-005",
  "source_image_id": "2410609",
  "relation_evidence": [
    {
      "relationship_id": 212462,
      "subject_id": 213653, "subject_name": "person", "subject_synset": "person.n.01",
      "predicate": "holding",
      "object_id": 213652, "object_name": "kite", "object_synset": "kite.n.03",
      "raw": ["person", "holding", "kite"]
    },
    {
      "relationship_id": 212468,
      "subject_id": 213652, "subject_name": "kite", "subject_synset": "kite.n.03",
      "predicate": "in",
      "object_id": 213654, "object_name": "sky", "object_synset": "sky.n.01",
      "raw": ["kite", "IN", "sky"]
    }
  ],
  "chain_object_ids": [213653, 213652, 213654],
  "hop_count": 2,
  "split": "val",
  "verification_status": "candidate",
  "reasoning_type": "path",
  "hidden_names": ["kite"]
}
```

Hai điểm cần nhớ:

- Đáp án chỉ có **1 ảnh** (chính ảnh nguồn). Đáp án là **cận dưới** (mục 5.4): người kiểm tra nên tìm thêm ảnh test đúng.
- Từ pilot lần 3, mỗi `relation_evidence` có thêm `subject_synset` / `object_synset` (synset WordNet có sẵn trong annotation VG), dùng để khớp đáp án.

Ví dụ **tổng hợp** chỉ để giải thích test (dữ liệu giả trong `tests/scene_graph_fixtures.py`, **không phải** ảnh Visual Genome): `person -holding-> frisbee`, `dog -looks at-> frisbee`.

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

Lệnh viết dạng `uv run ...`. Máy chưa có `uv` trong PATH thì dùng `python -m uv run ...`.

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

Code: `src/mmxai/queries/generator.py` (sinh, lọc), `src/mmxai/queries/validator.py` (đáp án), `src/mmxai/data/scene_graph.py` (đọc dữ liệu, `match_key`). Mọi ngưỡng và danh sách nằm trong `configs/query_benchmark_vg.yaml`. Bản này là **quy tắc v2**, đã sửa sau pilot val (mục 5.5).

### 5.1. Chuẩn hoá và whitelist vị từ

Vị từ trong Visual Genome rất lộn xộn ("ON", "is wearing", "on a"). Ta xử lý theo thứ tự:

| Bước | Ví dụ |
|---|---|
| Chữ thường, `_` thành khoảng trắng | "looks_at" → "looks at" |
| Bỏ "is/are" ở đầu | "is wearing" → "wearing" |
| Bỏ mạo từ ở cuối | "on a" → "on" |
| Gộp biến thể bằng `predicate_aliases` | "wears" → "wearing", "holds" → "holding" |
| Tra whitelist `predicate_templates` | không có trong whitelist → bỏ chuỗi (`predicate_not_in_whitelist`) |

`predicate_templates` cho mỗi vị từ một cụm động từ **viết tay** (ví dụ `wearing: is wearing`, `laying on: is lying on`), nên câu luôn đúng ngữ pháp, không cần đoán bằng heuristic. Whitelist chọn từ bảng tần suất vị từ trên **train** (không nhìn test). Cố ý bỏ: `has/have`, `of`, `with`, `by`, `for`, `and`, `at`, `from`, `belonging to`, `between`, `along` (sở hữu, bộ phận hoặc quá mơ hồ).

### 5.2. Dạng chuỗi và vị từ hành động

- Chỉ chuỗi **2 quan hệ** (`min_hops = max_hops = 2`), qua các `object_id` khác nhau.
- Chỉ **hai dạng** được giữ:

| Dạng (`reasoning_type` của câu gián tiếp) | Chuỗi |
|---|---|
| `shared_object` | A -p-> X <-q- B |
| `path` | A -p-> X -q-> C |

- Dạng `shared_subject` (A <-p- X -q-> B) bị **loại** (`shared_subject_chain`): pilot cho câu mơ hồ ("the thing that has an eye has a foot") và hai đối tượng thật ra là một người.
- Chuỗi phải có **≥ 1 vị từ hành động** trong `action_predicates` (wearing, holding, riding, carrying, eating, playing, watching, looking at, pulling, sitting on/in, standing on/in, walking on, laying on, hanging on/from, covering). Thiếu thì bị loại (`no_action_predicate`). Vì sao: để truy vấn gián tiếp giống mẫu "child - plays - football", không chỉ là quan hệ vị trí.
- Câu trực tiếp luôn có `reasoning_type = "explicit"`; dạng chuỗi chỉ ghi ở câu gián tiếp.

### 5.3. Bộ lọc

| Bộ lọc | Lý do loại (tên trong bảng tóm tắt) | Vì sao |
|---|---|---|
| Tên chung chung, gồm cả **bộ phận cơ thể** (`generic_names`: `thing`, `it`, `head`, `hand`..., cả số nhiều `hands`, `legs`...) | `generic_name` | "wearing something that is on a head" hiển nhiên, không cần suy luận quan hệ |
| Tên phải là **từ thường** (`PLAIN_WORDS`: chữ cái thường, cách nhau một khoảng trắng) | `name_not_plain_words` | Pilot sinh "A 52 is on a green/white uniform" |
| Vị từ ngoài whitelist | `predicate_not_in_whitelist` | Mục 5.1 |
| Tên đối tượng **duy nhất trong ảnh** | `ambiguous_name` | Ảnh có hai `dog` thì "the dog" mơ hồ |
| Hai node trong chuỗi **cùng synset** (`man`/`guy` cùng `man.n.01`) | `repeated_synset_in_chain` | Đáp án khớp theo synset; có thể là cùng một người với hai tên |
| Dạng chuỗi `shared_subject` | `shared_subject_chain` | Mục 5.2 |
| Không có vị từ hành động | `no_action_predicate` | Mục 5.2 |
| Tên node cầu (bị giấu) xuất hiện trong câu gián tiếp | `hidden_name_in_indirect_text` | Nếu không, đó chỉ là câu trực tiếp viết lại |
| Câu trùng (tối đa 1 cặp mỗi ảnh) | `duplicate_query_text` | Bộ truy vấn trải trên nhiều ảnh |
| Chuỗi có > `max_candidate_images = 25` ảnh test khớp | `too_many_candidate_images` | Quá chung, không kiểm tay được |

Thứ tự ảnh xáo bằng `seed`, duyệt tất định: chạy lại ra đúng cùng file.

### 5.4. Template câu, đáp án ứng viên

Template cố định theo dạng chuỗi, không dùng LLM:

| Dạng | Trực tiếp | Gián tiếp |
|---|---|---|
| `shared_object` | "A {p} a X, and a B {q} the X." | "A {p} the thing that a B {q}." |
| `path` | "A {p} a X, and the X {q} a C." | "A {p} something that {q} a C." |

`{p}`, `{q}` là cụm động từ trong `predicate_templates`. Mạo từ và động từ **be** theo số ít/nhiều:

- Tên số ít: "a" hoặc "an" (theo âm đầu), "is".
- Tên số nhiều: "some", "are". Hàm `is_plural` coi tên là số nhiều khi nó bằng lemma của synset + `s`/`es` (ví dụ "shoes" với `shoe.n.01`).
- Không bao giờ ghép "a the" hay "a a".

**Đáp án ứng viên** (`relevant`): mọi ảnh trong **cùng tập** (test cho bộ chính thức) có chuỗi **cùng dạng, cùng vị từ chuẩn hoá, cùng chiều**, qua các `object_id` khác nhau, và mỗi đối tượng có cùng **khoá khớp**:

| Đối tượng | Khoá khớp |
|---|---|
| Có synset (ví dụ `man.n.01`) | synset WordNet (có sẵn trong annotation VG, nên "man" và "guy" cùng khoá) |
| Không có synset | tên chuẩn hoá |

Code: `match_key` (`src/mmxai/data/scene_graph.py`, lúc dựng chuỗi) và `evidence_key` (`src/mmxai/queries/validator.py`, đọc lại từ JSONL; vì thế JSONL lưu `subject_synset`/`object_synset`). Synset của *quan hệ* không dùng vì không đáng tin (VG gán `ON` → `along.r.01`).

Đáp án là **cận dưới**: annotation Visual Genome không đầy đủ, ảnh đúng nhưng thiếu annotation sẽ không được tìm thấy tự động. Người kiểm tra được **thêm** ảnh test vào `verified_relevant` (mục 4).

### 5.5. Lịch sử pilot val

Chỉ chạy trên **val** (5 cặp mỗi lần), file ra nằm ngoài `queries/`. Quy tắc chốt trước khi sinh trên test.

| Lần | Quy tắc | Kết quả |
|---|---|---|
| 1 | Cũ: mọi vị từ trừ danh sách cấm, khớp đáp án theo tên, cho cả dạng `shared_subject` | Khoảng 2/5 cặp dùng được. Lỗi: "a a ground" (vị từ chứa mạo từ), quan hệ bộ phận tầm thường ("has"), `shared_subject` nối hai tên của cùng một người, chỉ có 1 đáp án |
| 2 | Sửa ngữ pháp: whitelist + template, bỏ `shared_subject`, bắt buộc vị từ hành động, khớp theo synset | Câu đúng ngữ pháp. Còn "A 52 is on a green/white uniform" và "wearing something that is on a head" |
| 3 | Thêm bộ lọc tên chỉ gồm từ thường và bộ phận cơ thể | **5/5 cặp đọc được**. Cả 5 vẫn chỉ có 1 ảnh đáp án, nên đáp án là cận dưới, người kiểm tra nên tìm thêm ảnh test đúng |

Số chuỗi bị loại ở pilot lần 3 (5.000 ảnh val):

| Lý do | Số chuỗi |
|---|---|
| `ambiguous_name` | 311 |
| `predicate_not_in_whitelist` | 168 |
| `generic_name` | 79 |
| `shared_subject_chain` | 43 |
| `no_action_predicate` | 32 |
| `name_not_plain_words` | 9 |

Các số này là **tập val**, không phải test; khi sinh trên test thì ghi lại bảng `rejected_chains` mới vào báo cáo.

## 6. Định dạng file

Giữ đúng các trường bắt buộc của `queries/README.md` (`qid`, `text`, `type`, `dataset`, `relevant`, `path_hint`, `author`) và thêm:

| Trường | Ý nghĩa |
|---|---|
| `pair_id` | Cặp trực tiếp/gián tiếp; hai truy vấn cùng cặp có cùng đáp án |
| `source_image_id` | Ảnh sinh ra chuỗi |
| `relation_evidence` | Các quan hệ thật đã dùng (id quan hệ, id đối tượng, tên chuẩn hoá, `subject_synset`/`object_synset`, chuỗi gốc `raw`) |
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


---

## Bộ truy vấn v2: khái niệm mô tả qua ConceptNet (ADR-0050)

### Ý tưởng

Bộ v1 (các mục trên) sinh câu gián tiếp từ chuỗi quan hệ trong annotation ảnh. Bộ v2 làm khác: lấy **một khái niệm** C (ví dụ môn `tennis` hoặc vật `train`), rồi **mô tả C mà không nói tên C**, bằng tri thức thường thức của ConceptNet.

| | Câu trực tiếp | Câu gián tiếp |
|---|---|---|
| Nói gì | Gọi tên C | Chỉ nêu 1 từ loại + 2 tính chất của C |
| Đáp án | Cùng một tập ảnh | Cùng một tập ảnh |
| Cần gì để trả lời | Khớp chữ | Biết "môn chơi trên sân và dùng vợt" là tennis |

Điều kiện công bằng: mô tả phải đúng với **chỉ C** trong nhóm của nó. Nếu "dùng vợt" cũng đúng với badminton thì ảnh badminton cũng là đáp án đúng, nên mô tả đó bị loại.

### Các loại fact (`src/mmxai/queries/kg_facts.py`, hàm `facts_from_edge`)

Một cạnh ConceptNet cho ra một fact về một khái niệm. Chiều cạnh quyết định khái niệm nào "sở hữu" fact.

| Cạnh | Khái niệm sở hữu | `kind` | Ví dụ cụm trong câu |
|---|---|---|---|
| `C IsA K` | C | `isa` (chỉ dùng làm từ loại) | "a sport" |
| `X UsedFor C` | C | `uses` | "uses a racket" |
| `C AtLocation Y` | C | `location` | "is played on a court" |
| `C HasA X` | C | `has` | "has a X" |
| `X PartOf C` | C | `has` | "has a X" |
| `C CapableOf V` | C | `can` | "can carry passengers" (động từ, không có "a") |

Cạnh khác bị bỏ. Cùng (khái niệm, `kind`, giá trị) từ nhiều cạnh thì giữ cạnh nặng nhất (`build_fact_index`).

### Cách chọn mô tả duy nhất (`choose_description`)

1. **Kho so sánh** (`pool`) là mọi khái niệm cùng nhóm xuất hiện ở ít nhất 1 ảnh train, không chỉ các khái niệm đích.
2. Chọn **từ loại**: cạnh `IsA` có giá trị nằm trong `category_words` của config. Ưu tiên từ loại ít khái niệm khác chung nhất, hoà thì theo thứ tự chữ.
3. Chọn **tính chất** theo kiểu tham lam: mỗi lần lấy fact mà ít khái niệm còn lại (cùng từ loại) có nhất. Hoà thì lấy cạnh có `weight` cao hơn, rồi theo thứ tự chữ.
4. Bỏ tính chất lộ tên (`leaks_name`: "tennis ball" khi C là tennis) và tính chất không phải chữ thường đơn giản.
5. Sau `num_properties` tính chất mà vẫn còn khái niệm giống C thì bỏ C (`no_unique_description`).

Thứ tự duyệt khái niệm cũng tất định: nhiều ảnh train nhất trước, hoà thì theo tên synset. Cả quy trình **không dùng seed**.

### Vì sao một khái niệm bị loại

`run_kg_generation` in số khái niệm bị loại theo lý do (`rejected_concepts`). Ghi các số này vào báo cáo.

| Lý do | Nghĩa |
|---|---|
| `too_few_train_images` | Ít hơn `min_train_images` ảnh train |
| `concept_already_used` | Synset này đã có cặp |
| `name_not_plain_words` | Tên không phải chữ thường đơn giản |
| `no_conceptnet_facts` | Tên không có trong ConceptNet |
| `answer_count_out_of_range` | Số ảnh đáp án ngoài `[min_answer_images, max_answer_images]` |
| `no_unique_description` | Không có mô tả đúng riêng cho C |

### Ví dụ (DỮ LIỆU TỔNG HỢP, không phải ConceptNet hay Visual Genome)

Ví dụ lấy từ test fixture `tests/conceptnet_fixtures.py` và `tests/test_kg_text.py`. Các cạnh do nhóm tự bịa để biết trước đáp án: tennis và badminton cùng "dùng vợt", chỉ tennis có "sân".

| Thành phần | Giá trị tổng hợp |
|---|---|
| Cạnh | `tennis IsA sport`, `racket UsedFor tennis`, `tennis AtLocation court`, `racket UsedFor badminton` |
| Câu trực tiếp | `A person playing tennis.` |
| Câu gián tiếp | `A person playing a sport that is played on a court and uses a racket.` |
| Vì sao đủ | "dùng vợt" cũng đúng với badminton, nhưng "chơi trên sân" chỉ đúng với tennis |

Nhóm object, cũng từ fixture: `A train.` và `A vehicle that is found at a railway and can carry passengers.`

### Các trường mới trong JSONL và manifest

| Trường | Nghĩa |
|---|---|
| `hidden_concept` | Synset bị giấu (ví dụ `tennis.n.01`) |
| `query_group` | `activity` hoặc `object` |
| `kg_evidence` | Danh sách cạnh ConceptNet đã dùng: `{uri, rel, start, end, weight}`. `check_kg_evidence` kiểm mỗi `uri` có trong file cạnh |
| `path_hint` | Các cạnh viết thành chuỗi "start - rel - end" |

Manifest CSV thêm cột `query_group`, `hidden_concept`, `kg_edges` (các URI cách nhau bằng dấu cách) để người duyệt đối chiếu. Kết quả đánh giá thêm `by_query_group` (direct/indirect cho từng nhóm) bên cạnh `all/direct/indirect`.

### Lệnh chạy theo thứ tự

Trước hết tự tải `conceptnet-assertions-5.7.0.csv.gz` (xem `data_cards/conceptnet.md`) và đặt vào `data/conceptnet/`. File này không được commit.

| # | Việc | Lệnh |
|---|---|---|
| 1 | Lọc ConceptNet một lần (chỉ cạnh tiếng Anh, 6 quan hệ) | `uv run python scripts/prepare_conceptnet.py --config configs/conceptnet_en.yaml` |
| 2 | Pilot trên **val**, đọc thử câu | `uv run python scripts/generate_kg_queries.py --config configs/query_benchmark_vg_kg_pilot.yaml` |
| 3 | Chốt quy tắc (chỉ dựa trên val) | sửa `configs/query_benchmark_vg_kg.yaml` và bản pilot cho giống nhau |
| 4 | Chạy test | `uv run pytest` |
| 5 | Sinh ứng viên trên **test** | `uv run python scripts/generate_kg_queries.py --config configs/query_benchmark_vg_kg.yaml` |
| 6 | Review: A duyệt, B kiểm lại (mục 4) | sửa `queries/query_review_manifest_kg.csv` |
| 7 | Dựng bộ chính thức từ các cặp `verified` (cần ít nhất `min_verified_pairs` = 10) | `uv run python scripts/build_kg_query_benchmark.py --config configs/query_benchmark_vg_kg.yaml` |
| 8 | Đánh giá CLIP, mỗi seed một lần | `uv run python scripts/evaluate_query_benchmark.py --config configs/eval_query_benchmark_vg_kg.yaml --seed <k>` |

### Lưu ý rò rỉ

- Khái niệm đích và kho so sánh chỉ đếm trên **train**. Ảnh test chỉ dùng để lấy đáp án.
- Pilot chạy trên val. Không chỉnh quy tắc sau khi nhìn kết quả truy hồi trên bộ test.
- Đáp án lấy từ annotation của ảnh test. Mô hình đồ thị **không được dựng cạnh từ annotation ảnh test** và không được đọc `hidden_concept`, `kg_evidence`, `path_hint`.
- Dùng cạnh ConceptNet bên trong mô hình là việc ngoài ADR-0050, cần ADR riêng.
