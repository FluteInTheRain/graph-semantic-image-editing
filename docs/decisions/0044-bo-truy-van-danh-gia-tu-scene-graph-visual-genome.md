# ADR-0044: Sinh bộ truy vấn đánh giá trực tiếp/gián tiếp từ scene graph Visual Genome, chỉ dùng cặp đã được hai thành viên kiểm tra

- **Trạng thái:** Đề xuất
- **Ngày:** 2026-11-06

## Bối cảnh

Nhiệm vụ 2 (PDF §5, §7) yêu cầu **bộ tối thiểu 20 truy vấn đánh giá**, có nhóm **gián tiếp** "cần khai thác quan hệ (kiểu child - plays - football)", và báo cáo **tách riêng** nhóm trực tiếp và gián tiếp. PDF §9 còn yêu cầu nộp kèm **mô tả cách xây dựng và kiểm tra** bộ truy vấn. Truy vấn của ADR-0043 là region description: đều là truy vấn trực tiếp, không bảo đảm cần quan hệ.

Khái niệm cho người mới:

- **Scene graph** của một ảnh: node là đối tượng (mỗi đối tượng có `object_id` riêng), cạnh là quan hệ có hướng `chủ thể -vị từ-> đối tượng`, ví dụ `man -riding-> horse`.
- **Chuỗi quan hệ (chain)**: k ≥ 2 quan hệ **thật** của **cùng một ảnh**, nối nhau qua đối tượng chung **theo `object_id`** (không theo tên).
- **Node cầu (bridge)**: đối tượng nằm giữa chuỗi. Truy vấn gián tiếp **giấu** tên node cầu, chỉ mô tả nó qua quan hệ.
- **Truy vấn trực tiếp**: gọi tên mọi đối tượng. **Truy vấn gián tiếp**: phải lần theo quan hệ mới biết đối tượng bị giấu là gì.

Ví dụ **tổng hợp, chỉ dùng trong test, không phải dữ liệu Visual Genome**: `person -holding-> frisbee`, `dog -looks at-> frisbee` → trực tiếp "A person is holding a frisbee, and a dog looks at the frisbee." / gián tiếp "A person is holding the thing that a dog looks at."

## Quyết định

| Thành phần | Quy định |
|---|---|
| Nguồn | `objects.json` + `relationships.json` của Visual Genome, **chỉ ảnh thuộc một tập** của `splits/visual_genome/split_seed0.json` (`test` cho bộ chính thức, `val` cho pilot) |
| Đọc dữ liệu | Chuẩn hoá tên và vị từ (chữ thường, `_` → khoảng trắng, gộp khoảng trắng, bỏ dấu câu ở hai đầu), **giữ nguyên chuỗi gốc**. Hai đối tượng cùng tên **không bao giờ bị gộp**. Bản ghi hỏng bị bỏ và **đếm theo lý do** (`src/mmxai/data/scene_graph.py`) |
| Chuỗi | Đường đi đơn qua các `object_id` khác nhau, `min_hops = max_hops = 2` (3 bước cho câu lồng nhau khó đọc, khó kiểm tra). **Chỉ hai dạng:** (a) node cầu là đối tượng của cả hai quan hệ `A -p-> X <-q- B`; (b) nối tiếp `A -p-> X -q-> C`. **Bỏ dạng (c)** node cầu là chủ thể của cả hai (`X -p-> A`, `X -q-> B`): pilot cho thấy dạng này sinh câu mơ hồ ("the thing that has an eye has a foot") và hai đối tượng thật ra là một người ("baseball player" / "batter") |
| Vị từ | **Whitelist + template viết tay** trong config (`predicate_templates`). Trước khi tra: bỏ mạo từ cuối ("on a" → "on"), gộp biến thể (`predicate_aliases`: "wears" → "wearing", "holds" → "holding"). Vị từ ngoài whitelist → bỏ chuỗi. Whitelist chọn từ bảng tần suất vị từ trên **train** (không nhìn test). Hành động: wearing, holding, riding, carrying, eating, playing, watching, looking at, pulling, sitting on/in, standing on/in, walking on, laying on, hanging on/from, covering, attached to, painted on. Không gian: on, on top of, in, inside, under, above, behind, in front of, next to, near, beside. Loại: has/have, of, with, by, for, and, at, from, belonging to, between, along (sở hữu/bộ phận/quá mơ hồ) |
| Lọc | (1) tên chung chung (`it`, `thing`...) và **bộ phận cơ thể** (`head`, `wrist`, `hand`...) theo danh sách `generic_names` trong config (pilot val lần 2: "wearing something that is on a head" là hiển nhiên, không cần suy luận quan hệ); (1b) **tên chỉ gồm chữ cái** và khoảng trắng (pilot val lần 2: "A 52 is on a green/white uniform"); (2) **mỗi tên trong chuỗi là duy nhất trong ảnh** (ảnh có hai `dog` thì "the dog" mơ hồ); (3) **mỗi chuỗi có ≥ 1 vị từ hành động** (`action_predicates`), để truy vấn gián tiếp thật sự cần quan hệ kiểu "child - plays - football" chứ không chỉ vị trí; (4) câu gián tiếp không được chứa tên node cầu; (5) bỏ câu trùng; (6) bỏ chuỗi xuất hiện ở hơn `max_candidate_images = 25` ảnh test (quá chung, không kiểm tra tay được) |
| Sinh câu | Template cố định theo dạng chuỗi, không dùng LLM. (a) Direct "A {p} a X, and a B {q} the X." / Indirect "A {p} the thing that a B {q}."; (b) Direct "A {p} a X, and the X {q} a C." / Indirect "A {p} something that {q} a C.". Mạo từ "a"/"an" theo âm đầu; tên số nhiều (tên khác lemma của synset, ví dụ "shoes" vs `shoe.n.01`) dùng "some"; không bao giờ ghép "a the". Mỗi chữ truy được về một annotation |
| Cặp | Một chuỗi → một truy vấn `direct` + một `indirect`, cùng `pair_id`, cùng `relation_evidence` (id quan hệ thật), cùng tập đáp án |
| Đáp án ứng viên | **Mọi ảnh trong cùng tập** có scene graph chứa chuỗi cùng dạng, cùng vị từ đã chuẩn hoá, cùng chiều, qua các `object_id` khác nhau, và mỗi object **cùng synset WordNet** (trường `synsets` có sẵn trong annotation VG, ví dụ `man.n.01`; object thiếu synset thì khớp theo tên chuẩn hoá). Không tự viết bảng đồng nghĩa. Synset của *quan hệ* không dùng vì không đáng tin (VG gán `ON` → `along.r.01`). Ảnh nguồn chỉ là một đáp án trong số đó |
| Kiểm tra bởi người | `candidate` → thành viên A xem ảnh, ghi `reviewer` + `verified_relevant` → `reviewed` → thành viên B **khác A** kiểm lại, ghi `checker` → `verified`. Có thể bỏ (`rejected`) hoặc sửa câu (`edited_*_text`, câu gián tiếp sửa xong vẫn không được lộ tên node cầu) |
| Bộ chính thức | `queries/visual_genome_test.jsonl` chỉ gồm cặp `verified`, đáp án là `verified_relevant` do người xác nhận; cần ≥ 10 cặp (20 truy vấn). Script từ chối ghi đè |
| Đánh giá | CLIP thuần qua giao diện `QueryRanker` (chỉ nhận **văn bản** truy vấn, không nhận đáp án hay evidence); kho = 5.000 ảnh test; Recall@1/5/10/100 + MRR@100 cho **tất cả**, **direct**, **indirect**; lưu thứ hạng từng truy vấn |
| Thứ tự | Chạy pilot 5 cặp trên `val`, chốt mọi quy tắc trong config, **rồi mới** sinh trên `test`. Không đổi quy tắc, template hay ngưỡng sau khi thấy kết quả truy hồi trên test |

Code: `src/mmxai/queries/{schema,generator,validator}.py`, `src/mmxai/eval/query_benchmark.py`, `scripts/{generate_queries,build_query_benchmark,evaluate_query_benchmark}.py`, `configs/query_benchmark_vg{,_pilot}.yaml`, `configs/eval_query_benchmark_vg.yaml`. Hướng dẫn: `docs/query_benchmark.md`.

### Pilot val lần 1 (2026-11-06) và các chỉnh sửa
Pilot 5 cặp trên `val` với quy tắc ban đầu (mọi vị từ trừ danh sách cấm, khớp đáp án theo tên) chỉ cho khoảng 2/5 cặp dùng được:
- câu sai ngữ pháp do vị từ chứa mạo từ ("A lines is painted on a a ground", "painted on a the thing"), thiếu "is" ("the thing that worn by a batter");
- quan hệ bộ phận tầm thường ("bird - has - foot; bird - has - eye");
- dạng (c) nối hai object có thể là cùng một người ("green helmet - on - baseball player; green helmet - worn by - batter");
- 4/5 cặp chỉ có 1 ảnh liên quan vì khớp theo tên ("man" ≠ "guy").

Các hàng *Chuỗi*, *Vị từ*, *Lọc*, *Sinh câu*, *Đáp án ứng viên* ở trên là bản đã sửa theo pilot này. Pilot val lần 2 (sau khi cài đặt các quy tắc trên): câu đã đúng ngữ pháp; thêm hai bộ lọc (1) bộ phận cơ thể và (1b) tên chỉ gồm chữ cái, quyết định bởi Thuat Nguyen, 2026-11-06. Quyết định bởi: Thuat Nguyen (brainstorm với Claude Code), 2026-11-06. Quy trình kiểm tra hai người và 40 cặp ứng viên giữ nguyên. Sau khi cài đặt, chạy lại pilot val để kiểm tra câu trước khi sinh trên test.

## Các phương án đã cân nhắc

| Phương án | Ưu | Nhược | Lý do không chọn |
|---|---|---|---|
| Sinh từ chuỗi quan hệ thật + template + hai người kiểm tra | Mỗi truy vấn có bằng chứng kiểm được; tái lập được bằng seed | Câu hơi cứng ("the thing that...") | **Được chọn** |
| Viết tay hoàn toàn | Câu tự nhiên | Chậm, dễ thiên vị theo điều mô hình làm tốt, khó chứng minh "cần quan hệ" | Vẫn cho phép sửa câu trong manifest, nhưng phải giữ chuỗi gốc |
| Dùng LLM viết lại câu | Câu tự nhiên | Thêm dependency, câu có thể thêm ý không có trong ảnh, khó truy nguồn | Cần ADR riêng về dependency; khó kiểm tính trung thực |
| Blacklist vị từ + heuristic ngữ pháp | Nhiều chuỗi, đa dạng | Heuristic tiếng Anh sai ở ca lạ (pilot: "a a ground"), khó giải thích khi vấn đáp | Whitelist cho câu luôn đúng, dễ kiểm |
| Để người kiểm duyệt sửa câu xấu | Ít code | Tốn công, dễ thiên vị khi sửa | Chỉ dùng cho chỉnh nhỏ |
| Khớp đáp án theo tên chuẩn hoá | Đơn giản | Bỏ sót đồng nghĩa ("man"/"guy"), đa số truy vấn 1 đáp án | Synset có sẵn trong annotation, không phải tự bịa |
| Cho phép chuỗi chỉ gồm quan hệ không gian | Nhiều cặp hơn | Truy vấn gián tiếp yếu, không giống mẫu "child - plays - football" | Bắt buộc ≥ 1 vị từ hành động |
| Đáp án = chỉ ảnh nguồn | Đơn giản | Ảnh khác cũng đúng bị tính sai (giống vấn đề trong ADR-0043) | Không công bằng |
| Đáp án theo độ tương đồng CLIP | Không cần scene graph | Vòng lặp: dùng chính baseline để chấm baseline | Thiên vị CLIP |
| Đếm chuỗi trên toàn bộ dữ liệu | Nhiều chuỗi hơn | Đáp án nằm ngoài kho test | Vi phạm ADR-0009 |

## Hệ quả

- **Có bằng chứng cho phần "cách xây dựng và kiểm tra"** của báo cáo: mỗi dòng JSONL ghi `relationship_id`, `object_id`, chuỗi gốc, người kiểm tra.
- **Đáp án là cận dưới**: annotation Visual Genome không đầy đủ, ảnh có cảnh đúng nhưng thiếu annotation sẽ không được tìm thấy tự động. Người kiểm tra được thêm ảnh test vào `verified_relevant`.
- **Rò rỉ dữ liệu, cần nhóm lưu ý:** truy vấn gián tiếp được sinh từ chính quan hệ annotation của ảnh test. Nếu CLIP + Graph dựng cạnh `scene_relation` từ annotation của **ảnh test** (`configs/clip_graph_vg.yaml`), đồ thị nhìn thấy đúng các quan hệ đã dùng để tạo truy vấn, nên kết quả nhóm gián tiếp sẽ **lạc quan**. Việc dùng scene graph có sẵn được README cho phép (liên kết ảnh ↔ thực thể), nhưng báo cáo phải nói rõ điều này và nên có ablation (ví dụ bỏ hoặc xáo cạnh `scene_relation` của ảnh test). Cạnh thống kê vẫn chỉ đếm trên train.
- **Tách khỏi bảng chính:** run của bộ truy vấn ghi vào `results/query_benchmark/runs/` để `scripts/summarize_runs.py` không trộn metric dạng lồng (all/direct/indirect) với bảng `results/tables/main.csv`. Cần thêm bước tổng hợp riêng khi có ≥ 3 seed.
- **Việc tiếp theo:** tải annotation, chạy pilot val, chốt config, sinh 40 cặp trên test, hai thành viên kiểm tra ≥ 10 cặp, build bộ chính thức, commit (sau đó file bị đóng băng bởi hook, ADR-0023), rồi mới đánh giá.

## Nguồn tham khảo

- Đề bài, §5 (nhiệm vụ 2), §6, §7, §9.
- ADR-0009, ADR-0023, ADR-0042, ADR-0043.
- Krishna et al., "Visual Genome: Connecting Language and Vision Using Crowdsourced Dense Image Annotations", IJCV 2017.
- `queries/README.md` (định dạng JSONL và quy tắc kiểm tra chéo).
