# ADR-0050: Sinh truy vấn gián tiếp bằng cách mô tả khái niệm bị giấu qua tri thức ConceptNet

- **Trạng thái:** Đề xuất
- **Ngày:** 2026-11-15

## Bối cảnh

Nhiệm vụ 2, cấp độ 2 (PDF §5) yêu cầu bộ **≥ 20 truy vấn đánh giá**, có nhóm **truy vấn gián tiếp cần khai thác quan hệ (kiểu child - plays - football)**, và báo cáo tách riêng nhóm trực tiếp với nhóm gián tiếp. PDF §2 giải thích loại truy vấn này: *"với truy vấn a child playing football, nếu trong ảnh chỉ nhận diện được ball và field, đường đi child - plays - football trong đồ thị tri thức giúp kết nối truy vấn với đúng bức ảnh"*.

Khái niệm cho người mới:

- **Khái niệm bị giấu** (hidden concept): thứ mà truy vấn không gọi tên, ví dụ *tennis*.
- **Tính chất** của khái niệm: một cạnh trong đồ thị tri thức, ví dụ `racket –UsedFor→ tennis` hay `tennis –IsA→ sport`.
- **ConceptNet**: đồ thị tri thức thông thường (commonsense KG) công khai, mỗi cạnh có dạng `(đầu, quan hệ, cuối, trọng số)`.
- **Truy vấn gián tiếp v2**: mô tả khái niệm bị giấu **chỉ qua tính chất của nó**, ví dụ "A person playing a sport that uses a racket and a net". Muốn nối câu này với ảnh, mô hình phải đi qua cạnh tri thức (racket → tennis), đúng như ví dụ trong đề.

ADR-0044 (Đề xuất) định nghĩa câu gián tiếp theo cách khác: giấu tên một đồ vật **có trong ảnh**, mô tả nó qua quan hệ trong **scene graph của chính ảnh đó**. Review trước PR #4 nêu ba điểm yếu của cách này:

1. Nó lệch với ví dụ của đề: ở đề, khái niệm không nhìn thấy trong ảnh, và mối nối nằm ở KG ngoài.
2. Câu gián tiếp rộng nghĩa hơn câu trực tiếp nhưng dùng chung đáp án. Vì vậy nhóm gián tiếp bị chấm thiệt (mục H1 của experiment-reviewer).
3. Mô tả và đáp án cùng lấy từ annotation của ảnh test, nên mô hình đồ thị dùng annotation đó sẽ rò nhãn (leakage-auditor).

Số đếm trên **train** của Visual Genome (2026-11-15, chỉ dùng để chọn khái niệm, không nhìn test): đối tượng của vị từ `playing` gồm frisbee 483 ảnh, tennis 295, baseball 172, ball 140, soccer 47. Bóng đá quá hiếm (ước khoảng 2 ảnh test), nên nhóm hoạt động dùng các môn phổ biến hơn, và có thêm nhóm đồ vật.

## Quyết định

| Thành phần | Quy định |
|---|---|
| Nguồn tri thức | ConceptNet 5.7, file `conceptnet-assertions-5.7.0.csv.gz`, đặt ở `data/conceptnet/` (không commit). Lọc **một lần** bằng `scripts/prepare_conceptnet.py`: chỉ giữ cạnh có hai đầu đều tiếng Anh (`/c/en/`), thuộc quan hệ `IsA, UsedFor, AtLocation, HasA, PartOf, CapableOf`, trọng số ≥ 1. File lọc ghi vào `artifacts/conceptnet/` kèm sha256. Mỗi cạnh giữ nguyên URI gốc để làm bằng chứng. Có data card `data_cards/conceptnet.md` |
| Nối VG với ConceptNet | Lemma của synset VG thành node ConceptNet: `tennis.n.01` → `/c/en/tennis`, `fire_hydrant.n.01` → `/c/en/fire_hydrant`. Không khớp thì bỏ khái niệm. Tên có nhiều synset trong cùng nhóm thì bỏ (`bat.n.01` và `bat.n.05` cho hai cặp cùng câu, khác đáp án) |
| Hai nhóm khái niệm | **activity**: đối tượng của vị từ `playing` (sau chuẩn hoá vị từ của ADR-0044), có chủ thể là người, xác định bằng danh sách synset trong config (man, woman, boy, girl, child, person, player…). **object**: synset của object VG |
| Chọn khái niệm đích | Chỉ đếm trên **train**: khái niệm phải xuất hiện ở ≥ `min_train_images` ảnh train. Khái niệm phải có ít nhất một cạnh `IsA` tới một **từ loại** trong danh sách config (sport, game, vehicle, animal, instrument, fruit, furniture, tool…), và ≥ 2 cạnh tính chất |
| Ảnh đáp án | Trong split **test**. activity: ảnh có quan hệ *(người) –playing→ C*, so khớp synset. object: ảnh có object mang synset C. Số ảnh đáp án phải nằm trong `[min_answer_images, max_answer_images]`; ngoài khoảng thì bỏ khái niệm. Khoảng này cố định trước, không đổi sau khi có kết quả truy hồi |
| Mô tả duy nhất | Mỗi câu gián tiếp gồm **1 từ loại + 2 tính chất**. Tập tính chất phải đúng với **chỉ C** trong kho so sánh. Kho so sánh gồm **mọi** khái niệm cùng nhóm xuất hiện ở ≥ 1 ảnh train và nối được sang ConceptNet, không chỉ các khái niệm đích. Ví dụ "uses a racket" phải loại được cả badminton dù badminton không phải khái niệm đích. Cách chọn: tham lam, ưu tiên tính chất ít khái niệm khác có, hoà thì theo trọng số rồi theo thứ tự chữ. Quy trình tất định: duyệt khái niệm theo tần suất trên train, hoà thì theo trọng số cạnh rồi theo thứ tự chữ, không dùng seed. Không tìm được mô tả duy nhất thì bỏ khái niệm |
| Không lộ tên | Bỏ tính chất có đầu kia chứa tên hoặc lemma của C ("tennis racket" khi C là tennis). Câu gián tiếp không được chứa tên bị giấu (validator v1, `hidden_names_in_text`) |
| Tính chất số nhiều | Bỏ tính chất (trừ loại "can") có từ cuối tận cùng bằng "s" nhưng không phải "ss" ("legs", "glasses"; giữ "glass", "grass") vì mẫu câu "has a legs" sai ngữ pháp; quy tắc cố ý hẹp, nhận nhầm "bus" chỉ làm mất một tính chất ứng viên (review 2026-11-15) |
| Template | Theo quan hệ và chiều cạnh: `X UsedFor C` → "that uses a X"; `C AtLocation Y` → activity "that is played on a Y", object "that is found at a Y"; `C HasA X` hoặc `X PartOf C` → "that has a X"; `C CapableOf V` → "that can V". activity: direct "A person playing C." / indirect "A person playing a K that …"; object: direct "A C." / indirect "A K that …" (K là từ loại). "a/an/some" như ADR-0044 |
| Cặp và đáp án | Một khái niệm → một cặp direct + indirect, cùng `pair_id`, **cùng đáp án**. Dùng chung đáp án hợp lệ vì mô tả trỏ duy nhất tới C, nên vấn đề H1 không còn |
| Bằng chứng trong JSONL | Thêm các trường tuỳ chọn `hidden_concept` (synset), `query_group` (`activity` / `object`), `kg_evidence` (danh sách `{uri, rel, start, end, weight}`). Giữ các trường bắt buộc của `queries/README.md` |
| Kiểm duyệt | Như ADR-0044: manifest CSV, A duyệt rồi B (khác A) kiểm lại; người duyệt xác nhận tính chất đúng với khái niệm, câu đọc được, ảnh đúng. Có thể sửa câu (`edited_*_text`) nhưng không được thêm tên bị giấu |
| Số lượng | Sinh khoảng 30 cặp ứng viên, cần ≥ 10 cặp `verified` (≥ 20 truy vấn), trong đó ≥ 4 cặp activity |
| Thứ tự | Pilot trên **val** để chỉnh từ loại, template và ngưỡng, rồi chốt config (`configs/query_benchmark_vg_kg.yaml` và bản pilot), sau đó mới sinh trên test. Một test bắt hai config dùng chung mọi quy tắc |
| Đánh giá | Dùng lại `src/mmxai/eval/query_benchmark.py`: Recall@1/5/10/100 + MRR cho all / direct / indirect, thêm tách theo `query_group` |
| Bộ v1 (ADR-0044) | Giữ code, không đưa vào bộ chính thức |
| Flickr30k | Thiết kế để thêm sau. Chỉ phần "tìm ảnh đáp án" phụ thuộc dataset (Flickr30k: caption nhắc C, tính trên split Karpathy test); ConceptNet, chọn mô tả và template dùng chung |

Code: `src/mmxai/data/conceptnet.py`, `src/mmxai/queries/kg_facts.py`, `src/mmxai/queries/kg_targets.py`, `src/mmxai/queries/kg_text.py`, `src/mmxai/queries/kg_generator.py`; config `configs/conceptnet_en.yaml`, `configs/query_benchmark_vg_kg.yaml`, `configs/query_benchmark_vg_kg_pilot.yaml`, `configs/eval_query_benchmark_vg_kg.yaml`; script `scripts/prepare_conceptnet.py`, `scripts/generate_kg_queries.py`, `scripts/build_kg_query_benchmark.py`.

Ngưỡng ban đầu, sẽ chỉnh trên pilot val trước khi sinh trên test: `min_train_images = 20`, `min_answer_images = 3`, `max_answer_images = 50`.

## Các phương án đã cân nhắc

| Phương án | Ưu | Nhược | Lý do không chọn |
|---|---|---|---|
| ConceptNet + template + hai người duyệt | Đúng ví dụ của đề; mỗi chữ truy được về một cạnh ConceptNet; tái lập bằng seed; cạnh tri thức không phụ thuộc split nên mô hình đồ thị dùng được mà không rò nhãn | Phải tải khoảng vài trăm MB; câu hơi cứng | **Được chọn** |
| Bảng tính chất viết tay (có ghi nguồn) | Nhanh, không tải gì | Câu hỏi và tri thức do cùng người viết; mô hình đồ thị không có các cạnh đó | Khó bảo vệ khi vấn đáp |
| WordNet (gloss, hypernym) | VG đã có synset | Ít quan hệ kiểu "uses / played on"; câu giống từ điển | Không thể hiện được quan hệ như đề |
| Giữ định nghĩa của ADR-0044 | Code đã có | Lệch ví dụ của đề, chấm thiệt nhóm gián tiếp, rủi ro rò nhãn | Xem Bối cảnh |
| Cho mô tả mơ hồ, đáp án gồm mọi khái niệm khớp | Câu tự nhiên hơn | Câu gián tiếp và trực tiếp có đáp án khác nhau; phụ thuộc độ đầy đủ của ConceptNet | Bắt buộc mô tả duy nhất đơn giản và kiểm được hơn |
| Chỉ nhóm activity | Sát ví dụ nhất | Trên VG chỉ khoảng 5–6 khái niệm đủ ảnh | Thêm nhóm object để đủ số lượng |
| Script gợi ý, người viết câu / viết tay hoàn toàn | Câu tự nhiên | Khó tái lập, dễ thiên vị; PDF §8 cần mô tả cách xây dựng và kiểm tra | Template vẫn cho phép người duyệt sửa câu |

## Hệ quả

- **Đúng yêu cầu đề hơn:** truy vấn gián tiếp cần một bước suy luận qua KG ngoài, đúng kiểu child – plays – football.
- **Rò rỉ dữ liệu:**
  - Khái niệm đích chọn theo tần suất trên train; quy tắc chỉnh trên val; đáp án chỉ lấy trong test.
  - ConceptNet là tri thức ngoài, không phụ thuộc split.
  - Đáp án được định nghĩa từ annotation VG của ảnh test (quan hệ `playing`, object synset). Vì vậy mô hình đồ thị **không được** dựng cạnh từ annotation của ảnh test. ADR-0046 đã quy định cạnh `contains` của mọi split lấy từ CLIP top-k, còn cạnh annotation chỉ làm ablation cận trên; kết quả ablation đó trên bộ này phải ghi là **oracle**.
- **Hợp với phương pháp cấp độ 2:** cạnh ConceptNet đã lọc dùng được làm đồ thị khái niệm cho G-Retriever / PCST. Nếu dùng, phải ghi rõ trong báo cáo, kèm ablation bỏ hoặc xáo các cạnh này.
- **Khó hơn:**
  - Cần tải và lọc ConceptNet; cần data card và ghi license CC BY-SA 4.0.
  - Số khái niệm activity trên VG ít, nên nhóm activity có thể dưới 10 cặp. Báo cáo phải ghi rõ cỡ mẫu từng nhóm.
- **Đáp án là cận dưới:** annotation VG không đầy đủ, nên ảnh có tennis mà không được ghi `playing tennis` sẽ bị coi là sai. Người duyệt được thêm ảnh đúng nhưng không được dùng kết quả truy hồi để tìm.
- **Không thêm dependency Python:** chỉ dùng `csv` và `gzip` của thư viện chuẩn.
- **Việc tiếp theo:**
  1. Tải ConceptNet.
  2. Cài đặt theo kế hoạch.
  3. Pilot trên val.
  4. Nhóm duyệt ADR này và quyết trạng thái của ADR-0044.
  5. Sinh trên test, kiểm duyệt, build, đánh giá.

## Nguồn tham khảo

- Đề bài, §2, §5, §6, §7, §8.
- ADR-0009, ADR-0040, ADR-0044, ADR-0046.
- Speer, Chin, Havasi, "ConceptNet 5.5: An Open Multilingual Graph of General Knowledge", AAAI 2017.
- ConceptNet downloads: https://github.com/commonsense/conceptnet5/wiki/Downloads (link tải và dung lượng file cần kiểm lại trước khi tải).
- Kết quả review leakage-auditor và experiment-reviewer trước PR #4 (2026-11-06).
