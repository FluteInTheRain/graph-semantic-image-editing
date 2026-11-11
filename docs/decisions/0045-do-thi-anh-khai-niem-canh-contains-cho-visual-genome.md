# ADR-0045: Dựng đồ thị ảnh – khái niệm chỉ với cạnh `contains` cho Visual Genome

- **Trạng thái:** Đề xuất
- **Ngày:** 2026-11-07

## Bối cảnh

Cấp độ 1 (PDF §4.2, §5) cần một **đồ thị tĩnh** dựng từ dữ liệu có sẵn để GNN (GCN/GAT/GraphSAGE) tính điểm đồ thị, rồi kết hợp với điểm CLIP. Đây là bước đầu tiên của phần đồ thị: chưa có GNN, chưa có rerank.

Khái niệm cho người mới:

- **Đỉnh (node)**: một thứ trong đồ thị. Ở đây có hai loại: **ảnh** và **khái niệm** (loại đồ vật, ví dụ "dog").
- **Cạnh `contains`**: nối một ảnh với một khái niệm khi annotation của Visual Genome ghi ảnh đó có đồ vật thuộc khái niệm này.
- **Synset**: mã khái niệm của WordNet mà Visual Genome gắn sẵn vào đồ vật, ví dụ `dog.n.01`. Nhờ synset, "dog" và "puppy" thành **một** đỉnh.
- Mỗi khái niệm chỉ có **một** đỉnh dù xuất hiện trong hàng nghìn ảnh. Nhờ vậy các ảnh được nối với nhau qua khái niệm chung, và GNN có đường để truyền thông tin.

Ràng buộc:

- Bộ truy vấn của ADR-0044 được sinh từ **quan hệ** (`holding`, `on`...) trong scene graph của ảnh test. Nếu đồ thị có cạnh quan hệ lấy từ annotation của ảnh test, đồ thị chứa đúng chuỗi quan hệ đã dùng để ra đề (ADR-0044, mục Hệ quả).
- Thống kê (đếm tần suất) chỉ được tính trên **train** (PDF §6, ADR-0009).
- Code GNN và rerank phải dùng chung cho Visual Genome và Flickr30k sau này. Chỉ phần đọc dữ liệu là riêng cho từng bộ.

## Quyết định

| Thành phần | Quy định |
|---|---|
| Dữ liệu | `objects.json` bản v1.4 từ trang chính thức của Visual Genome. Không cần `relationships.json` |
| Đỉnh ảnh | **Mọi** ảnh trong `splits/visual_genome/split_seed0.json` (train, val, test), kể cả ảnh không có đồ vật nào. Ảnh nào cũng phải có đỉnh, vì ảnh test là ứng viên khi tìm kiếm |
| Đỉnh khái niệm | Synset đầu tiên của mỗi đồ vật. Đồ vật **không có synset** được tra theo tên trong **từ điển tên → synset học trên train** (synset gặp nhiều nhất cho tên đó; hoà thì lấy synset đứng trước theo thứ tự chữ cái). Tên không có trong từ điển thì bỏ và đếm |
| Từ vựng khái niệm | Giữ synset xuất hiện trong **≥ `min_train_images` = 5 ảnh train**. Chỉ đếm trên train, có test chứng minh ảnh test không làm đổi từ vựng |
| Cạnh | Chỉ `contains`: ảnh → khái niệm, mỗi cặp một lần dù ảnh có nhiều đồ vật cùng khái niệm. **Không có** cạnh quan hệ giữa khái niệm, không có ConceptNet |
| Vector của đỉnh | Ảnh: embedding CLIP đã cache (ADR-0041). Khái niệm: CLIP text encoder (đóng băng, chỉ inference) với câu `"a photo of a {lemma}"`, ví dụ `fire_hydrant.n.01` → "a photo of a fire hydrant". Cùng model `ViT-B-32` / `laion2b_s34b_b79k`, nên ảnh và khái niệm nằm trong cùng không gian 512 chiều |
| Lưu trữ | `artifacts/graphs/visual_genome/contains_v1/`: `graph.npz` (id ảnh, khoá và tên khái niệm, vector khái niệm, hai mảng cạnh) + `meta.json` (sha256 của `objects.json` và split, số đếm, cấu hình, commit). Không ghi đè |
| Định dạng chung | `src/mmxai/graph/contains.py` (chung) nhận `{image_id: tập khái niệm}`. `src/mmxai/graph/visual_genome.py` là phần riêng của VG. `src/mmxai/graph/pyg.py` đổi sang `torch_geometric.data.Data`: ảnh trước, khái niệm sau, cạnh lưu cả hai chiều |

Quyết định bởi: qphulong, 2026-11-07 (tải từ trang chính thức, chỉ cạnh `contains`, tra synset theo tên bằng từ điển học trên train). Các giá trị còn lại (`min_train_images = 5`, câu mẫu `"a photo of a {}"`) do Claude Code đề xuất, **nhóm chưa duyệt**.

## Các phương án đã cân nhắc

| Phương án | Ưu | Nhược | Lý do không chọn |
|---|---|---|---|
| Chỉ cạnh `contains`, khái niệm theo synset | Đơn giản, dễ giải thích khi vấn đáp, ít rủi ro rò rỉ với ADR-0044 | Không có quan hệ giữa đồ vật, nên truy vấn gián tiếp ít được hỗ trợ | **Được chọn** cho bản đầu |
| Thêm cạnh quan hệ scene graph (`on`, `holding`...) | Phong phú, hợp với truy vấn gián tiếp | Quan hệ của ảnh test chính là nguồn sinh truy vấn (ADR-0044), kết quả nhóm gián tiếp sẽ lạc quan | Để PR sau, kèm ablation bỏ/xáo cạnh |
| Khái niệm theo tên chuẩn hoá thay vì synset | Không mất đồ vật nào | "man", "men", "guy" thành ba đỉnh; hàng chục nghìn tên rác chỉ gặp một lần | Synset có sẵn, gom đồng nghĩa mà không phải tự viết bảng |
| Chỉ dùng đồ vật có synset, bỏ phần còn lại (bản chạy đầu tiên) | Đơn giản nhất | 13.320 ảnh có đồ vật nhưng VG để trường `synsets` rỗng cho mọi đồ vật, nên ~14% ảnh không có cạnh nào (trên **val**: 717/5.000; quyết định dựa trên train/val, không dựa trên test) | Đồ thị bỏ rơi quá nhiều ảnh |
| Synset, thiếu thì tra theo tên bằng từ điển học trên train | Cứu được ~94% đồ vật thiếu synset; ảnh không có cạnh còn ~1,7% | Thêm một bước; từ điển chọn một nghĩa cho mỗi tên ("bank" luôn thành một synset) | **Được chọn**. Trên train, synset được chọn khớp với synset thật ở 99,8% đồ vật |
| Synset, thiếu thì dùng tên làm khoá (như `match_key` của ADR-0044) | Không cần từ điển | Tạo đỉnh trùng nghĩa: tên "man" (không synset) tách khỏi `man.n.01` | Từ điển gom về đúng đỉnh synset |
| Không lọc theo tần suất | Không mất khái niệm hiếm | Đỉnh chỉ gặp ở 1 ảnh không nối ảnh nào với nhau, chỉ thêm nhiễu | Ngưỡng 5 ảnh train, đếm trên train |
| Đếm từ vựng trên mọi ảnh | Ảnh test không mất cạnh nào | Thống kê nhìn thấy test (vi phạm PDF §6) | Không được phép |
| Lưu thẳng `Data` của PyG (`.pt`) | Nạp nhanh | Cần torch để đọc, khó kiểm tra bằng numpy, nhân đôi 220 MB vector ảnh | Lưu `.npz`, đổi sang `Data` lúc dùng |

## Hệ quả

- GNN và rerank làm việc trên một `Data` chung. Flickr30k chỉ cần viết thêm một hàm trả `{image_id: tập khái niệm}`.
- Khái niệm chỉ xuất hiện trong ảnh test (hoặc gặp ở < 5 ảnh train) không có đỉnh. Ảnh test mất cạnh tới các khái niệm đó. Đây là cái giá của việc không nhìn thống kê trên test.
- **Rò rỉ, cần nhóm lưu ý (leakage-auditor, 2026-11-07):** cạnh `contains` của ảnh test lấy từ annotation của chính ảnh đó. README §6.1 cho phép việc này (scene graph có sẵn thay cho detector). Nhưng nhãn đúng của bộ truy vấn cũng lấy từ **cùng annotation**:
  - ADR-0044: một ảnh chỉ được tính là đáp án khi scene graph của nó có **đủ mọi synset của chuỗi**, kể cả đồ vật cầu nối bị giấu trong câu gián tiếp (`src/mmxai/queries/validator.py`, `graph_has_pattern`). Vì vậy **mọi ảnh đúng đều có cạnh `contains` tới mọi khái niệm của chuỗi** (nếu khái niệm có trong từ vựng). Đây là điều kiện cần của nhãn, không chỉ là "thường có sẵn". Ảnh thật sự chứa cảnh đó nhưng thiếu annotation thì vừa không được tính là đúng, vừa không có cạnh.
  - ADR-0043: region description và đồ vật của ảnh do cùng người gán nhãn viết.

  Kết quả CLIP + Graph trên test sẽ **lạc quan một cách có hệ thống**. Bắt buộc khi làm thực nghiệm: (1) đo tỉ lệ ảnh đúng có đủ cạnh tới mọi khái niệm của truy vấn, so với ảnh test ngẫu nhiên; (2) ablation trên **cạnh của ảnh test**: thay bằng cạnh không lấy từ annotation (ví dụ top-k khái niệm theo độ giống CLIP ảnh – text, giữ nguyên số cạnh), xáo cạnh giữ bậc, và bỏ hẳn cạnh (chỉ còn vector CLIP); (3) báo cáo cả bản có và không có annotation của ảnh test, tách direct/indirect.
- **Bắt buộc trước PR GNN (leakage-auditor, 2026-11-07):** `to_pyg_data` (`src/mmxai/graph/pyg.py`) gộp ảnh train, val, test vào một `Data` và chưa đánh dấu split. GNN **không được** truyền thông tin qua đỉnh/cạnh của ảnh val/test lúc huấn luyện (PDF §6: ảnh ứng viên test không dùng để huấn luyện GNN). PR GNN phải: thêm split cho từng đỉnh ảnh; huấn luyện chỉ trên đồ thị con gồm ảnh train + khái niệm; gắn ảnh val/test vào lúc suy luận; có test chứng minh đỉnh test không nằm trong `edge_index` lúc huấn luyện; ghi quyết định (inductive) vào ADR.
- Từ điển tên → synset là một thống kê nên **chỉ đếm trên train** (test: `test_name_dictionary_is_learned_on_train_images_only`). Ảnh val/test chỉ **tra** từ điển, không đóng góp vào nó.
- **Lần chạy 2026-11-07 (có tra theo tên):** 108.077 đỉnh ảnh, 3.880 đỉnh khái niệm, 1.719.645 cạnh. Nguồn synset: 80,4% có sẵn trong annotation, 15,1% tra theo tên, 4,5% không tra được (bỏ). Ảnh không có cạnh: train 1.643, val 92, test 70 (~1,7%), gần hết là 1.793 ảnh không có đồ vật nào.
- Bản chạy đầu (chưa tra theo tên) có ~14% ảnh không có cạnh; vì vậy bước tra theo tên được thêm vào.
- Việc tiếp theo: GNN scorer (đọc `Data`, theo yêu cầu inductive ở trên), rồi rerank.
- `min_train_images = 5` và câu mẫu `"a photo of a {}"` chưa được tinh chỉnh. Nếu tinh chỉnh sau này, chỉ dùng metric trên **val**, không dùng `queries/`.

## Nguồn tham khảo

- Đề bài, §4.1, §4.2, §5, §6.
- ADR-0009, ADR-0041, ADR-0042, ADR-0043, ADR-0044.
- Krishna et al., "Visual Genome: Connecting Language and Vision Using Crowdsourced Dense Image Annotations", IJCV 2017.
- Trang tải Visual Genome: https://homes.cs.washington.edu/~ranjay/visualgenome/api.html
