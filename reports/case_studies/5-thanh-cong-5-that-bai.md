# Phân tích 5 truy vấn thành công + 5 truy vấn thất bại

Đề yêu cầu (PDF §5 cấp độ 1, §7 Nhiệm vụ 3, §8): hiển thị quan hệ được dùng khi xếp hạng và phân tích chi tiết
tối thiểu 5 truy vấn thành công, 5 truy vấn thất bại. Bảng tóm tắt và phần nhận xét ngắn nằm trong báo cáo
(`reports/report/project3_bao_cao.tex`, mục "Cấp độ 1: Occlusion và phân tích 5 thành công + 5 thất bại").
File này là bản đầy đủ: từng truy vấn, quan hệ đã dùng, phép kiểm bằng can thiệp và cách đọc.

## 1. Đọc file này thế nào

**Mô hình.** `clip_graph_vg_lr1e4`: GraphSAGE một lớp trên đồ thị `khái niệm → contains → ảnh` (ADR-0046).
Mỗi ảnh nối với 16 khái niệm (synset Visual Genome) gần nó nhất theo CLIP. Vector của ảnh sau GNN:

```
h_i = W_self x_i + b + mean_{c in N(i)} W_nb x_c
```

Điểm đồ thị là `g = cos(q, h_i)`. Điểm cuối là `α·cos(q, x_i) + (1 − α)·g`. Ở seed 0, `α = 0` (chọn trên val),
nên **thứ hạng chỉ do `g` quyết định**.

**Quan hệ được hiển thị = cạnh khái niệm → ảnh, đo bằng Occlusion.** Bỏ từng cạnh một, chấm lại bằng chính mô hình
đã xếp hạng. Điểm đồ thị giảm bao nhiêu thì đó là đóng góp của cạnh:

- `+`: cạnh đẩy ảnh lên (bỏ đi thì điểm giảm).
- `−`: cạnh kéo ảnh xuống (bỏ đi thì điểm tăng).

Không có bước tìm đường đi "hợp lý" nào sau khi đã xếp hạng (ADR-0016).

**Hai phép kiểm bằng can thiệp.** Mỗi phép chấm lại mọi ứng viên của truy vấn. Chỉ ảnh được giải thích bị đổi cạnh.

| Phép | Làm gì | Trả lời câu hỏi |
|---|---|---|
| Xoá 3 | Bỏ cùng lúc 3 cạnh `+` mạnh nhất | Giải thích có thật sự ảnh hưởng tới điểm và hạng không? |
| Bỏ mọi cạnh | Bỏ cả 16 cạnh, chỉ còn `W_self x_i + b` | Cả nhánh láng giềng (đồ thị) giúp hay hại ảnh này? |

Cần đọc **cả hai**. "Xoá 3" chỉ cho biết 3 cạnh tốt nhất có tác dụng. Nó không cho biết tổng thể đồ thị có lợi
cho ảnh hay không (xem truy vấn S3).

**Diễn giải nội dung ảnh** trong mục 4 chỉ suy từ **tên khái niệm**. Máy chạy chỉ có đặc trưng CLIP, chưa có file
ảnh, nên chưa ai mở 15 ảnh này để kiểm (xem mục 7).

## 2. Cách chọn truy vấn

Quy tắc cài trong `src/mmxai/eval/query_ranks.py`:

1. Chỉ xét truy vấn có ảnh đúng nằm trong top 100 ứng viên của CLIP. Ngoài top 100 thì xếp hạng lại không cứu được.
2. `change` = hạng của ảnh đúng đầu tiên theo CLIP − hạng theo CLIP + Graph.
3. Thành công: `change > 0`, lớn nhất trước. Thất bại: `change < 0`, âm nhất trước. Hoà thì lấy truy vấn có chỉ số nhỏ hơn.
4. Bỏ câu đã chọn rồi (giống chữ, không phân biệt hoa/thường và khoảng trắng).

Quy tắc 1–3 cố định trước khi xem kết quả. Quy tắc 4 thêm sau lần chạy đầu, vì câu "part of a trouser" chiếm 3/5
chỗ thành công: Visual Genome có nhiều region trùng câu, và các câu trùng có cùng thứ hạng. Quy tắc 4 chỉ loại câu
trùng, không đổi hướng chọn. Lần chạy đầu lưu ở `results/records/20261113_cap-do-1/run_copies/20261113-1002_*`.

**Hệ quả của quy tắc chọn.** Danh sách ứng viên có 100 ảnh, nên `change` lớn nhất chỉ xảy ra khi ảnh đúng nằm sát
đáy (thành công: CLIP hạng 98–100) hoặc bị đẩy xuống sát đáy (thất bại: hạng 92–100). 10 truy vấn này là các **ca
cực trị**, không đại diện cho truy vấn điển hình. Trên toàn val (25.000 truy vấn): 7.387 tăng hạng, 4.620 giảm,
1.463 giữ nguyên, 11.530 có ảnh đúng ngoài top 100.

## 3. Bảng tổng quan (val, seed 0)

| Mã | Truy vấn | Hạng CLIP → CLIP + Graph | Xoá 3: hạng | Bỏ mọi cạnh: hạng | Đồ thị giúp hay hại ảnh đúng? |
|---|---|---|---|---|---|
| S1 | "man in a grey shirt and blue jeans" | 99 → 1 | 1 → 1 | 1 → 7 | giúp 6 bậc |
| S2 | "Blue jacket with orange stripes" | 100 → 3 | 3 → 5 | 3 → 5 | giúp 2 bậc |
| S3 | "part of a trouser" | 100 → 3 | 3 → 4 | 3 → 1 | **hại** 2 bậc |
| S4 | "a girl wearing blue jeans." | 99 → 3 | 3 → 3 | 3 → 3 | không đổi |
| S5 | "the grass beside the car" | 98 → 3 | 3 → 3 | 3 → 8 | giúp 5 bậc |
| F1 | "scattered waves in water" | 7 → 100 | 100 → 100 | 100 → 96 | hại 4 bậc |
| F2 | "tall green and tan grass" | 15 → 96 | 96 → 96 | 96 → 94 | hại 2 bậc |
| F3 | "wheel of car" | 15 → 96 | 96 → 97 | 96 → 82 | hại 14 bậc |
| F4 | "a wooden sign" | 12 → 92 | 92 → 95 | 92 → 34 | **hại 58 bậc** |
| F5 | "handicap parking sign on metal post" | 15 → 95 | 95 → 98 | 95 → 75 | hại 20 bậc |

Cột cuối so hạng khi có cạnh với hạng khi bỏ mọi cạnh của riêng ảnh đúng. Số liệu gốc:
`clip_graph_vg_lr1e4_val_seed0/analysis.json` và `analysis.md`.

## 4. Phân tích từng truy vấn

Ký hiệu: `g` là điểm đồ thị (cũng là điểm cuối, vì `α = 0`). Đóng góp ghi sau tên khái niệm.

### Thành công

**S1. "man in a grey shirt and blue jeans"** (ảnh 2398911, CLIP 99 → 1)

- Đẩy lên: `competition` +0,0030, `action` +0,0025, `scene` +0,0017.
- Kéo xuống: `tennis` −0,0027, `seesaw` −0,0021, `trick` −0,0017.
- Xoá 3: `g` 0,2046 → 0,1962, hạng giữ 1. Bỏ mọi cạnh: `g` → 0,1679, hạng 1 → 7.
- Đọc: phần lớn cú nhảy (99 → 7) đến từ vector của chính ảnh. Đồ thị thêm bước cuối (7 → 1). Các khái niệm có lợi
  là về **bối cảnh** (thi đấu, hành động), không có gì về áo xám hay quần jeans: đồ thị không có khái niệm thuộc
  tính như màu sắc, quần áo.

**S2. "Blue jacket with orange stripes"** (ảnh 2326205, CLIP 100 → 3)

- Đẩy lên: `ski_run` +0,0017, `ski_resort` +0,0016, `downhill` +0,0014.
- Kéo xuống: `ski_cap` −0,0041, `ski_pole` −0,0018, `snowshoe` −0,0016.
- Xoá 3: `g` 0,1771 → 0,1717, hạng 3 → 5. Bỏ mọi cạnh: hạng 3 → 5.
- Đọc: ca giải thích rõ nhất. Đúng 3 cạnh được chọn đã tạo ra toàn bộ phần đồ thị đóng góp (2 bậc): xoá 3 và bỏ mọi
  cạnh cho cùng hạng. Bối cảnh trượt tuyết hợp với "áo khoác", trong khi các vật dụng trượt tuyết cụ thể lại kéo xuống.

**S3. "part of a trouser"** (ảnh đúng đầu tiên 2387245, CLIP 100 → 3)

- Câu này có ở nhiều ảnh. Mọi ảnh có cùng câu đều tính là đúng (ADR-0043), nên ảnh đúng đầu tiên là 2387245,
  không phải ảnh nguồn 2326733.
- Đẩy lên: `catcher` +0,0014, `player` +0,0009, `outfielder` +0,0007.
- Kéo xuống: `pitcher` −0,0009, `one-third` −0,0008, `pitch` −0,0006.
- Xoá 3: `g` 0,0727 → 0,0693, hạng 3 → 4. Bỏ mọi cạnh: `g` **tăng** lên 0,0987, hạng 3 → **1**.
- Đọc: "thành công" này **không nhờ đồ thị**. Tổng thể 16 cạnh kéo ảnh xuống 2 bậc. Nếu chỉ hiển thị 3 cạnh `+`
  (chủ đề bóng chày), người xem sẽ tưởng đồ thị đã giúp. Đây là lý do mỗi giải thích phải kèm phép "bỏ mọi cạnh".
  `g` chỉ 0,07 mà vẫn hạng 3: truy vấn mơ hồ nên mọi ứng viên đều có điểm thấp.

**S4. "a girl wearing blue jeans."** (ảnh 2341744, CLIP 99 → 3)

- Đẩy lên: `sociable` +0,0023, `main` +0,0022, `course` +0,0016.
- Kéo xuống: `adolescent` −0,0029, `student` −0,0018, `friend` −0,0014.
- Xoá 3: hạng giữ 3. Bỏ mọi cạnh: hạng giữ 3.
- Đọc: cạnh không đổi hạng. Cú nhảy 99 → 3 hoàn toàn đến từ vector của chính ảnh. Ba khái niệm gần "girl" nhất
  (`adolescent`, `student`) lại kéo xuống (mẫu lặp lại, xem mục 5.3).

**S5. "the grass beside the car"** (ảnh 2368850, CLIP 98 → 3)

- Đẩy lên: `sedan` +0,0014, `caption` +0,0012, `ready` +0,0009.
- Kéo xuống: `pursuit` −0,0022, `police` −0,0017, `policeman` −0,0017.
- Xoá 3: `g` 0,1761 → 0,1722, hạng giữ 3. Bỏ mọi cạnh: hạng 3 → 8.
- Đọc: đồ thị giúp 5 bậc, và `sedan` (gần "car") đẩy lên đúng nghĩa. Nhưng 3 cạnh tốt nhất một mình không đổi hạng:
  phần đóng góp rải trên nhiều cạnh. Mô hình lấy trung bình 16 cạnh, nên mỗi cạnh chỉ góp phần nhỏ.

### Thất bại

Với truy vấn thất bại, mình giải thích cả ảnh đúng lẫn ảnh đang đứng hạng 1, để thấy vì sao ảnh hạng 1 thắng.

**F1. "scattered waves in water"** (ảnh đúng 2798, CLIP 7 → 100)

- Ảnh đúng. Đẩy lên: `current` +0,0023, `view` +0,0016, `screen_saver` +0,0014. Kéo xuống: `blue_sky` −0,0018,
  `marsh` −0,0016, `sky` −0,0014. Bỏ mọi cạnh: hạng 100 → 96.
- Ảnh hạng 1 (2353055, CLIP 4). Đẩy lên: `competition`, `action`, `fun`. Kéo xuống: `swell`, `surf`, `surfboard`.
  Bỏ mọi cạnh: `g` 0,2300 → 0,2297, vẫn hạng 1.
- Đọc: nguyên nhân chính **không phải cạnh**. Bỏ mọi cạnh thì ảnh đúng vẫn ở 96, và ảnh hạng 1 gần như không nhờ cạnh.
  Phép chiếu học được đã đẩy ảnh đúng xuống. Theo tên khái niệm, ảnh hạng 1 có thể là cảnh lướt sóng, khá hợp với
  "waves in water". Nhãn đúng chỉ tính ảnh có cùng câu, nên ca này có thể là **nhãn thiếu** chứ không hẳn là lỗi
  (cần mở ảnh để xác nhận).

**F2. "tall green and tan grass"** (ảnh đúng 2405444, CLIP 15 → 96)

- Ảnh đúng. Đẩy lên: `body_waste` +0,0015, `scrape` +0,0014, `plot` +0,0013. Kéo xuống: **`grass` −0,0031**,
  `ditch` −0,0014, `grave` −0,0013. Bỏ mọi cạnh: hạng 96 → 94.
- Ảnh hạng 1 (2410837, CLIP 3). Đẩy lên: `course`, `gazelle`, `antelope`. Kéo xuống: **`grassland` −0,0032**,
  `savanna`, `graze`. Bỏ mọi cạnh: vẫn hạng 1.
- Đọc: lại do phép chiếu là chính, cạnh chỉ góp 2 bậc. Khái niệm đúng nghĩa nhất (`grass`, `grassland`) kéo xuống ở
  **cả hai** ảnh. Các khái niệm đẩy lên ảnh đúng là nhiễu (`body_waste`). Ảnh hạng 1 có vẻ là đồng cỏ có linh dương,
  cũng có thể là ảnh hợp nghĩa.

**F3. "wheel of car"** (ảnh đúng 2406229, CLIP 15 → 96)

- Ảnh đúng. Đẩy lên: `utility` +0,0029, `valve` +0,0015, `drain` +0,0009. Kéo xuống: `drinking_fountain` −0,0025,
  `fireplug` −0,0020, `spout` −0,0010. `g` chỉ 0,0020. Bỏ mọi cạnh: `g` → 0,0331, hạng 96 → 82.
- Ảnh hạng 1 (2363372, CLIP 1, giữ hạng 1). Đẩy lên: `sedan`, `car`, `car.n.02`. Kéo xuống: `hubcap`, `car_tire`,
  `tire`. Bỏ mọi cạnh: vẫn hạng 1.
- Đọc: 16 khái niệm của ảnh đúng mô tả **vật chính của cả ảnh** (có vẻ là trụ cứu hoả), không mô tả bánh xe. Truy vấn
  là mô tả **một vùng** (region description), còn cạnh gắn với **cả ảnh**. Vì vậy đồ thị kéo ảnh đúng xuống thêm
  14 bậc. Ảnh hạng 1 là ảnh xe hơi, đứng đầu cả theo CLIP: có thể cũng là ảnh hợp nghĩa.

**F4. "a wooden sign"** (ảnh đúng 2367684, CLIP 12 → 92)

- Ảnh đúng. Đẩy lên: `rung` +0,0041, `tongs` +0,0023, `kickstand` +0,0019. Kéo xuống: `bus_stop` −0,0027,
  `bollard` −0,0023, `stopping_point` −0,0018. Bỏ mọi cạnh: `g` 0,0014 → 0,0721, hạng 92 → **34**.
- Ảnh hạng 1 (2363366, CLIP 1). Đẩy lên: `adam`, `bric-a-brac`, `amour_propre` (nhiễu). Kéo xuống: **`street_sign`
  −0,0047**, `signpost` −0,0038, `street_name` −0,0032. Bỏ mọi cạnh: `g` tăng lên 0,2044, vẫn hạng 1.
- Đọc: ca **đồ thị gây hại rõ nhất**: 58 bậc. Phép chiếu đưa ảnh đúng xuống 34, cạnh kéo tiếp xuống 92. Ở ảnh hạng 1,
  khái niệm biển báo (đúng nghĩa) kéo xuống, khái niệm nhiễu đẩy lên.

**F5. "handicap parking sign on metal post"** (ảnh đúng 2388012, CLIP 15 → 95)

- Ảnh đúng. Đẩy lên: `passerby` +0,0036, `utility` +0,0035, `post` +0,0017. Kéo xuống: `parking_meter` −0,0035,
  `bollard` −0,0023, `drinking_fountain` −0,0019. Bỏ mọi cạnh: hạng 95 → 75.
- Ảnh hạng 1 (2393216, CLIP 3). Đẩy lên: `seller`, `customer`, `campaign`. Kéo xuống: **`parking_meter` −0,0054**,
  `parking_space` −0,0040, `parking_lot` −0,0029. Bỏ mọi cạnh: vẫn hạng 1.
- Đọc: đồ thị hại 20 bậc. `post` (khớp "metal post") đẩy lên đúng nghĩa, nhưng các khái niệm về đỗ xe kéo xuống ở cả
  hai ảnh.

## 5. Các mẫu lặp lại

**5.1. Vector của chính ảnh (`W_self x_i + b`) là yếu tố chính, cả khi thành công lẫn thất bại.**

- Bỏ mọi cạnh, 5 ảnh thành công vẫn đứng hạng 7 / 5 / 1 / 3 / 8 (CLIP: 98–100).
- Ảnh đúng của 5 truy vấn thất bại vẫn ở 96 / 94 / 82 / 34 / 75 (CLIP: 7–15).
- Cả 5 ảnh hạng 1 của truy vấn thất bại vẫn hạng 1 khi bỏ mọi cạnh.
- Khớp với ablation toàn cục: đồ thị thật chỉ hơn đối chứng không đồ thị và đồ thị xáo khoảng 0,001–0,002 MRR
  (`results/tables/main.csv`).

**5.2. Đồ thị giúp ít, hại nhiều hơn ở các ca cực trị.** Trong 5 thành công, đồ thị giúp 3 ca (2–6 bậc), không đổi
1 ca, hại 1 ca. Trong 5 thất bại, đồ thị hại cả 5 (2–58 bậc).

**5.3. Khái niệm gần nghĩa truy vấn nhất thường có đóng góp âm.**

- Ví dụ: `adolescent`/`student` (S4, "girl"); `grass`, `grassland` (F2); `hubcap`/`car_tire`/`tire` (F3, "wheel");
  `street_sign`/`signpost` (F4, "sign"); `parking_meter`/`parking_space` (F5, "parking").
- Ngược lại: `sedan` (S5, "car"), `car` (F3, ảnh hạng 1), `post` (F5, "post") vẫn đóng góp dương.
- Lý do có thể: mô hình lấy trung bình 16 khái niệm rồi mới tính cosine. Bỏ một cạnh làm cả vector trung bình đổi
  hướng, nên dấu đóng góp phụ thuộc vào khái niệm đó so với **trung bình** của 15 khái niệm còn lại (qua `W_nb`),
  không chỉ vào nghĩa của nó.
- Đây là **giả thuyết, chưa kiểm**. Cách kiểm: tính `qᵀ W_nb x_c` cho các khái niệm trên và so với `qᵀ W_nb` nhân
  vector trung bình láng giềng.

**5.4. Cạnh gắn với cả ảnh, còn truy vấn mô tả một vùng hoặc thuộc tính.** Truy vấn Visual Genome là region
description ("wheel of car", "part of a trouser", "grey shirt and blue jeans"). Khái niệm của ảnh mô tả bối cảnh
hoặc vật chính, không có thuộc tính (màu sắc) hay bộ phận (bánh xe). Đồ thị hiện tại khó giúp các truy vấn này.

**5.5. Một số "thất bại" có thể là nhãn thiếu.** Ảnh đúng chỉ gồm ảnh có cùng câu (ADR-0043). Theo tên khái niệm,
ảnh hạng 1 của F1 (lướt sóng), F2 (đồng cỏ) và F3 (xe hơi, CLIP cũng xếp 1) có thể hợp nghĩa với truy vấn. Chưa
mở ảnh nên chưa kết luận.

**5.6. Cạnh lấy từ CLIP top-16 có nhiễu.** Ví dụ `screen_saver`, `body_waste`, `adam`, `amour_propre`. Cạnh không
lấy từ annotation, để không "đọc đáp án" trên val/test (ADR-0046, mục A2). Giải thích hiển thị đúng đồ thị mà mô
hình đã dùng, kể cả phần nhiễu.

## 6. Liên hệ với đánh giá định lượng cấp độ 3

- Trên 15 ảnh ở đây, xoá 3 cạnh do Occlusion chọn đều làm `g` giảm. Giải thích có tác động đúng hướng.
- Kết luận này được kiểm trên mẫu ngẫu nhiên (20 truy vấn val, 3 seed; `results/runs/20261114-0453_explain_vg_val/summary.md`):
  - Xoá `m = 10` cạnh do Occlusion chọn làm ảnh tụt trung bình 3,47 ± 0,46 bậc.
  - Xoá 10 cạnh ngẫu nhiên chỉ làm tụt 0,22 ± 0,03 bậc (p = 3,05e-05).
- Cùng lúc, Fid+ tuyệt đối nhỏ (≈ 0,02). Điều đó khớp với mục 5.1: giải thích trung thực với nhánh đồ thị, nhưng
  nhánh đồ thị chỉ là một phần nhỏ của điểm.

## 7. Giới hạn và việc nhóm cần làm trước khi nộp

- [ ] **Mở 15 ảnh** (10 ảnh đúng, 5 ảnh hạng 1 của truy vấn thất bại) để kiểm phần "Đọc" ở mục 4 và mục 5.5. Hiện
      diễn giải chỉ dựa trên tên khái niệm.
- [ ] Kiểm giả thuyết ở mục 5.3, hoặc ghi rõ trong báo cáo đây là giả thuyết.
- Chỉ 1 seed (0) và chỉ trên val. Script từ chối `test`, vì test chỉ chạy một lần trong `evaluate_graph_rerank.py`.
- Truy vấn là region description (ADR-0043), không phải bộ truy vấn trực tiếp/gián tiếp của cấp độ 2 (bộ này chưa có).

## Phụ lục: chạy lại

```bash
uv run python scripts/explain.py --config configs/clip_graph_vg_lr1e4.yaml --seed 0 \
    --split val --num-edges 3 --vg-data-dir data/visual_genome
```

| Mục | Giá trị |
|---|---|
| Mô hình | `clip_graph_vg_lr1e4`, seed 0, `α = 0` (chọn trên val khi huấn luyện) |
| Tập | val, 25.000 truy vấn |
| Code | commit `9689c79`, `git_dirty: false` (`clip_graph_vg_lr1e4_val_seed0/env.json`) |
| Số liệu gốc | `clip_graph_vg_lr1e4_val_seed0/analysis.json`, chép từ `results/runs/20261113-1036_clip_graph_vg_lr1e4_explain_seed0/` |
| Kiểm tra khớp | MRR@100 val của CLIP + Graph trong phân tích bằng đúng số `selection.json` lưu cho seed 0 khi huấn luyện (0,12855). Script tự kiểm và dừng nếu khác |
| Dữ liệu | `region_descriptions.json` (v1.2), sha256 `19624f10afb10bf69e30ecf48d85b1a71420c7432e8c88f1011a3c4fd60ec768` |

Checkpoint được huấn luyện lại, chỉ dùng train + val. Kết quả val khớp tới 4 chữ số với số đã ghi trong
`docs/guides/huan-luyen-va-xep-hang-lai.md` §7: epoch tốt nhất 6/8/6, `α` 0/0/0,1, val MRR@100 0,1281 ± 0,0008.
