# Phân tích 5 truy vấn thành công + 5 truy vấn thất bại (cấp độ 1)

PDF §5 cấp độ 1 yêu cầu: hiển thị quan hệ được dùng khi xếp hạng cho một số truy vấn, và phân tích 5 truy vấn
thành công + 5 truy vấn thất bại. File này là bản nháp phân tích. Nhóm cần đọc lại và viết lại bằng lời của mình
trước khi đưa vào báo cáo.

## 1. Ý tưởng trong 1 phút

- Với mỗi truy vấn, so **thứ hạng ảnh đúng đầu tiên** khi chỉ dùng CLIP và khi dùng CLIP + Graph.
  - Ảnh được đẩy lên nhiều nhất: **thành công**.
  - Ảnh bị kéo xuống nhiều nhất: **thất bại**.
- Giải thích bằng **Occlusion**: lần lượt bỏ từng cạnh `khái niệm → ảnh` của ảnh đó, rồi chạy lại **chính mô hình
  đã xếp hạng**. Điểm đồ thị giảm bao nhiêu thì đó là đóng góp của cạnh.
  - Không có bước "tìm đường đi hợp lý" nào sau khi đã xếp hạng (ADR-0016).
- Hai phép kiểm tra bằng can thiệp, mỗi lần chấm lại mọi ứng viên của truy vấn:
  - "xoá 3": bỏ cùng lúc 3 cạnh đẩy ảnh lên mạnh nhất;
  - "bỏ mọi cạnh": bỏ cả 16 cạnh vào ảnh. Cái còn lại là vector của chính ảnh qua phép chiếu học được (`W_self x_i + b`).

## 2. Chạy lại

```bash
uv run python scripts/explain.py --config configs/clip_graph_vg_lr1e4.yaml --seed 0 \
    --split val --num-edges 3 --vg-data-dir data/visual_genome
```

| Mục | Giá trị |
|---|---|
| Mô hình | `clip_graph_vg_lr1e4` (GraphSAGE một lớp, ADR-0046), seed 0, `alpha = 0.0` (chọn trên val khi huấn luyện) |
| Tập | **val** (25.000 truy vấn). Script từ chối `test`, vì test chỉ chạy một lần trong `evaluate_graph_rerank.py` |
| Code | commit `9689c79`, `git_dirty: false` (`clip_graph_vg_lr1e4_val_seed0/env.json`) |
| Số liệu gốc | `reports/case_studies/clip_graph_vg_lr1e4_val_seed0/analysis.json` (bản chép từ `results/runs/20261113-1036_clip_graph_vg_lr1e4_explain_seed0/`) |
| Kiểm tra khớp | MRR@100 val của CLIP + Graph trong phân tích bằng đúng số `selection.json` lưu cho seed 0 khi huấn luyện (0,12855). Script tự kiểm điều này và dừng nếu khác |

Checkpoint được huấn luyện lại trên máy này, chỉ dùng train + val. Kết quả val khớp tới 4 chữ số với số mà nhóm đã ghi trong
`docs/guides/huan-luyen-va-xep-hang-lai.md` §7: epoch tốt nhất 6/8/6, `alpha` 0/0/0,1, val MRR@100 0,1281 ± 0,0008.

### Quy tắc chọn truy vấn (cố định trong code, `src/mmxai/eval/query_ranks.py`)

1. Chỉ xét truy vấn có ảnh đúng nằm trong top 100 của CLIP.
2. `change` = hạng theo CLIP − hạng theo CLIP + Graph.
3. Thành công: `change > 0`, lớn nhất trước. Thất bại: `change < 0`, âm nhất trước. Hoà: lấy truy vấn có chỉ số nhỏ hơn.
4. Câu đã được chọn rồi (giống chữ, không phân biệt hoa/thường và khoảng trắng) thì bỏ qua.

Quy tắc 1–3 cố định trước khi xem kết quả. Quy tắc 4 được thêm sau lần chạy đầu tiên. Lần đó, câu "part of a trouser" chiếm 3 trong 5 chỗ thành công, vì Visual
Genome có nhiều region trùng câu và các câu trùng có cùng thứ hạng. Quy tắc này chỉ loại câu trùng, không đổi hướng
chọn. Lần chạy đầu được lưu ở `results/records/20261113_cap-do-1/run_copies/20261113-1002_*`.

## 3. Bảng tổng quan (val, seed 0)

Tổng thể: 7.387 truy vấn tăng hạng, 4.620 giảm hạng, 1.463 giữ nguyên. Với 11.530 truy vấn, ảnh đúng không có trong
top 100 của CLIP, nên re-rank không cứu được.

| Nhóm | Truy vấn | Hạng CLIP → CLIP + Graph | Hạng khi ảnh đúng bị bỏ mọi cạnh |
|---|---|---|---|
| Thành công | "man in a grey shirt and blue jeans" | 99 → 1 | 7 |
| Thành công | "Blue jacket with orange stripes" | 100 → 3 | 5 |
| Thành công | "part of a trouser" | 100 → 3 | 1 |
| Thành công | "a girl wearing blue jeans." | 99 → 3 | 3 |
| Thành công | "the grass beside the car" | 98 → 3 | 8 |
| Thất bại | "scattered waves in water" | 7 → 100 | 96 |
| Thất bại | "tall green and tan grass" | 15 → 96 | 94 |
| Thất bại | "wheel of car" | 15 → 96 | 82 |
| Thất bại | "a wooden sign" | 12 → 92 | 34 |
| Thất bại | "handicap parking sign on metal post" | 15 → 95 | 75 |

Cột cuối: chỉ ảnh đúng mất cạnh, các ảnh khác giữ nguyên cạnh. Ảnh hạng 1 của cả 5 truy vấn thất bại vẫn hạng 1 khi bị
bỏ mọi cạnh.

Chi tiết từng truy vấn (3 cạnh đẩy lên, 3 cạnh kéo xuống, kiểm tra xoá): `clip_graph_vg_lr1e4_val_seed0/analysis.md`.

## 4. Một ví dụ đọc giải thích

Truy vấn "Blue jacket with orange stripes", ảnh đúng 2326205, hạng 100 → 3:

| Cạnh (khái niệm → ảnh) | Đóng góp vào điểm cuối |
|---|---|
| `ski_run.n.01` → ảnh | +0,0017 (đẩy lên) |
| `ski_resort.n.01` → ảnh | +0,0016 |
| `downhill.n.01` → ảnh | +0,0014 |
| `ski_cap.n.01` → ảnh | −0,0041 (kéo xuống) |

- Kiểm tra xoá: bỏ 3 cạnh đẩy lên thì điểm đồ thị giảm 0,1771 → 0,1717, hạng tụt 3 → 5, đúng hướng mong đợi.
- Cách đọc: các khái niệm về trượt tuyết đẩy ảnh lên một chút. Đây là suy từ **tên khái niệm**; chưa mở ảnh 2326205
  để xem (máy chạy chỉ có đặc trưng CLIP, không có file ảnh).
- Bỏ mọi cạnh: hạng 3 → 5. Không có cạnh, ảnh vẫn đứng hạng 5 (CLIP xếp 100), nên phần lớn cú nhảy đến từ vector của
  chính ảnh. Cạnh chỉ thêm 2 bậc.

## 5. Nhận xét (nháp, cần nhóm kiểm lại)

1. **Thành công chủ yếu đến từ vector của chính ảnh (phép chiếu học được), không phải từ cạnh.**
   - Bằng chứng là phép "bỏ mọi cạnh": 5 ảnh thành công vẫn đứng hạng 7 / 5 / 1 / 3 / 8, so với 99–100 theo CLIP.
   - Đóng góp từng cạnh (0,001–0,004) và "xoá 3" **không** đủ để kết luận điều này. Mô hình lấy trung bình 16 cạnh,
     nên bỏ 1 hay 3 cạnh luôn chỉ đổi điểm rất ít, dù nhánh láng giềng mạnh đến đâu.
   - Cạnh giúp 3/5 ca (7 → 1, 5 → 3, 8 → 3), không đổi 1 ca, và kéo xuống 1 ca ("part of a trouser": 1 → 3).
   - Điều này khớp với ablation: đồ thị chỉ thêm khoảng +0,0018 MRR so với đối chứng không đồ thị.
     Xem `docs/guides/huan-luyen-va-xep-hang-lai.md`, mục ablation.
2. **Thất bại chủ yếu vì `alpha = 0` và vector của ảnh; cạnh làm nặng thêm ở một số ca.**
   - `alpha = 0` bỏ hạng tử cosine CLIP khỏi điểm cuối. Vector CLIP của ảnh vẫn có trong điểm đồ thị qua `W_self`, nên
     không nói "điểm CLIP bị bỏ hoàn toàn".
   - Ảnh đúng vốn hạng 7–15 theo CLIP, có cạnh thì rơi xuống 92–100. Bỏ mọi cạnh thì vẫn ở 96 / 94 / 82 / 34 / 75.
   - Cạnh kéo xuống thêm rõ nhất ở "a wooden sign" (34 → 92), "handicap parking sign…" (75 → 95), "wheel of car" (82 → 96).
   - Ảnh hạng 1 của cả 5 truy vấn thất bại vẫn hạng 1 khi bỏ mọi cạnh: chúng thắng nhờ vector của chính ảnh. Vì vậy giải
     thích bằng cạnh chỉ giải thích được một phần các ca thất bại.
   - Một `alpha` cố định cho mọi truy vấn là điểm yếu. Đây là hướng cải tiến cấp độ 4: kết hợp điểm thích nghi theo truy vấn.
3. **Cạnh đúng nghĩa có khi lại kéo ảnh xuống.**
   - Ví dụ: với "tall green and tan grass", cạnh `grass.n.01` → ảnh đúng có đóng góp −0,0031.
   - Lý do: mô hình lấy **trung bình** 16 khái niệm láng giềng rồi tính cosine với truy vấn. Thêm hay bớt một cạnh làm
     đổi cả hướng của vector trung bình, nên đóng góp không cộng dồn và không luôn cùng chiều với nghĩa của khái niệm.
4. **Cạnh lấy từ CLIP top-16 nên có nhiễu.**
   - Có những khái niệm không liên quan: `screen_saver.n.01`, `body_waste.n.01`, `adam.n.01`, `amour_propre.n.01`.
   - Lý do: cạnh `contains` lấy từ độ tương đồng CLIP, không lấy từ annotation, để không "đọc đáp án" trên val/test
     (ADR-0046, mục A2). Giải thích phản ánh đúng đồ thị mà mô hình đã dùng, kể cả phần nhiễu đó.

## 6. Giới hạn

- Chỉ chạy 1 seed (0) và trên val. Chưa có fidelity so với xoá **ngẫu nhiên**, chưa có validity/sparsity tổng hợp
  trên ≥ 10 truy vấn × 3 seed, và chưa chọn kích thước giải thích (`--num-edges 3`) trên val bằng fidelity. Các phần
  này thuộc cấp độ 3 (ADR-0016, `.claude/rules/explanations.md`).
- Truy vấn ở đây là region description (ADR-0043), không phải bộ truy vấn trực tiếp/gián tiếp trong `queries/`
  (bộ này chưa có).
- `region_descriptions.json` (v1.2) được tải từ trang Visual Genome. sha256 trên máy đã chạy là
  `19624f10afb10bf69e30ecf48d85b1a71420c7432e8c88f1011a3c4fd60ec768`.
