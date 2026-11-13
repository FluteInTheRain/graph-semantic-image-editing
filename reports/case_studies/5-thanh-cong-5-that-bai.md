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
- Kiểm tra xoá: bỏ cùng lúc 3 cạnh đẩy ảnh lên mạnh nhất, rồi xem điểm và thứ hạng đổi ra sao.

## 2. Chạy lại

```bash
uv run python scripts/explain.py --config configs/clip_graph_vg_lr1e4.yaml --seed 0 \
    --split val --num-edges 3 --vg-data-dir data/visual_genome
```

| Mục | Giá trị |
|---|---|
| Mô hình | `clip_graph_vg_lr1e4` (GraphSAGE một lớp, ADR-0046), seed 0, `alpha = 0.0` (chọn trên val khi huấn luyện) |
| Tập | **val** (25.000 truy vấn). Script từ chối `test`, vì test chỉ chạy một lần trong `evaluate_graph_rerank.py` |
| Code | commit `fc26470`, `git_dirty: false` (`clip_graph_vg_lr1e4_val_seed0/env.json`) |
| Số liệu gốc | `reports/case_studies/clip_graph_vg_lr1e4_val_seed0/analysis.json` (bản chép từ `results/runs/20261113-1004_clip_graph_vg_lr1e4_explain_seed0/`) |
| Kiểm tra khớp | MRR@100 val của CLIP + Graph trong phân tích = 0,12855, đúng bằng số trong `selection.json` của seed 0 khi huấn luyện |

Checkpoint được huấn luyện lại trên máy này, chỉ dùng train + val. Kết quả val khớp với số mà nhóm đã ghi trong
`docs/guides/huan-luyen-va-xep-hang-lai.md` §7: epoch tốt nhất 6/8/6, `alpha` 0/0/0,1, val MRR@100 0,1281 ± 0,0008.

### Quy tắc chọn truy vấn (cố định trong code, `src/mmxai/eval/query_ranks.py`)

1. Chỉ xét truy vấn có ảnh đúng nằm trong top 100 của CLIP.
2. `change` = hạng theo CLIP − hạng theo CLIP + Graph.
3. Thành công: `change > 0`, lớn nhất trước. Thất bại: `change < 0`, âm nhất trước. Hoà: lấy truy vấn có chỉ số nhỏ hơn.
4. Câu đã được chọn rồi (giống chữ, không phân biệt hoa/thường và khoảng trắng) thì bỏ qua.

Quy tắc 4 được thêm sau lần chạy đầu tiên. Lần đó, câu "part of a trouser" chiếm 3 trong 5 chỗ thành công, vì Visual
Genome có nhiều region trùng câu và các câu trùng có cùng thứ hạng. Quy tắc này chỉ loại câu trùng, không đổi hướng
chọn. Lần chạy đầu vẫn còn trong `results/runs/20261113-1002_*` trên máy đã chạy.

## 3. Bảng tổng quan (val, seed 0)

Tổng thể: 7.387 truy vấn tăng hạng, 4.620 giảm hạng, 1.463 giữ nguyên. Với 11.530 truy vấn, ảnh đúng không có trong
top 100 của CLIP, nên re-rank không cứu được.

| Nhóm | Truy vấn | Hạng CLIP → CLIP + Graph |
|---|---|---|
| Thành công | "man in a grey shirt and blue jeans" | 99 → 1 |
| Thành công | "Blue jacket with orange stripes" | 100 → 3 |
| Thành công | "part of a trouser" | 100 → 3 |
| Thành công | "a girl wearing blue jeans." | 99 → 3 |
| Thành công | "the grass beside the car" | 98 → 3 |
| Thất bại | "scattered waves in water" | 7 → 100 |
| Thất bại | "tall green and tan grass" | 15 → 96 |
| Thất bại | "wheel of car" | 15 → 96 |
| Thất bại | "a wooden sign" | 12 → 92 |
| Thất bại | "handicap parking sign on metal post" | 15 → 95 |

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
- Cách đọc: các khái niệm về trượt tuyết (ảnh chụp người mặc áo khoác trên dốc trượt) đẩy ảnh lên một chút.

## 5. Nhận xét (nháp, cần nhóm kiểm lại)

1. **Thành công chủ yếu đến từ phép chiếu học được, không phải từ cạnh.**
   - Đóng góp của từng cạnh chỉ khoảng 0,001–0,004, trong khi điểm đồ thị của ảnh khoảng 0,07–0,20.
   - Xoá 3 cạnh mạnh nhất làm điểm giảm 0,003–0,008. Thứ hạng chỉ đổi ở 2/5 ca: 3 → 5 và 3 → 4. Ba ca còn lại giữ nguyên hạng.
   - Điều này khớp với ablation: đồ thị chỉ thêm khoảng +0,0018 MRR so với đối chứng không đồ thị.
     Xem `docs/guides/huan-luyen-va-xep-hang-lai.md`, mục ablation.
2. **Thất bại vì `alpha = 0`: điểm CLIP bị bỏ hoàn toàn.**
   - Ảnh đúng vốn đứng hạng 7–15 theo CLIP, nhưng có điểm đồ thị rất thấp (0,001–0,09), nên rơi xuống khoảng hạng 92–100.
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
