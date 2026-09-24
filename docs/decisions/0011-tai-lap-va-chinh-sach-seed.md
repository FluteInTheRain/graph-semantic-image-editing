# ADR-0011: Chính sách tái lập: seed cố định, ≥ 3 seed, ghi môi trường

- **Trạng thái:** Đã chấp nhận
- **Ngày:** 2026-11-01

## Bối cảnh

**Seed** = số khởi tạo cho bộ sinh số ngẫu nhiên. Cùng seed → cùng khởi tạo trọng số, cùng thứ tự batch → kết quả gần như giống nhau. GNN nhỏ khá nhạy với khởi tạo, nên một lần chạy có thể "may mắn".

Đề (§6, §7): chạy tối thiểu 3 seed cho thí nghiệm chính, báo cáo trung bình và độ lệch chuẩn; §9: phải chạy lại được thí nghiệm nhỏ tại buổi chấm.

## Quyết định

| Hạng mục | Quy định |
|---|---|
| Seed mặc định | `[0, 1, 2]` (ghi trong config, khoá `seeds`) |
| Hàm seed | Một hàm duy nhất trong `src/mmxai/utils/` seed `random`, `numpy`, `torch` (+ CUDA nếu có) |
| Deterministic | Có tuỳ chọn `torch.use_deterministic_algorithms(True, warn_only=True)`; bật khi chạy số cuối |
| Seed chia dữ liệu | **Tách riêng** seed mô hình: split cố định (ADR-0009), seed chỉ thay đổi huấn luyện |
| Báo cáo | mean ± std (độ lệch chuẩn mẫu, `ddof=1`) qua các seed, ghi rõ số seed |
| Ghi môi trường | `env.json` mỗi run: git commit, phiên bản Python/torch/PyG/open_clip/faiss, thiết bị |
| Thí nghiệm nhỏ cho buổi chấm | Một config "mini" (vài trăm ảnh, chạy < 5 phút trên CPU) để demo tái lập |

Lưu ý: CLIP đóng băng + FAISS Flat là tất định; nguồn ngẫu nhiên chủ yếu ở huấn luyện GNN và ablation "cạnh ngẫu nhiên" (cũng phải seed).

## Các phương án đã cân nhắc

| Phương án | Ưu | Nhược | Lý do không chọn |
|---|---|---|---|
| 3 seed cố định + deterministic tuỳ chọn | Đáp ứng đề, chi phí vừa | Deterministic có thể chậm hơn | **Được chọn** |
| 5–10 seed | Ước lượng std tốt hơn | Tốn thời gian GPU gấp 2–3 lần | Có thể nâng nếu dư thời gian |
| Một seed | Nhanh | Vi phạm §6 | Không đạt yêu cầu |
| Deterministic luôn bật | Tái lập chặt | Một số op trên MPS/CUDA không hỗ trợ, chậm | Dùng `warn_only` và bật khi cần |
| `lightning.seed_everything` | Có sẵn | Thêm phụ thuộc lớn chỉ để seed | Tự viết vài dòng là đủ |

## Hệ quả

- Kết quả giữa Mac (MPS) và Colab (CUDA) có thể lệch nhỏ ở chữ số thập phân; ghi thiết bị vào `env.json` và báo cáo số từ cùng một loại máy.
- Mọi kết quả trong báo cáo đều có dạng `x.xx ± y.yy (n=3)`.

## Nguồn tham khảo

- PyTorch Reproducibility: https://pytorch.org/docs/stable/notes/randomness.html
- Đề bài, §6, §7, §9.
