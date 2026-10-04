# ADR-0013: Demo chính bằng notebook, UI Gradio là tuỳ chọn

- **Trạng thái:** Đã chấp nhận
- **Ngày:** 2026-11-01

## Bối cảnh

Đề (§3): "không bắt buộc xây dựng hệ thống web hoàn chỉnh, một notebook hoặc giao diện nhỏ chạy được và minh hoạ kết quả là đủ". §8: demo chạy trực tiếp khi vấn đáp, chuẩn bị video dự phòng ≤ 5 phút. Buổi trình bày tối đa 20 phút.

Demo cần hiển thị cho mỗi truy vấn:

| Thành phần | Ví dụ |
|---|---|
| Top-k ảnh | 5–10 ảnh thumbnail |
| Ba điểm | `cosine`, `graph`, `final = alpha·cosine + (1−alpha)·graph` |
| Thứ hạng trước/sau re-rank | ảnh nhảy từ hạng 7 lên hạng 2 |
| Giải thích | đường đi `query: child → plays → football ← contains ← image_123` |

## Quyết định

1. **Bản chính**: `notebooks/01_demo_truy_van.ipynb` (ADR-0012). Luôn phải chạy được.
2. **Bản thêm (tuỳ chọn)**: `app/gradio_app.py` — ô nhập truy vấn, thanh trượt `alpha`, gallery ảnh, khung text đường đi. Chỉ làm khi pipeline cấp 1 đã xong.
3. Cả hai chỉ gọi hàm trong `src/mmxai/` (cùng một hàm `search_and_explain(query, k, alpha)`), không chứa logic riêng → demo và đánh giá dùng **cùng một code path**.
4. Demo chạy từ artifact đã cache (đặc trưng, FAISS, checkpoint GNN), không trích đặc trưng lại khi trình bày; chỉ encode câu truy vấn.
5. Quay video dự phòng ≤ 5 phút từ notebook hoặc UI, lưu link trong `README.md`.

Gradio nằm trong nhóm phụ thuộc `demo` (ADR-0006).

## Các phương án đã cân nhắc

| Phương án | Ưu | Nhược | Lý do không chọn |
|---|---|---|---|
| Notebook làm bản chính | Đề chấp nhận, ít code, dễ giải thích từng ô | Giao diện kém trực quan | **Được chọn** |
| Gradio (tuỳ chọn) | Vài chục dòng có gallery + slider, hợp demo ML | Thêm phụ thuộc | **Được chọn** làm phần thêm |
| Streamlit | Phổ biến | Chạy lại toàn script mỗi tương tác, cần cache cẩn thận | Gradio hợp kiểu "input → output" hơn |
| ipywidgets trong notebook | Không thêm app riêng | Hiển thị phụ thuộc môi trường Jupyter | Có thể dùng trong notebook nếu muốn |
| Web app (FastAPI + frontend) | Chuyên nghiệp | Tốn công, đề không yêu cầu | Ngoài phạm vi |

## Hệ quả

- Rủi ro demo thấp: luôn có notebook + video dự phòng.
- Thanh trượt `alpha` trong UI chỉ để minh hoạ; `alpha` báo cáo vẫn là giá trị chọn trên val (ADR-0009).

## Nguồn tham khảo

- Gradio: https://www.gradio.app/docs
- Streamlit: https://docs.streamlit.io/
- Đề bài, §3, §8, §9.
