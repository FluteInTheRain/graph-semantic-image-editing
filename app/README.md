# app/ — giao diện nhỏ (tuỳ chọn)

**Là gì?** Chỗ đặt một UI Gradio nhỏ (`app/gradio_app.py`) nếu nhóm muốn demo đẹp hơn notebook.

- Bắt buộc theo đề chỉ là "notebook **hoặc** giao diện nhỏ". Notebook demo là bản chính; UI là phần thêm.
- UI chỉ gọi hàm trong `src/mmxai/`, không chứa logic riêng.
- Lý do chọn Gradio: `docs/decisions/0013-demo-notebook-va-gradio.md`.
