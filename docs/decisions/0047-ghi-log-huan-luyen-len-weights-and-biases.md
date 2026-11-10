# ADR-0047: Ghi log quá trình huấn luyện GNN lên Weights & Biases (tuỳ chọn, không bao giờ làm hỏng lần chạy)

- **Trạng thái:** Đề xuất
- **Ngày:** 2026-11-10

## Bối cảnh

Huấn luyện GNN của ADR-0046 chạy lâu: đo trên CPU 14 core khoảng 9 phút một epoch, vài giờ cho 3 seed. Hiện tiến trình chỉ in ra terminal và lưu `history` vào `artifacts/checkpoints/<experiment>_seed<k>/meta.json` **sau khi** chạy xong. Trong lúc chạy, cả nhóm không xem được đường loss, cũng không so được các seed trên cùng một biểu đồ.

**Weights & Biases (W&B, thư viện `wandb`)** là dịch vụ lưu log thí nghiệm trên cloud. Mỗi lần chạy gửi các con số (loss, metric) lên một trang web, nơi vẽ biểu đồ và so sánh các lần chạy. Thành viên đã dùng nó ở dự án VaLiK (`repro/common.py`, hàm `wandb_run`).

Thêm dependency là quyết định cần ADR (`CLAUDE.md`, ADR-0006).

## Quyết định

| Thành phần | Quy định |
|---|---|
| Dependency | `wandb>=0.19` trong nhóm **tuỳ chọn** mới `tracking` (`pyproject.toml`). Cài bằng `--group tracking`. Code chạy được khi không có nhóm này |
| Bật / tắt | Bật khi config có `wandb_project` (mặc định `mmxai-gnn` trong `configs/gnn_vg.yaml`). Tắt khi `wandb_project: null`, khi `WANDB_MODE=disabled`, khi chưa cài `wandb`, hoặc khi không kết nối được |
| Không bao giờ làm hỏng lần chạy | Mọi lỗi của W&B chỉ in một dòng cảnh báo. Lần train vẫn chạy y hệt và vẫn lưu checkpoint (`src/mmxai/utils/tracking.py`, `start_run` trả về `NoRun`) |
| Log gì | Mỗi epoch: `train/loss`, `val/MRR@100`; mỗi 50 bước: `train/batch_loss`; cuối lần chạy: `best_epoch`, `best_val/MRR@100`; toàn bộ config + seed. Tên run: `<experiment>_seed<k>` |
| Đăng nhập | `uv run --group tracking wandb login` một lần, hoặc biến môi trường `WANDB_API_KEY`. `WANDB_ENTITY` chọn team. **Không** ghi key vào repo |
| Nguồn sự thật | Vẫn là `meta.json` của checkpoint và `results/runs/` (ADR-0010). W&B chỉ để theo dõi; số trong báo cáo **không** lấy từ W&B |

Quyết định bởi: thiennp3 yêu cầu tích hợp W&B, 2026-11-10. Cách làm (nhóm tuỳ chọn, không bao giờ làm hỏng lần chạy, danh sách số liệu log) do Claude Code đề xuất theo mẫu VaLiK, **nhóm chưa duyệt**.

## Các phương án đã cân nhắc

| Phương án | Ưu | Nhược | Lý do không chọn |
|---|---|---|---|
| **W&B, nhóm tuỳ chọn, không bao giờ làm hỏng lần chạy** | Xem tiến trình trực tiếp trên cloud, so được các seed; thành viên đã quen | Cần tài khoản, cần mạng | **Được chọn** |
| W&B trong nhóm `ml` (bắt buộc khi train) | Lệnh cài ngắn hơn | Ai train cũng phải cài và đăng nhập, kể cả máy không có mạng | Không bắt buộc ai |
| TensorBoard | Chạy local, không cần tài khoản | Không tự đồng bộ lên cloud để cả nhóm xem | Yêu cầu là đẩy log lên cloud |
| MLflow | Mã nguồn mở, tự host được | Phải tự dựng server để chia sẻ | Tốn công hơn cho nhóm nhỏ |
| Chỉ in terminal + `meta.json` (như cũ) | Không thêm gì | Không xem được khi đang chạy, khó so seed | Không đáp ứng yêu cầu |

## Hệ quả

- **Rò rỉ dữ liệu (PDF §6):** không đổi. Chỉ log số liệu train và val đã có; không log gì của test.
- **Tái lập (ADR-0010, ADR-0011):** không đổi; W&B không ảnh hưởng tới seed hay kết quả. Test `tests/test_utils_tracking.py` kiểm tra tắt được.
- **Bảo mật:** API key nằm trong `~/.netrc` (do `wandb login` tạo) hoặc biến môi trường, không nằm trong repo. Thư mục log cục bộ `wandb/` đã được thêm vào `.gitignore`.
- **Lệnh chạy:** train kèm log dùng `uv run --group ml --group graph --group tracking python scripts/train.py` (xem `docs/guides/chay-pipeline.md`).

## Nguồn tham khảo

- ADR-0006 (nhóm phụ thuộc), ADR-0010 (thư mục run là nguồn sự thật), ADR-0046 (mô hình được huấn luyện).
- VaLiK, `repro/common.py`, hàm `wandb_run` (mẫu "không bao giờ làm hỏng lần chạy").
- Tài liệu `wandb.init`: https://docs.wandb.ai/ref/python/init
