# ADR-0005: Stack kỹ thuật PyTorch + PyTorch Geometric + open_clip + FAISS

- **Trạng thái:** Đã chấp nhận
- **Ngày:** 2026-11-01

## Bối cảnh

Hệ thống có bốn khối kỹ thuật, mỗi khối cần một thư viện:

| Khối | Việc | Khái niệm cho người mới |
|---|---|---|
| Encoder | Biến ảnh và câu văn thành vectơ cùng không gian | **CLIP**: mô hình học ghép ảnh với mô tả; ảnh và câu giống nghĩa → vectơ gần nhau |
| Tìm kiếm vectơ | Lấy top-k ảnh gần truy vấn nhất | **FAISS**: thư viện tìm láng giềng gần rất nhanh |
| Mô hình đồ thị | Học biểu diễn đỉnh từ cạnh | **GNN** (GCN/GAT/GraphSAGE): mỗi đỉnh tổng hợp thông tin từ đỉnh kề |
| Nền tảng | Tensor, huấn luyện, GPU | **PyTorch** |

Đề (§4.1, §10) gợi ý đúng các thư viện: open_clip, sentence-transformers, PyTorch Geometric, FAISS. Đề cũng yêu cầu CLIP **đóng băng** (không huấn luyện lại) và **không huấn luyện bộ phát hiện đối tượng**.

## Quyết định

| Khối | Thư viện | Cách dùng |
|---|---|---|
| Nền tảng | `torch` | CPU/MPS trên Mac, CUDA trên Colab |
| CLIP | `open-clip-torch` | Mặc định `ViT-B-32` (`laion2b_s34b_b79k`); chỉ inference, `model.eval()` + `torch.no_grad()` |
| Tìm kiếm | `faiss-cpu` | `IndexFlatIP` trên vectơ đã chuẩn hoá L2 (= cosine chính xác). Dữ liệu ≤ vài trăm nghìn ảnh nên không cần index xấp xỉ |
| GNN | `torch-geometric` | `GCNConv`, `GATConv`, `SAGEConv`; chỉ cài gói lõi, **không** cài `pyg-lib`/`torch-scatter` trừ khi thật cần |
| Đồ thị phụ trợ | `networkx` | Tìm đường đi ngắn nhất để hiển thị/kiểm tra giải thích |
| Text encoder phụ | `sentence-transformers` | Tuỳ chọn, để liên kết khái niệm ConceptNet |

Các thư viện này nằm trong nhóm phụ thuộc tuỳ chọn, không cài mặc định (ADR-0006).

## Các phương án đã cân nhắc

| Phương án | Ưu | Nhược | Lý do không chọn |
|---|---|---|---|
| PyTorch + PyG | Đề gợi ý, tài liệu nhiều, có sẵn GCN/GAT/SAGE | Extension biên dịch (pyg-lib) khó cài trên Mac | **Được chọn**, chỉ dùng gói lõi thuần Python |
| DGL | Mạnh cho đồ thị dị thể | Bảo trì chậm lại, wheel macOS hạn chế | Ít tài liệu theo đề, rủi ro cài đặt |
| open_clip | Nhiều checkpoint (OpenAI, LAION), đề gợi ý | — | **Được chọn** |
| HF `transformers` CLIP | Quen thuộc | Ít checkpoint LAION hơn | Không cần thêm phụ thuộc |
| FAISS | Chuẩn ngành, đề gợi ý | `faiss-gpu` khó cài | **Được chọn** bản CPU, đủ nhanh |
| Annoy / hnswlib / numpy brute force | Nhẹ | Kết quả xấp xỉ hoặc chậm khi lớn | FAISS Flat vừa chính xác vừa nhanh |

## Hệ quả

- Đặc trưng CLIP trích **một lần**, lưu cache (ADR-0008); GNN huấn luyện trên vectơ đã cache → chạy được cả trên laptop.
- Model CLIP mặc định ghi trong config; đổi model = đổi config, không đổi code.
- Phương pháp cấp độ 2 (NativE, AdaMF-MAT...) có thể cần phụ thuộc riêng: thêm nhóm mới + ADR mới.

## Nguồn tham khảo

- OpenCLIP: https://github.com/mlfoundations/open_clip
- PyG installation: https://pytorch-geometric.readthedocs.io/en/stable/notes/installation.html
- FAISS: https://github.com/facebookresearch/faiss/wiki
- Đề bài, §4.1 và §10.
