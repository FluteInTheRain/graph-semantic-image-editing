# Hướng dẫn đóng góp

Tài liệu này dành cho thành viên mới vào repo. Đọc theo thứ tự từ trên xuống. Lý do của từng quy tắc nằm trong các ADR ở `docs/decisions/`.

## Mục lục

1. [Cài môi trường](#1-cài-môi-trường)
2. [Repo có gì](#2-repo-có-gì)
3. [Quy trình làm một việc](#3-quy-trình-làm-một-việc)
4. [Viết commit](#4-viết-commit)
5. [Pull Request và review](#5-pull-request-và-review)
6. [Quy tắc dữ liệu và chống rò rỉ](#6-quy-tắc-dữ-liệu-và-chống-rò-rỉ)
7. [Notebook](#7-notebook)
8. [Ghi lại quyết định (ADR)](#8-ghi-lại-quyết-định-adr)

## 1. Cài môi trường

**uv** là công cụ quản lý môi trường của repo (ADR-0002): tự tải Python 3.11 theo `.python-version`, tạo `.venv`, cài đúng phiên bản thư viện ghi trong `uv.lock`. Không cần cài Python tay, không cần `activate`: mọi lệnh chạy qua `uv run`.

### 1.1. Cài uv (một lần trên mỗi máy)

| Máy | Lệnh |
|---|---|
| macOS | `brew install uv` |
| Linux / macOS (không có brew) | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Windows (PowerShell) | `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 \| iex"` |

Kiểm tra: `uv --version`. Chi tiết: https://docs.astral.sh/uv/getting-started/installation/

### 1.2. Môi trường nhẹ (viết code, tài liệu, chạy test)

```bash
git clone https://github.com/FluteInTheRain/graph-semantic-image-editing.git
cd graph-semantic-image-editing
uv sync                                   # gói nhẹ + công cụ dev (vài giây)
uv run pytest                             # phải xanh
uv run pre-commit install                 # bật hook: ruff, nbstripout, chặn file lớn
git config user.name  "Tên Thật"          # khớp tài khoản GitHub (commit được chấm điểm)
git config user.email "email@github"
```

### 1.3. Thư viện ML (chạy pipeline thật)

Thư viện nặng được tách thành **nhóm** (dependency group, ADR-0006), chỉ cài khi cần:

| Nhóm | Gói chính | Cần cho |
|---|---|---|
| `ml` | `torch`, `open-clip-torch`, `faiss-cpu` | Trích đặc trưng CLIP, FAISS, mọi bước pipeline |
| `graph` | `torch-geometric`, `networkx` | Đồ thị + GNN (`scripts/setup.py`, `train.py`, `run_pipeline.py`) |
| `tracking` | `wandb` | Log huấn luyện lên Weights & Biases (tuỳ chọn, ADR-0047) |
| `demo` | `jupyterlab`, `matplotlib`, `gradio` | Notebook, UI demo |

Hai cách dùng, chọn một:

```bash
# Cách 1 (khuyên dùng): để uv run tự cài nhóm cần cho lệnh đó
uv run --group ml --group graph python scripts/setup.py

# Cách 2: cài trước rồi chạy
uv sync --group ml --group graph
uv run python scripts/setup.py
```

Lưu ý:

- **`uv sync` cài đúng những nhóm được liệt kê và gỡ các nhóm còn lại.** Ví dụ đã cài `ml` + `graph` mà chạy `uv sync --group ml` thì `torch-geometric` bị gỡ. Luôn liệt kê đủ nhóm, hoặc dùng `uv sync --all-groups`. Cách 1 (`uv run --group ...`) **không** gỡ gói nào, nên an toàn hơn.
- **Lần đầu cài `ml` rất lâu**: trên Linux, `torch` kèm thư viện CUDA của NVIDIA, tải vài GB (đo ngày 2026-11-10: hơn 15 phút, cache uv khoảng 5,5 GB). Trong lúc cài, `.venv` bị dựng lại nên `ruff` / `pytest` tạm thời không chạy được; chờ cài xong.
- **Không có GPU vẫn chạy được** toàn bộ pipeline trên CPU (xem `docs/guides/chay-pipeline.md` mục 4 cho thời gian).
- **Google Colab / Kaggle:** torch có sẵn, chỉ chạy `pip install -e .` và cài gói còn thiếu; không cài lại torch.

### 1.4. W&B (chỉ khi muốn xem log huấn luyện trên cloud)

```bash
uv run --group tracking wandb login       # dán API key từ https://wandb.ai/authorize; lưu ở ~/.netrc
```

Hoặc đặt biến môi trường `WANDB_API_KEY`. **Không** ghi key vào repo. `WANDB_ENTITY` chọn team; `WANDB_MODE=disabled` tắt log.

### 1.5. Lỗi thường gặp khi cài

| Lỗi | Nguyên nhân | Cách sửa |
|---|---|---|
| `invalid peer certificate: UnknownIssuer` | Mạng công ty / proxy có chứng chỉ riêng; uv mặc định không dùng chứng chỉ của hệ điều hành | Thêm `UV_NATIVE_TLS=1` trước lệnh: `UV_NATIVE_TLS=1 uv sync --group ml --group graph` |
| `No module named 'torch'` (hoặc `faiss`, `torch_geometric`) | Chưa cài nhóm, hoặc lần `uv sync` sau đã gỡ nhóm đó | Chạy lại với đủ nhóm: `uv sync --group ml --group graph` |
| `401 Unauthorized` từ Hugging Face khi tải CLIP | Token HF trên máy hỏng / hết hạn | `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` trước lệnh (model CLIP công khai) |
| Test bị `SKIPPED ... could not import 'torch'` | Bình thường khi chỉ có môi trường nhẹ | Cài nhóm `ml` + `graph` nếu cần chạy các test đó |

## 2. Repo có gì

| Đường dẫn | Vai trò |
|---|---|
| `src/mmxai/` | Code chính, mỗi thư mục con là một bước pipeline (ADR-0003) |
| `tests/` | pytest |
| `configs/` | YAML thí nghiệm (ADR-0007) |
| `scripts/` | Lệnh chạy pipeline, gọi vào `mmxai` |
| `notebooks/`, `app/` | Demo (ADR-0012, ADR-0013) |
| `splits/`, `queries/` | File chia dữ liệu, bộ truy vấn đánh giá (ADR-0009) |
| `mmkg/`, `data_cards/` | MMKG tiếng Việt, data card (ADR-0015) |
| `results/`, `reports/` | Bảng kết quả, báo cáo, slide (ADR-0010) |
| `data/`, `artifacts/` | Dữ liệu thô và cache — **không commit** (ADR-0008) |
| `docs/decisions/` | ADR |

## 3. Quy trình làm một việc

```bash
git switch main && git pull
git switch -c feat/faiss-index            # nhánh mới: <loại>/<mô-tả>
# ... sửa code + viết test ...
uv run ruff format && uv run ruff check && uv run pytest
git add -p && git commit                  # commit nhỏ, một ý
git push -u origin feat/faiss-index       # rồi mở Pull Request trên GitHub
```

Loại nhánh: `feat/`, `fix/`, `docs/`, `test/`, `refactor/`, `exp/` (thí nghiệm), `chore/`.

## 4. Viết commit

Theo Conventional Commits (ADR-0014):

```
<type>(<scope>): <mô tả ngắn, thể mệnh lệnh>

<thân: vì sao thay đổi, nếu cần>
Co-authored-by: Tên Bạn <email>          # nếu làm cặp
```

| `type` | Khi nào |
|---|---|
| `feat` | Thêm chức năng |
| `fix` | Sửa lỗi |
| `test` | Thêm/sửa test |
| `docs` | Tài liệu, ADR, data card |
| `refactor` | Sửa cấu trúc, không đổi hành vi |
| `exp` | Config/kết quả thí nghiệm |
| `chore` | Môi trường, phụ thuộc, cấu hình công cụ |

`scope` = tên subpackage hoặc thư mục: `data`, `features`, `index`, `graph`, `models`, `rerank`, `explain`, `eval`, `mmkg`, `queries`, `report`.

Ví dụ: `feat(graph): add co-occurrence edges built from train split only`.

Không làm: commit hộ người khác, commit "update", commit gộp cả tuần.

## 5. Pull Request và review

- PR nhỏ (≤ ~400 dòng), mô tả: làm gì, vì sao, cách kiểm tra.
- Cần **≥ 1 người khác approve**. Người review phải hiểu code — giảng viên có thể hỏi bất kỳ ai về bất kỳ đoạn nào.
- Ba lệnh `uv run ruff check`, `uv run ruff format --check`, `uv run pytest` phải xanh.
- Merge bằng **merge commit** hoặc **rebase**, **không squash** (giữ lịch sử từng người).
- Thêm phụ thuộc: `uv add --group <nhóm> <gói>`, commit `pyproject.toml` + `uv.lock` cùng nhau.

## 6. Quy tắc dữ liệu và chống rò rỉ

| Không bao giờ | Thay vào đó |
|---|---|
| Commit ảnh, `.npy`, `.pt`, chỉ mục FAISS | Để trong `data/` hoặc `artifacts/`, chia sẻ qua link cloud |
| Dùng tập test để huấn luyện, chọn `alpha`, early stopping | Dùng `val` |
| Xây cạnh thống kê từ toàn bộ dữ liệu | Chỉ từ `train` |
| Chỉnh tay file trong `splits/` | Sinh lại bằng script, tạo file mới |
| Tinh chỉnh mô hình trên `queries/` | Bộ truy vấn chỉ để đánh giá |
| Báo cáo số từ một seed | ≥ 3 seed, mean ± std |

Số trong báo cáo chỉ lấy từ `results/tables/`.

## 7. Notebook

- Notebook chỉ gọi hàm trong `mmxai`; logic dài chuyển vào `src/`.
- Output được `nbstripout` xoá khi commit (cần đã chạy `uv run pre-commit install`).
- Mỗi notebook có một người chịu trách nhiệm chính.

## 8. Ghi lại quyết định (ADR)

Quyết định ảnh hưởng nhiều người (thêm thư viện, đổi cấu trúc dữ liệu, đổi cách đánh giá) → viết ADR theo mẫu trong `docs/decisions/0001-ghi-lai-quyet-dinh-kien-truc-bang-adr.md`, mở PR để nhóm review.

Dùng công cụ AI: được phép nhưng phải khai báo trong báo cáo và tự giải thích được toàn bộ (§9 của đề).
