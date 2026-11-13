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

Cần: `git` và `uv` (macOS: `brew install uv`). Không cần cài Python tay: uv tự tải Python 3.11 theo `.python-version`.

```bash
git clone <url-repo> && cd project_graph_ml
uv sync                                   # gói nhẹ + công cụ dev (vài giây)
uv run pytest                             # phải xanh
uv run pre-commit install                 # bật hook: ruff, nbstripout, chặn file lớn
git config user.name  "Tên Thật"          # khớp tài khoản GitHub (commit được chấm điểm)
git config user.email "email@github"
```

Cài thêm khi cần chạy pipeline thật (ADR-0006):

| Việc | Lệnh |
|---|---|
| Trích đặc trưng CLIP, FAISS | `uv sync --group ml` |
| Đồ thị + GNN | `uv sync --group ml --group graph` |
| Notebook / UI demo | `uv sync --group ml --group graph --group demo` |
| Tất cả | `uv sync --all-groups` |

Google Colab: torch có sẵn, chỉ chạy `pip install -e .` và cài gói còn thiếu; không cài lại torch.

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
