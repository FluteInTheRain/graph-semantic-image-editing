# ADR-0014: Quy trình Git: nhánh ngắn, Conventional Commits, PR có review, không squash

- **Trạng thái:** Đã chấp nhận
- **Ngày:** 2026-11-01

## Bối cảnh

Đề (§9): điểm cá nhân dựa trên vấn đáp **và minh chứng đóng góp (lịch sử commit, bảng phân công)**. Vậy lịch sử git phải:
- cho thấy **ai làm gì** (tác giả commit đúng người);
- dễ đọc (thông điệp commit rõ ràng);
- không bị "gộp mất" đóng góp cá nhân.

Nhóm ≤ 5 người, một học kỳ, không cần quy trình phức tạp như Git Flow.

## Quyết định

| Hạng mục | Quy định |
|---|---|
| Mô hình nhánh | GitHub Flow: `main` luôn chạy được; mọi việc làm trên nhánh ngắn rồi mở PR |
| Tên nhánh | `<loại>/<mô-tả-ngắn>`, vd. `feat/faiss-index`, `fix/split-overlap`, `docs/data-card-vg` |
| Commit | Conventional Commits: `<type>(<scope>): <mô tả>`; type ∈ `feat, fix, docs, test, refactor, chore, exp`; scope = subpackage (`data`, `graph`, `eval`...) |
| Kích thước | Commit nhỏ, một ý; PR ≤ ~400 dòng thay đổi |
| Review | Mỗi PR cần ≥ 1 thành viên khác approve; người review cũng phải hiểu code (chuẩn bị vấn đáp) |
| Kiểm tra trước PR | `uv run ruff check`, `uv run ruff format --check`, `uv run pytest` đều xanh |
| Cách merge | **Merge commit** hoặc **rebase merge**; **không dùng squash** (squash gộp mọi commit thành một, mất lịch sử từng người) |
| Danh tính | Mỗi người cấu hình `git config user.name/user.email` khớp tài khoản GitHub; không commit hộ |
| Làm cặp | Thêm dòng `Co-authored-by: Tên <email>` cuối commit |
| Bảo vệ `main` | Bật branch protection: cấm push thẳng, yêu cầu PR + 1 approve |
| Dùng AI | Commit có phần do AI hỗ trợ vẫn do người đó chịu trách nhiệm; khai báo theo quy định nhóm (§9 của đề) |

Ví dụ commit: `feat(index): build FAISS IndexFlatIP from cached CLIP features`.

Chi tiết từng bước cho người mới: `CONTRIBUTING.md`.

## Các phương án đã cân nhắc

| Phương án | Ưu | Nhược | Lý do không chọn |
|---|---|---|---|
| GitHub Flow + PR + merge/rebase | Đơn giản, giữ lịch sử từng người | Cần kỷ luật mở PR | **Được chọn** |
| Push thẳng lên `main` | Nhanh | Dễ làm hỏng `main`, không có review | Rủi ro cao, không có minh chứng review |
| Git Flow (develop/release/hotfix) | Chuẩn cho sản phẩm phát hành | Quá nặng cho đồ án | Không cần release |
| Squash merge | Lịch sử `main` gọn | Gộp nhiều commit của nhiều người thành một | Làm mờ minh chứng đóng góp được chấm |
| Commit tự do (không quy ước) | Không phải học | Lịch sử khó đọc, khó viết bảng phân công | Conventional Commits rẻ và rõ |

## Hệ quả

- Bảng phân công trong báo cáo có thể tạo từ `git shortlog -sn` và lọc theo scope.
- Mỗi người nên có commit ở nhiều phần (code, test, tài liệu) để trả lời vấn đáp tốt.

## Nguồn tham khảo

- Conventional Commits 1.0.0: https://www.conventionalcommits.org/
- GitHub Flow: https://docs.github.com/en/get-started/using-github/github-flow
- GitHub, các cách merge PR: https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/configuring-pull-request-merges/about-merge-methods-on-github
- Đề bài, §9.
