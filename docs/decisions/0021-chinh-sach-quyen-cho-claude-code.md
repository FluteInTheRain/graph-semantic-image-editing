# ADR-0021: Chính sách quyền (permissions) dùng chung trong `.claude/settings.json`

- **Trạng thái:** Đề xuất
- **Ngày:** 2026-11-01

## Bối cảnh

**Permissions là gì?** Claude Code hỏi người dùng trước khi chạy lệnh shell, sửa file, hay truy cập web.
File `.claude/settings.json` (commit vào repo) cho phép định nghĩa sẵn ba danh sách:
- `allow`: chạy không cần hỏi.
- `ask`: luôn hỏi.
- `deny`: luôn chặn.

Thứ tự xét là **deny → ask → allow**; luật đầu tiên khớp quyết định. Cú pháp `Bash(git diff *)` khớp
mọi lệnh bắt đầu bằng `git diff` (dấu cách trước `*` là quan trọng). Claude Code hiểu lệnh ghép
(`&&`, `|`...): mỗi lệnh con phải được cho phép riêng.

Mục tiêu: bớt hỏi với lệnh an toàn hằng ngày (test, lint, xem git), nhưng chặn việc nguy hiểm hoặc
làm lộ bí mật (token Hugging Face, Kaggle, `.env`).

## Quyết định

- **allow**: `uv run pytest`, `uv run ruff check|format`, `uv sync`, `uv lock --check`, `uv tree`,
  `uv pip list`, `git status|diff|log|show|add|commit`, `git branch --show-current`, `tree`, `jq`,
  `nvidia-smi`; WebFetch tới các domain tài liệu/bài báo (arxiv.org, huggingface.co, github.com,
  raw.githubusercontent.com, pytorch-geometric.readthedocs.io, docs.pytorch.org, docs.astral.sh,
  aclanthology.org, openreview.net, code.claude.com).
- **ask**: `git push`, `git reset --hard`, `git clean`, `git checkout --`, `uv add`, `uv remove`,
  `pip install`, `uv pip install` (thêm dependency là quyết định cần ADR).
- **deny**: đọc/sửa `.env`, `.env.*`, `secrets/**`, `~/.cache/huggingface/token`,
  `~/.kaggle/kaggle.json`, `~/.ssh/**`; `cat .env*`; `rm -rf *`; `git push --force|-f|--force-with-lease`.
- Chạy script huấn luyện/đánh giá (`uv run python scripts/...`) **không** nằm trong allow: lệnh có
  thể chạy lâu, tốn GPU, nên để người dùng duyệt từng lần.
- Ngoại lệ cá nhân đặt trong `.claude/settings.local.json` (gitignored).

## Các phương án đã cân nhắc

| Phương án | Ưu | Nhược | Lý do không chọn |
|---|---|---|---|
| Allowlist vừa phải + deny bí mật/lệnh phá huỷ | Ít hỏi với việc thường ngày, vẫn an toàn | Cần cập nhật khi có lệnh mới | **Được chọn** |
| Không cấu hình gì | Đơn giản | Hỏi liên tục; không chặn đọc token | Mất thời gian, rủi ro lộ token |
| Allow rộng (`Bash(uv *)`, `Bash(git *)`) | Gần như không hỏi | Cho phép cả `git push`, `uv add` không kiểm soát | Quá rộng cho repo nhóm |
| Deny `git push` hoàn toàn | An toàn nhất | Cản quy trình PR bình thường | Dùng `ask` là đủ |

## Hệ quả

- Lưu ý: luật Bash khớp theo **chuỗi lệnh**, không phải ranh giới bảo mật tuyệt đối (ví dụ
  `/bin/rm -rf` không khớp `Bash(rm -rf *)`). Muốn chặn cứng phải dùng sandbox hoặc hook.
- `deny` có hiệu lực ngay; `allow` trong file commit chỉ áp dụng sau khi mỗi thành viên "trust" thư mục.
- `rm -rf` bị chặn cả với AI: muốn dọn thư mục, người dùng tự chạy trong terminal.

## Nguồn tham khảo

- Configure permissions: https://code.claude.com/docs/en/permissions
- Settings files and precedence: https://code.claude.com/docs/en/settings
