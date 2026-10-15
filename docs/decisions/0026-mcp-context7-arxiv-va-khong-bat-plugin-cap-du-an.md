# ADR-0026: MCP cấp dự án gồm context7 và arxiv; không bật plugin ở cấp dự án

- **Trạng thái:** Đề xuất
- **Ngày:** 2026-11-01

## Bối cảnh

**MCP là gì?** Model Context Protocol – cách cắm thêm công cụ cho Claude (tra tài liệu, tìm bài báo...).
File `.mcp.json` ở gốc repo khai báo server dùng chung cho cả nhóm; mỗi người được hỏi duyệt lần đầu.

**Plugin là gì?** Gói gồm skill/agent/hook/MCP cài từ marketplace. Bật ở cấp dự án bằng
`enabledPlugins` trong `.claude/settings.json`, nhưng mỗi thành viên vẫn phải tự cài
(`claude plugin install ... --scope project`).

Nhu cầu của đồ án: API của PyG, open_clip, FAISS đổi theo phiên bản (AI hay nhớ API cũ); cần tìm/đọc
bài báo (NativE, AdaMF-MAT, MKGformer, GNNExplainer...). Yêu cầu: **không cần secret** để không ai phải
commit token.

## Quyết định

1. `.mcp.json`:
   - `context7` – HTTP `https://mcp.context7.com/mcp`: tài liệu thư viện theo phiên bản. Chạy ẩn danh
     được; API key chỉ để tăng giới hạn (nếu cần, mỗi người tự thêm ở cấp user).
   - `arxiv` – stdio `uvx arxiv-mcp-server@0.8.2` (PyPI `arxiv-mcp-server`, ghim phiên bản): tìm, tải,
     đọc bài arXiv. Không cần API key. Bài tải về lưu ở `~/.arxiv-mcp-server/papers` (ngoài repo).
2. **Không** bật plugin nào trong `enabledPlugins` của dự án. Các plugin hữu ích (pyright-lsp,
   Hugging Face) cần cài thêm binary hoặc đăng nhập → để mỗi người tự cài theo danh sách khuyến nghị.
3. Không thêm Hugging Face MCP (`https://huggingface.co/mcp`) vào `.mcp.json`: cấu hình được sinh từ
   trang cài đặt cần đăng nhập HF → để ở danh sách khuyến nghị.

## Các phương án đã cân nhắc

| Phương án | Ưu | Nhược | Lý do không chọn |
|---|---|---|---|
| context7 + arxiv, không secret | Cả nhóm dùng ngay, không lộ token | Phụ thuộc dịch vụ ngoài | **Được chọn** |
| Thêm Hugging Face MCP | Tìm model/dataset/paper trên Hub | Cần đăng nhập để cấu hình đầy đủ | Để khuyến nghị cá nhân |
| Bật `pyright-lsp@claude-plugins-official` cấp dự án | Kiểm tra kiểu, nhảy định nghĩa | Mỗi người phải cài binary `pyright`; nếu không sẽ báo lỗi plugin | Để khuyến nghị cá nhân |
| Bật plugin `context7` thay vì `.mcp.json` | Cài một lệnh | Mỗi người vẫn phải cài plugin | `.mcp.json` đơn giản hơn |
| Không dùng MCP | Không phụ thuộc ngoài | AI dễ dùng API cũ, bịa trích dẫn | Mất lợi ích lớn |

## Hệ quả

- Lần đầu mở Claude Code trong repo, mỗi người được hỏi duyệt 2 server; `arxiv` cần `uv` (đã là công cụ của nhóm).
- Nội dung trả về từ MCP là dữ liệu bên ngoài: vẫn phải kiểm chứng trước khi đưa vào báo cáo.
- Chưa chạy thử server `arxiv` trong lúc setup (môi trường setup chặn tải và chạy code bên ngoài);
  người đầu tiên dùng hãy kiểm bằng `/mcp`.
- Muốn đổi/ghim lại phiên bản: sửa `.mcp.json` và ghi ADR mới.

## Nguồn tham khảo

- Connect Claude Code to tools via MCP (`.mcp.json`, duyệt server dự án, biến môi trường): https://code.claude.com/docs/en/mcp
- Install and manage plugins (scope dự án, `enabledPlugins`): https://code.claude.com/docs/en/discover-plugins
- Context7: https://github.com/upstash/context7
- arxiv-mcp-server: https://github.com/blazickjp/arxiv-mcp-server và https://pypi.org/project/arxiv-mcp-server/
- Hugging Face MCP Server: https://huggingface.co/docs/hub/en/hf-mcp-server
