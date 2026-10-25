# ADR-0024: Bốn subagent review bám theo yêu cầu đề bài

- **Trạng thái:** Đề xuất
- **Ngày:** 2026-11-01

## Bối cảnh

**Subagent là gì?** Là một "trợ lý con" với prompt và bộ công cụ riêng, định nghĩa bằng file Markdown
trong `.claude/agents/`. Claude chính giao việc cho nó; nó làm trong context riêng và chỉ trả lại kết
luận. Lợi ích: một người review có checklist cố định, không bị ảnh hưởng bởi lập luận của người viết code.

Những lỗi đắt nhất của đồ án này là lỗi **phương pháp**, không phải lỗi cú pháp: rò rỉ dữ liệu (§6),
giải thích hợp lý hoá sau (§2), báo cáo thiếu seed/ablation (§7). Đây đúng là việc cần checklist.

## Quyết định

Tạo 4 subagent trong `.claude/agents/`, viết tiếng Việt, mô tả rõ khi nào dùng:

| Subagent | Việc | Công cụ |
|---|---|---|
| `leakage-auditor` | Rà diff theo 6 quy tắc PDF §6, trả bảng vi phạm có `file:dòng` | Read, Grep, Glob, Bash; cấm Edit/Write |
| `explanation-faithfulness-reviewer` | Lần theo `final_score` ngược về giải thích; kiểm test fidelity/validity/sparsity | Read, Grep, Glob, Bash; cấm Edit/Write |
| `experiment-reviewer` | Seed ≥ 3, mean ± std, cùng split, metric, ablation, trực tiếp/gián tiếp, alpha trên val | Read, Grep, Glob, Bash; cấm Edit/Write |
| `paper-researcher` | Tìm/tóm tắt NativE, AdaMF-MAT, MKGformer, GNNExplainer...; đánh giá khả năng tái lập | Kế thừa công cụ (web, MCP), cấm Edit/Write |

Không chỉ định `model` để dùng thứ tự model mặc định của Claude Code.

## Các phương án đã cân nhắc

| Phương án | Ưu | Nhược | Lý do không chọn |
|---|---|---|---|
| 4 subagent chuyên biệt theo rủi ro của đề | Checklist rõ, độc lập với người viết | Phải bảo trì 4 file | **Được chọn** |
| Một subagent "reviewer" chung | Ít file | Prompt dài, dễ bỏ sót mục | Kém tập trung |
| Chỉ dùng plugin review có sẵn (code-review) | Không phải viết | Không biết PDF §6, fidelity | Không bám đề |
| Không dùng subagent | Đơn giản | Phụ thuộc trí nhớ người review | Rủi ro cao |

## Hệ quả

- CLAUDE.md dặn gọi reviewer trước khi mở PR chạm vào data/graph/eval/explain.
- Reviewer có Bash nên về kỹ thuật vẫn chạy được lệnh ghi; prompt giới hạn ở lệnh đọc và quyền
  (ADR-0021) vẫn áp dụng.
- Kết quả review là gợi ý; thành viên vẫn phải tự hiểu và quyết định.

## Nguồn tham khảo

- Create custom subagents (vị trí file, frontmatter `tools`, `disallowedTools`, `model`, `color`): https://code.claude.com/docs/en/sub-agents
- Đề bài `Project03_Multimodal_XAI.pdf` §2, §5, §6, §7, §10.
