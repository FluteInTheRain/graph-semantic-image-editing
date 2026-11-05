# CLAUDE.md – hướng dẫn cho trợ lý AI trong repo này

Đồ án môn "Học máy với dữ liệu đồ thị" – Project 3: truy vấn văn bản → ảnh (text-to-image) bằng CLIP
đóng băng + FAISS top-k, xếp hạng lại bằng điểm đồ thị từ GNN, kèm giải thích **trung thực** bằng
đường đi / đồ thị con. Nhóm ≤ 5 người, **mỗi thành viên bị vấn đáp và phải giải thích được mọi dòng code**.

## Giai đoạn hiện tại

- Đã xong phần setup (cấu trúc repo, môi trường, công cụ AI). Phần cài đặt sẽ làm **từng bước** sau.
- KHÔNG tự ý scaffold nhiều code khi chưa được yêu cầu. Làm đúng phạm vi task được giao, mỗi lần một
  thay đổi nhỏ, dễ review (một hàm / một module + test của nó), rồi dừng lại cho người duyệt.
- Chưa rõ yêu cầu → hỏi lại, không đoán rồi viết cả loạt file.

## Nguồn sự thật

1. `Project03_Multimodal_XAI.pdf` (đề bài gốc) – thắng mọi tài liệu khác khi mâu thuẫn.
2. `README.md` – giải thích đề cho người mới, checklist cấp độ, bản đồ repo.
3. `docs/decisions/` – ADR: vì sao repo được làm như vậy. Đọc ADR liên quan trước khi đổi thứ nó quy định.

## Lệnh thường dùng

```bash
uv sync                                  # cài môi trường nhẹ (dev: ruff, pytest)
uv sync --group ml --group graph         # thêm torch, open_clip, faiss, PyG (xem ADR-0006)
uv run ruff check                        # lint
uv run ruff format                       # format
uv run pytest                            # test
uv run python scripts/<tên>.py --config configs/<file>.yaml   # chạy một bước pipeline
```

Hook tự chạy `ruff check --fix` + `ruff format` trên file `.py` vừa sửa (`.claude/hooks/`). Trước khi
báo xong vẫn phải chạy `uv run ruff check` và `uv run pytest`, và dán kết quả thật.

## Bố cục repo

- `src/mmxai/` – package chính, chia theo pipeline: `data/` (split, loader), `features/` (CLIP),
  `index/` (FAISS), `graph/` (dựng đồ thị), `models/` (GNN), `rerank/` (kết hợp điểm),
  `eval/` (metric), `explain/` (giải thích + fidelity), `utils/` (config, seed).
- `scripts/` – CLI mỏng, chỉ đọc tham số rồi gọi hàm trong `src/mmxai/`. `configs/` – YAML, một file = một thí nghiệm.
- `splits/`, `queries/` – file split và bộ truy vấn đánh giá **đã commit, đóng băng** (hook chặn sửa).
- `data/` (gitignored) – dữ liệu thô. `artifacts/` (gitignored) – đặc trưng, index, checkpoint. `results/runs/` (gitignored) – output từng lần chạy;
  `results/tables/`, `results/figures/` – bảng/hình tổng hợp được commit.
- `tests/`, `notebooks/`, `app/` (demo), `mmkg/` + `data_cards/` (MMKG tự xây, cấp độ 3), `reports/`.
- `docs/decisions/` – ADR. `docs/ai-usage-log.md` – nhật ký dùng công cụ AI.

## Quy ước code

- Python 3.11, type hints cho hàm public, docstring ngắn. Code, docstring, comment viết **tiếng Anh**.
- **Viết để vấn đáp được**: code đơn giản, dễ đọc hơn là ngắn; không one-liner "thông minh", không
  metaprogramming, không abstraction thừa. Comment giải thích **vì sao**, không lặp lại code làm gì.
- Logic nằm trong `src/mmxai/`, có test trong `tests/`. Test dùng dữ liệu tổng hợp nhỏ, không tải mạng.
- Không hard-code đường dẫn hay siêu tham số: đọc từ `configs/*.yaml`. Không thêm dependency
  (`uv add`) khi chưa được đồng ý – thêm dependency là quyết định cần ADR.
- Ưu tiên thư viện đã chọn (ADR-0005): PyTorch, PyTorch Geometric, open_clip, FAISS.

## Rò rỉ dữ liệu – checklist BẮT BUỘC (PDF §6)

Trước khi viết/sửa code về dữ liệu, đồ thị, huấn luyện hay đánh giá, kiểm từng mục:

- [ ] Ảnh ứng viên và truy vấn của **test** không dùng để huấn luyện (lớp chiếu, GNN), chọn siêu
      tham số, hay xây cạnh thống kê.
- [ ] Cạnh suy từ thống kê (đồng xuất hiện, tần suất) chỉ đếm trên **train**.
- [ ] `alpha` và **mọi** siêu tham số chọn trên **validation**. Test chỉ chạy một lần với cấu hình đã chốt.
- [ ] Bộ ≥ 20 truy vấn cấp độ 2 trong `queries/` **chỉ để đánh giá**: không tinh chỉnh, không chọn
      ngưỡng, không debug-đến-khi-đẹp trên bộ này.
- [ ] Mọi mô hình so sánh (CLIP thuần, CLIP + Graph, phương pháp mới, ablation) dùng **cùng file split** trong `splits/`.
- [ ] Thực nghiệm chính chạy **≥ 3 seed**, báo cáo **mean ± std**.

Chi tiết + test bắt buộc: ADR-0009. Nghi ngờ rò rỉ → dừng lại, báo người dùng, đề xuất gọi subagent `leakage-auditor`.

## Giải thích phải trung thực (PDF §2, §5, §7)

- Giải thích (quan hệ / đường đi / đồ thị con) phải lấy **từ chính cơ chế tính điểm đồ thị**. Cấm sinh
  một đường đi "hợp lý" sau khi đã có kết quả nếu đường đi đó không ảnh hưởng tới điểm.
- Mỗi giải thích phải kiểm được bằng **fidelity**: bỏ/làm nhiễu các cạnh được dùng làm giải thích thì
  điểm hoặc thứ hạng phải thay đổi theo hướng hợp lý. Kèm validity và sparsity (cấp độ 3).
- `explain/` chỉ đọc bản ghi đóng góp do graph scorer trả về; không tự tìm đường đi mới (ADR-0016).
- Code giải thích cần test chứng minh điều trên. Chi tiết: `.claude/rules/explanations.md`.

## Tái lập (reproducibility)

- Seed mặc định `[0, 1, 2]`, std mẫu (`ddof=1`), một hàm seed duy nhất trong `src/mmxai/utils/` (ADR-0011).
- Mỗi lần chạy ghi `results/runs/<YYYYMMDD-HHMM>_<experiment>_seed<k>/` gồm `config.yaml`, `metrics.json`,
  `env.json` (commit, phiên bản); số trong báo cáo chỉ lấy từ `results/tables/` (ADR-0010).
- Split và bộ truy vấn được commit; muốn đổi → ADR mới + chạy lại mọi mô hình.
- CLIP / mô hình thị giác: **đóng băng, chỉ inference**. Đặc trưng trích **một lần** rồi cache trong
  `artifacts/`. Không fine-tune CLIP, không huấn luyện bộ phát hiện đối tượng. Chỉ mô hình đồ thị được huấn luyện.

## Quyết định và tài liệu

- Mọi quyết định quan trọng (dữ liệu, cách dựng cạnh, mô hình, metric, dependency, đổi split) →
  **ADR mới trong `docs/decisions/` theo template ADR-0001**. Dùng skill `/new-adr`.
- Tài liệu trong repo viết bằng tiếng Việt; giữ nguyên thuật ngữ kỹ thuật/tên code bằng tiếng Anh;
  code, docstring, comment viết tiếng Anh. Viết cho người mới: giải thích khái niệm trước, chi tiết sau.

## Khai báo dùng AI (PDF §9)

- Sau mỗi phần việc đáng kể do AI hỗ trợ, thêm một dòng vào `docs/ai-usage-log.md` (skill `/log-ai-usage`).
- Cột "Đã tự kiểm tra/hiểu chưa" do **thành viên** điền; AI luôn ghi `Chưa` và nhắc họ tự kiểm tra.
- Khi viết code, giải thích ngắn gọn cho người dùng ý tưởng chính để họ trả lời được khi vấn đáp.

## Công cụ AI có sẵn trong repo

- Subagent (`.claude/agents/`): `leakage-auditor`, `explanation-faithfulness-reviewer`,
  `experiment-reviewer`, `paper-researcher`. Gọi reviewer trước khi mở PR chạm vào data/graph/eval/explain.
- Skill (`.claude/skills/`): `/new-adr`, `/run-experiment`, `/log-ai-usage`.
- MCP (`.mcp.json`): `context7` (tài liệu thư viện mới nhất: PyG, open_clip, FAISS), `arxiv` (tìm/đọc bài báo).
- Ghi chú cá nhân (tên thành viên, máy, GPU) để trong `CLAUDE.local.md` (gitignored), không sửa file này.

## Không được làm

- Không đọc `.env` hay token; không commit dữ liệu, checkpoint, đặc trưng (`*.pt`, `*.npy`, `*.faiss`...).
- Không `git push --force`, không push thẳng `main`; làm trên nhánh `<loại>/<mô-tả>`, Conventional Commits,
  merge qua PR, không squash (ADR-0014).
- Không báo "đã xong / test pass" khi chưa chạy lệnh và thấy output.
