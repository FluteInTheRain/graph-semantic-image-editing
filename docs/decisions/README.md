# Mục lục quyết định (ADR)

Thư mục này chứa các **ADR** (Architecture Decision Record): mỗi file ghi lại **một** quyết định quan trọng của dự án, gồm bối cảnh, lựa chọn, các phương án đã cân nhắc và hệ quả. Mục đích là để cả nhóm tra cứu nhanh "vì sao repo lại như thế này", và để trả lời được khi giảng viên vấn đáp (PDF §9).

- Cách viết ADR mới: xem template trong [ADR-0001](0001-ghi-lai-quyet-dinh-kien-truc-bang-adr.md), hoặc gọi skill `/new-adr` trong Claude Code.
- ADR đã chấp nhận **không sửa nội dung**. Muốn đổi quyết định thì viết ADR mới và đánh dấu ADR cũ là "Bị thay thế bởi ADR-XXXX".
- Nhớ thêm một dòng vào bảng bên dưới mỗi khi tạo ADR mới.

## Quy ước đánh số

| Khoảng số | Chủ đề |
|---|---|
| 0001–0019 | Cấu trúc repo, môi trường, dữ liệu, thực nghiệm, quy trình nhóm |
| 0020–0039 | Thiết lập công cụ AI (Claude Code: CLAUDE.md, hook, quyền, subagent, skill, MCP) |
| 0040 trở đi | Các quyết định phát sinh khi implement (dữ liệu, cách dựng cạnh, mô hình, metric...) |

## Thứ tự đọc gợi ý cho người mới

1. [0001](0001-ghi-lai-quyet-dinh-kien-truc-bang-adr.md): ADR là gì, cách viết.
2. [0003](0003-src-layout-va-cau-truc-package-theo-pipeline.md): repo chia thư mục thế nào, mỗi package làm bước nào của pipeline.
3. [0009](0009-chong-ro-ri-du-lieu-va-split-co-dinh.md) và [0016](0016-giai-thich-trung-thuc-tu-co-che-xep-hang.md): hai luật quan trọng nhất của đề, chống rò rỉ dữ liệu và giải thích trung thực.
4. [0002](0002-python-3-11-va-uv.md), [0006](0006-nhom-phu-thuoc-nhe-va-nang.md): cài môi trường.
5. [0014](0014-quy-trinh-git-va-commit.md): quy trình làm việc nhóm với Git.
6. Các ADR còn lại khi cần.

## Cấu trúc repo và môi trường (0001–0019)

| ADR | Quyết định |
|---|---|
| [0001](0001-ghi-lai-quyet-dinh-kien-truc-bang-adr.md) | Ghi mọi quyết định quan trọng bằng ADR trong `docs/decisions/`, có template chung |
| [0002](0002-python-3-11-va-uv.md) | Python 3.11 (ghim trong `.python-version`), quản lý môi trường bằng uv, commit `uv.lock` |
| [0003](0003-src-layout-va-cau-truc-package-theo-pipeline.md) | src layout, package `mmxai` chia subpackage theo từng bước pipeline |
| [0004](0004-ruff-va-pytest.md) | ruff để lint + format, pytest để test; test mặc định chạy được khi chưa cài torch |
| [0005](0005-stack-pytorch-pyg-open-clip-faiss.md) | Stack: PyTorch + PyTorch Geometric + open_clip (đóng băng) + FAISS |
| [0006](0006-nhom-phu-thuoc-nhe-va-nang.md) | Phụ thuộc mặc định nhẹ; thư viện nặng tách nhóm `ml`, `graph`, `nlp`, `demo` |
| [0007](0007-cau-hinh-yaml-va-dataclass.md) | Cấu hình thí nghiệm bằng YAML thuần + dataclass, không dùng Hydra/OmegaConf |
| [0008](0008-luu-tru-du-lieu-va-artifact.md) | Dữ liệu và artifact lớn để ngoài git (link cloud); cache đặc trưng dạng `.npy` + JSON |
| [0009](0009-chong-ro-ri-du-lieu-va-split-co-dinh.md) | Chống rò rỉ: file split cố định được commit trong `splits/`, kèm test tự động |
| [0010](0010-theo-doi-thi-nghiem-bang-file.md) | Theo dõi thí nghiệm bằng file JSON/CSV, không dùng W&B/MLflow |
| [0011](0011-tai-lap-va-chinh-sach-seed.md) | Tái lập: seed [0, 1, 2], báo cáo mean ± std, ghi lại môi trường chạy |
| [0012](0012-chinh-sach-notebook.md) | Notebook mỏng, chỉ gọi code trong `mmxai`; xoá output bằng nbstripout |
| [0013](0013-demo-notebook-va-gradio.md) | Demo chính bằng notebook, giao diện Gradio là tuỳ chọn |
| [0014](0014-quy-trinh-git-va-commit.md) | Git: nhánh ngắn, Conventional Commits, PR có review, không squash |
| [0015](0015-mmkg-tieng-viet-va-data-card.md) | MMKG tiếng Việt lưu dạng CSV trong `mmkg/`, data card trong `data_cards/` |
| [0016](0016-giai-thich-trung-thuc-tu-co-che-xep-hang.md) | Giải thích phải tính trên chính mô hình xếp hạng và được kiểm bằng fidelity, không hợp lý hoá sau khi đã có kết quả |

## Thiết lập công cụ AI (0020–0039)

| ADR | Quyết định |
|---|---|
| [0020](0020-claude-md-tieng-viet-va-rules-theo-duong-dan.md) | Một `CLAUDE.md` tiếng Việt ngắn gọn; quy tắc chi tiết đặt trong `.claude/rules/`, chỉ nạp theo đường dẫn file |
| [0021](0021-chinh-sach-quyen-cho-claude-code.md) | Chính sách quyền dùng chung trong `.claude/settings.json` (allow / ask / deny) |
| [0022](0022-hook-tu-dong-format-python-bang-ruff.md) | Hook PostToolUse tự chạy ruff trên file `.py` sau mỗi lần AI sửa |
| [0023](0023-hook-bao-ve-split-query-va-chan-file-lon.md) | Hook chặn sửa file split/query đã commit và chặn ghi file lớn hoặc nhị phân vào repo |
| [0024](0024-subagent-review-theo-de-bai.md) | Bốn subagent review: rò rỉ dữ liệu, tính trung thực của giải thích, thực nghiệm, tìm bài báo |
| [0025](0025-skill-du-an-new-adr-run-experiment-log-ai-usage.md) | Ba skill dự án: `/new-adr`, `/run-experiment`, `/log-ai-usage` |
| [0026](0026-mcp-context7-arxiv-va-khong-bat-plugin-cap-du-an.md) | MCP cấp dự án gồm context7 (tài liệu thư viện) và arxiv; không bật plugin ở cấp dự án |
| [0027](0027-nhat-ky-su-dung-ai.md) | Ghi nhật ký dùng công cụ AI trong `docs/ai-usage-log.md` để khai báo trong báo cáo |

## Quyết định khi cài đặt (0040 trở đi)

| ADR | Quyết định |
|---|---|
| [0040](0040-chon-visual-genome-va-flickr30k.md) | Dùng Visual Genome và Flickr30k + ConceptNet làm hai tập công khai |
| [0041](0041-dinh-dang-embedding-npz-va-chia-se-qua-kaggle.md) | Lưu embedding dạng `.npz` + `meta.json`, chia sẻ qua Kaggle, notebook chứa bản sao code |
| [0042](0042-split-co-dinh-cho-visual-genome.md) | Chia Visual Genome theo ảnh: test 5.000, val 5.000, train phần còn lại; ảnh trùng luôn cùng tập |
| [0043](0043-giao-thuc-danh-gia-clip-thuan-tren-visual-genome.md) | Đánh giá văn bản → ảnh: truy vấn là region description, mọi ảnh có cùng câu đều là đáp án, FAISS top 100, Recall@1/5/10 + MRR@100 |
| [0044](0044-bo-truy-van-danh-gia-tu-scene-graph-visual-genome.md) | (Đề xuất) Sinh cặp truy vấn trực tiếp/gián tiếp từ chuỗi quan hệ thật của scene graph Visual Genome; đáp án là mọi ảnh test có cùng chuỗi; chỉ cặp được hai thành viên kiểm tra mới vào bộ chính thức |
| [0045](0045-do-thi-anh-khai-niem-canh-contains-cho-visual-genome.md) | (Đề xuất) Đồ thị Visual Genome bản đầu: đỉnh ảnh + đỉnh khái niệm (synset, ≥ 5 ảnh train), chỉ cạnh `contains`; vector đỉnh lấy từ CLIP đóng băng |
| [0046](0046-gnn-mot-lop-huan-luyen-cho-truy-van-anh-voi-canh-clip-top-k.md) | (Đề xuất) GNN scorer: một lớp GraphSAGE (khái niệm → ảnh), phân rã được để kiểm explainer, huấn luyện tương phản trên region description của train với hard negative từ CLIP top 100; cạnh `contains` lấy từ CLIP top-k cho mọi split, cạnh annotation chỉ làm ablation cận trên |
| [0047](0047-quy-trinh-huan-luyen-chon-alpha-va-chay-test-cho-clip-graphsage.md) | (Đề xuất) CLIP + GraphSAGE: early stopping và chọn `alpha` trên val bằng MRR@100 (hoà thì `alpha` lớn hơn), test chạy một lần bằng script riêng không chọn gì, đối chứng không đồ thị `W_nb = 0` |
| [0048](0048-canh-dong-xuat-hien-giua-khai-niem-dem-tren-train-va-graphsage-hai-lop.md) | (Đề xuất) Cạnh khái niệm → khái niệm từ đồng xuất hiện (PMI, ≥ 10 ảnh, top 10) đếm trên annotation train, mang thông tin ngoài CLIP; GraphSAGE hai lớp tuyến tính, huấn luyện và chọn như ADR-0047 |
| [0049](0049-giao-thuc-danh-gia-v2-ndcg-anh-sang-van-ban-nhom-truy-van.md) | (Đề xuất) Giao thức đánh giá v2: nDCG@10, chiều ảnh → văn bản (`α` chọn trên val), nhóm truy vấn (riêng/chung; hành động/vị trí/không quan hệ); chạy lại 5 experiment dưới tên `_eval2` theo 5 điều kiện, báo cáo mọi số, giữ dòng cũ trong `main.csv` |
| [0050](0050-mien-di-tich-di-san-viet-nam-cho-mmkg-thu-thap-tu-wikidata-commons.md) | MMKG tiếng Việt cấp độ 3: miền di tích và di sản Việt Nam (Wikidata `P1435`, `P17 = Q881`); thu thập tự động thực thể/triple từ Wikidata, ảnh có giấy phép từ Wikimedia Commons, thành viên duyệt trước khi dùng |
| [0051](0051-g-retriever-chon-do-thi-con-pcst-de-xep-hang-lai.md) | (Đề xuất) Phương pháp cấp độ 2: tái lập phần truy xuất đồ thị con của G-Retriever (PCST trên cạnh PMI train, giải thưởng theo hạng, phí cạnh hằng số), điểm ảnh = tỉ lệ chạm đồ thị con (bậc 1) rồi GraphSAGE trên đồ thị con (bậc 2); thêm `pcst-fast` |
| [0052](0052-luu-snapshot-goc-loc-o-buoc-dung-va-duyet-rieng-cho-mmkg.md) | (Đề xuất) MMKG tiếng Việt: snapshot gốc `mmkg/raw/<YYYYMMDD>/` commit nguyên trạng, bộ chính thức dựng lại bằng `scripts/build_mmkg.py` theo luật trong `configs/mmkg_vi_build.yaml` (3–5 ảnh mỗi di tích, khái niệm ≥ 3 di tích, giấy phép PD/CC0/CC BY/CC BY-SA, 18 mã quan hệ), kết quả duyệt lưu riêng trong `mmkg/review.csv`; SPARQL qua QLever, ảnh qua Commons API |
| [0053](0053-danh-gia-chuyen-giao-mo-hinh-vg-sang-mmkg-tieng-viet-khong-huan-luyen-lai.md) | (Đề xuất) Chuyển giao cấp độ 3: dùng nguyên 2 mô hình GNN và `alpha` của Visual Genome trên MMKG tiếng Việt, không huấn luyện lại; 119 thực thể làm đỉnh khái niệm, cạnh CLIP top-16, triple MMKG làm cạnh khái niệm cho mô hình hai lớp; 318 truy vấn (trực tiếp/gián tiếp × Anh/Việt), đáp án = mọi hiện vật khớp mô tả |
| [0054](0054-danh-gia-giai-thich-dinh-luong-tren-mmkg-tieng-viet.md) | (Đề xuất) Cấp độ 3: đánh giá giải thích định lượng trên MMKG tiếng Việt, dùng nguyên lựa chọn của VG (Integrated Gradients, `m = 10`, GNN một lớp, `alpha` của VG); toàn bộ 318 truy vấn, ảnh hạng 1, chạy một lần |
| [0055](0055-tap-du-lieu-thu-hai-flickr30k-chay-lai-pipeline-cap-do-1.md) | (Đề xuất) Tập thứ hai Flickr30k: chạy lại 4 experiment cấp độ 1 y nguyên, split cùng quy trình ADR-0042, đỉnh khái niệm chép từ đồ thị VG, cạnh CLIP top-16; không chạy mô hình đồng xuất hiện |
| [0056](0056-g-retriever-tren-flickr30k-dung-lai-canh-pmi-cua-vg.md) | G-Retriever bậc 2 trên Flickr30k: dùng lại cạnh PMI đếm trên train VG (từ vựng khái niệm chép từ VG nên khớp hàng), thêm trường `concept_edges_split_file`, siêu tham số giữ nguyên bản VG, kèm đối chứng không đồ thị |
| [0057](0057-truy-van-gian-tiep-mo-ta-khai-niem-qua-conceptnet.md) | Truy vấn gián tiếp v2: giấu một khái niệm (activity `playing` hoặc object) và mô tả nó bằng 1 từ loại + 2 tính chất ConceptNet trỏ duy nhất tới nó; đáp án là ảnh test có annotation của khái niệm; thay định nghĩa câu gián tiếp của ADR-0044 |
