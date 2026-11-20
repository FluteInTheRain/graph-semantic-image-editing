# ADR-0050: Dùng miền "di tích và di sản Việt Nam" cho MMKG tiếng Việt, thu thập tự động từ Wikidata + Wikimedia Commons rồi nhóm duyệt

- **Trạng thái:** Đã chấp nhận
- **Ngày:** 2026-11-16

## Bối cảnh

Cấp độ 3 (PDF §5) yêu cầu **tự xây một MMKG nhỏ cho một miền hẹp, ưu tiên tiếng Việt**, quy mô gợi ý 50–150 thực thể, 300–1.000 triple, 200–500 ảnh; chạy phương pháp tốt nhất của cấp độ 1/2 trên dữ liệu này, mô tả nguồn, cách liên kết ảnh – thực thể và phân tích khả năng chuyển giao. Khi nộp phải kèm data card, nguồn ảnh/văn bản, cách tạo triple, điều kiện sử dụng (PDF §8). Dữ liệu tự thu thập phải ghi nguồn, tuân thủ giấy phép, không thu thập dữ liệu cá nhân nhạy cảm (PDF §9). Quan hệ giữa thực thể được phép lấy từ KG có sẵn như Wikidata/ConceptNet (PDF §4.1).

ADR-0015 đã chốt **định dạng** (CSV trong `mmkg/`, data card `data_cards/mmkg_vi.md`, ảnh ưu tiên Wikimedia Commons) nhưng **chưa chốt miền** và **cách thu thập**. Tài liệu chỉ nêu ví dụ "món ăn Việt, di tích" (`README.md` §16).

Khái niệm cho người mới:

- **Wikidata** = kho tri thức mở, mỗi thực thể có mã `Q...` (vd. `Q881` = Việt Nam), mỗi quan hệ có mã `P...` (vd. `P131` = nằm trong đơn vị hành chính). Một cặp (thực thể, quan hệ, thực thể) chính là một **triple**.
- **Wikimedia Commons** = kho ảnh mở; mỗi file có tác giả và giấy phép rõ (CC0, CC-BY, CC-BY-SA...). Thực thể Wikidata trỏ tới ảnh qua `P18` (ảnh đại diện) và `P373` (danh mục Commons chứa nhiều ảnh).
- **SPARQL** = ngôn ngữ truy vấn của Wikidata, dùng để đếm và lấy thực thể.

Nhóm quyết định **viết script thu thập tự động, sau đó thành viên duyệt** (FluteInTheRain, 2026-11-16), nên miền phải đủ dữ liệu trên Wikidata/Commons để đạt quy mô gợi ý mà không phải viết tay phần lớn.

## Quyết định

1. **Miền:** di tích và di sản ở Việt Nam = thực thể Wikidata có danh hiệu di sản bất kỳ (`P1435`) và quốc gia là Việt Nam (`P17 = Q881`). Nhóm này gồm cả công trình (chùa, thành, lăng) và hiện vật (bảo vật quốc gia như trống đồng, tượng).
2. **Nguồn:**
   - Thực thể và triple: **Wikidata** (nhãn tiếng Việt `rdfs:label@vi`, nhãn tiếng Anh cho cột `name_en`).
   - Ảnh: **Wikimedia Commons**, từ `P18` và danh mục `P373` của từng thực thể. Chỉ nhận file có giấy phép CC0, CC-BY, CC-BY-SA hoặc public domain; ghi `license` + `author` + `source_url` từng ảnh (ADR-0015 quy tắc 3).
3. **Cách làm:** script thu thập sinh ra `mmkg/entities.csv`, `mmkg/triples.csv`, `mmkg/images.csv` ở trạng thái **chưa duyệt**; thành viên duyệt (bỏ ảnh sai thực thể, ảnh có mặt người nhận diện được, triple sai; sửa nhãn tiếng Việt) trước khi dùng. Phần duyệt và chỉnh sửa được ghi trong data card: đây là phần "tự xây" của nhóm.
4. **Quy mô mục tiêu sau duyệt:** 50–150 thực thể, 300–1.000 triple, 200–500 ảnh (tối đa khoảng 5 ảnh mỗi thực thể).
5. **Phương án dự phòng** nếu sau duyệt không đủ quy mô hoặc miền bị đánh giá là quá rộng: chỉ giữ công trình kiến trúc của nhóm này và bổ sung từ "công trình tôn giáo ở Việt Nam" (`Q24398318`).

Các chi tiết **chưa** chốt trong ADR này (để design/ADR sau): danh sách quan hệ Wikidata được giữ và tên `relation` snake_case tương ứng, luật chọn ảnh khi một danh mục có nhiều file, cách sinh truy vấn trên MMKG, ngôn ngữ truy vấn (CLIP hiện dùng chỉ hiểu tiếng Anh), split `splits/mmkg_vi/`.

## Các phương án đã cân nhắc

Số đo ngày 2026-11-16 bằng SPARQL trên `query.wikidata.org` và Commons API (`prop=categoryinfo`). Cột "Triple" đếm quan hệ tới thực thể khác, **bỏ** `P17` (quốc gia), `P31` (loại), `P495` (nguồn gốc) vì thực thể nào cũng có. Cột "Ảnh" = tổng của min(số file trong danh mục, 5).

| Miền (mẫu Wikidata) | Thực thể | Có `P18` | Nhãn `vi` | Có `P373` | Triple | Ảnh (≤5/thực thể) |
|---|---|---|---|---|---|---|
| Di tích, di sản (`P1435`, `P17 = Q881`) | 170 | 118 | 164 | 98 | 2.071 (87 loại quan hệ) | ~440 |
| Công trình tôn giáo (`Q24398318`, `P17 = Q881`) | 444 | 188 | 423 | 207 | 1.035 (33 loại) | ~890 |
| Cầu (`Q12280`) | 268 | 96 | 257 | 74 | 847 | ~310 |
| Khu bảo tồn, vườn quốc gia (`Q473972`) | 66 | 38 | 55 | 25 | 250 | ~110 |
| Món ăn nguồn gốc Việt Nam (`Q746549`, `P495 = Q881`) | 38 | 29 | 38 | 13 | 111 | ~50 |
| Lễ hội / loài đặc hữu / nhạc cụ | 25 / 22 / 1 | 6 / 11 / 1 | | | 45 / 211 / 14 | rất ít |
| Núi / đảo | 3.378 / 579 | 17 / 53 | | | | rất ít |

| Phương án | Ưu | Nhược | Lý do không chọn |
|---|---|---|---|
| **Di tích và di sản** | Đạt cả 3 mốc quy mô chủ yếu bằng dữ liệu tự động; quan hệ phong phú nhất (địa điểm `P131`/`P276`, danh hiệu `P1435`, chất liệu `P186`, niên đại `P2348`, khắc hoạ `P180`, bộ sưu tập `P195`), đủ cho truy vấn gián tiếp; gần như mọi thực thể có nhãn tiếng Việt | Trộn công trình với hiện vật nên miền hơi rộng; ảnh chùa/đình giống nhau, CLIP khó phân biệt | **Được chọn** |
| Công trình tôn giáo | Miền hẹp, đồng nhất; nhiều ảnh nhất (~890) | Triple chủ yếu là "nằm ở tỉnh nào", "tôn giáo nào", đồ thị gần hình sao, ít đường đi nhiều bước; ảnh rất giống nhau | Giữ làm phương án dự phòng (Quyết định mục 5) |
| Cầu | Quan hệ có nghĩa (bắc qua sông `P177`, loại cầu); ~310 ảnh | Chỉ 74 thực thể có danh mục ảnh; gắn văn hoá Việt yếu, khó bảo vệ "ưu tiên tiếng Việt" | Kém hơn di tích ở mọi tiêu chí trừ độ đồng nhất |
| Món ăn Việt Nam | CLIP nhận món ăn tốt; dễ hiểu khi trình bày | Wikidata chỉ 38 món (< 50 thực thể tối thiểu), ~50 ảnh; Commons "Cuisine of Vietnam" có 589 file nhưng xếp theo nhóm món, không theo thực thể, nên triple (nguyên liệu, vùng miền) phải viết tay gần hết | Không thu thập tự động được, trái với cách làm đã chọn |
| Khu bảo tồn, lễ hội, loài đặc hữu, nhạc cụ, núi, đảo | Có nhãn tiếng Việt | Quá ít thực thể hoặc quá ít ảnh | Không đạt quy mô |
| Tự chụp ảnh / tự viết toàn bộ triple | Toàn quyền giấy phép, "tự xây" rõ nhất | Tốn rất nhiều công, dễ lẫn ảnh có người, khó đạt 200–500 ảnh trong thời gian môn học | Chi phí quá cao |

## Hệ quả

- **Dễ hơn:** nguồn và giấy phép từng ảnh lấy được tự động từ Commons, nên data card và yêu cầu PDF §9 đáp ứng gần như sẵn; triple có nguồn (`source` = URL thực thể Wikidata) nên kiểm lại được.
- **Khó hơn:**
  - Cần trả lời được câu hỏi vấn đáp "phần tự xây nằm ở đâu": data card phải ghi rõ luật lọc, số ảnh/triple bị bỏ hoặc sửa khi duyệt, và ai duyệt.
  - Ảnh công trình giống nhau, kết quả CLIP trên MMKG có thể thấp. Đây là một phần của phân tích chuyển giao, không phải lý do đổi miền.
  - Wikidata thay đổi theo thời gian: script phải ghi ngày truy vấn, và CSV đã duyệt được commit để kết quả không phụ thuộc lần tải sau.
- **Rò rỉ dữ liệu (ADR-0009):** khi dựng đồ thị, nếu cạnh `ảnh → contains → thực thể` của ảnh **test** lấy từ `images.csv` (ảnh thuộc thực thể nào) thì cạnh đó chính là đáp án. Cách dựng cạnh cho MMKG phải theo cùng nguyên tắc ADR-0046 (không đọc nhãn của ảnh val/test). Cần quyết định ở ADR sau và gọi `leakage-auditor` khi cài đặt.
- **Giải thích trung thực (ADR-0016):** mô hình chính `ConceptToImageSage` một lớp chỉ dùng cạnh khái niệm → ảnh; triple thực thể – thực thể của MMKG chỉ ảnh hưởng điểm (và do đó mới được xuất hiện trong giải thích) nếu dùng mô hình có cạnh khái niệm – khái niệm như `CooccurrenceSage` (ADR-0048). Cần quyết định khi áp dụng phần đánh giá giải thích lên MMKG.
- **Việc tiếp theo:** (1) thiết kế script thu thập và luật lọc; (2) chạy, sinh CSV chưa duyệt; (3) thành viên duyệt; (4) viết `data_cards/mmkg_vi.md`; (5) ADR cho split, cách dựng cạnh và truy vấn trên MMKG; (6) chạy mô hình tốt nhất và đánh giá giải thích.

## Nguồn tham khảo

- Đề bài `Project03_Multimodal_XAI.pdf`: §4.1 (nguồn quan hệ, không huấn luyện detector), §5 cấp độ 3, §8 (data card), §9 (giấy phép, dữ liệu cá nhân).
- ADR-0015 (định dạng MMKG), ADR-0009 (chống rò rỉ), ADR-0016 (giải thích trung thực), ADR-0046, ADR-0048.
- Wikidata Query Service: https://query.wikidata.org/
- Commons API `categoryinfo`: https://commons.wikimedia.org/w/api.php
- Commons licensing: https://commons.wikimedia.org/wiki/Commons:Licensing

### Phụ lục: truy vấn dùng để đo (chạy lại được)

Đếm thực thể, ảnh, nhãn tiếng Việt, danh mục Commons (thay `<MẪU>` bằng mẫu ở bảng trên, ví dụ `?x wdt:P1435 ?h; wdt:P17 wd:Q881.`):

```sparql
SELECT (COUNT(DISTINCT ?x) AS ?n) (COUNT(DISTINCT ?xi) AS ?img)
       (COUNT(DISTINCT ?xv) AS ?vi) (COUNT(DISTINCT ?xc) AS ?cat) WHERE {
  <MẪU>
  OPTIONAL { ?x wdt:P18 ?i. BIND(?x AS ?xi) }
  OPTIONAL { ?x rdfs:label ?l FILTER(lang(?l) = "vi") BIND(?x AS ?xv) }
  OPTIONAL { ?x wdt:P373 ?c. BIND(?x AS ?xc) }
}
```

Đếm triple tới thực thể khác:

```sparql
SELECT (COUNT(*) AS ?t) (COUNT(DISTINCT ?p) AS ?props) WHERE {
  <MẪU>
  ?x ?p ?o. ?o wdt:P31 ?any.
  FILTER(STRSTARTS(STR(?p), "http://www.wikidata.org/prop/direct/"))
  FILTER(?p NOT IN (wdt:P17, wdt:P31, wdt:P495, wdt:P1343, wdt:P5008, wdt:P6104))
}
```

Số ảnh: lấy mọi `P373` của miền, gọi `action=query&prop=categoryinfo&titles=Category:<tên>|...` theo lô 50 danh mục, cộng `min(files, 5)`.
