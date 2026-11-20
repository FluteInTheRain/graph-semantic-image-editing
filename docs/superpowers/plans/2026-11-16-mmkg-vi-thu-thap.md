# Vietnamese Heritage MMKG Collection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collect a dated raw snapshot of Vietnamese heritage entities, triples and candidate images from Wikidata + Wikimedia Commons, and build the official ADR-0015 MMKG files from it with config-driven filter rules and a separately stored member review.

**Architecture:** Three layers, each a small module in `src/mmxai/data/` with a thin script: (1) a polite HTTP client for Wikidata SPARQL and the Commons API; (2) a collector that writes `mmkg/raw/<YYYYMMDD>/` once and never edits it; (3) a pure, deterministic builder that reads the snapshot + `mmkg/review.csv` + `configs/mmkg_vi_build.yaml` and writes `mmkg/{entities,triples,images}.csv` + `build_report.json`. A last module downloads only the selected images and checks their sha1.

**Tech Stack:** Python 3.11 standard library only for new code (`urllib`, `csv`, `json`, `html`, `hashlib`), `pyyaml` via the existing `load_yaml_dataclass`, pytest.

**Spec:** `docs/superpowers/specs/2026-11-16-mmkg-vi-thu-thap-design.md` (approved 2026-11-16). Read it first; this plan does not repeat its reasons. Related: ADR-0015 (format), ADR-0050 (domain, source).

## Global Constraints

- Branch `docs/adr-0050-mmkg-domain` (continue on it; rename is not needed). Conventional Commits, no squash, no push without asking.
- Code, docstrings, comments in English; repo docs (ADR, data card, README) in Vietnamese. Simple code for the oral exam: no one-liners, no metaprogramming.
- Logic in `src/mmxai/data/`, thin CLIs in `scripts/` with the module docstring format of `scripts/make_splits.py`, settings in `configs/*.yaml` loaded by `mmxai.utils.yaml_config.load_yaml_dataclass`.
- **No new dependency.** HTTP via `urllib.request`; CSV via `csv` (UTF-8, `newline=""`); HTML stripping via `re` + `html.unescape`.
- Tests use tiny synthetic data and a fake HTTP opener; **no network in tests**.
- Every file written under `mmkg/` must be < 1 MiB (`max_file_bytes: 1048576`); no `.gz`.
- Domain: Wikidata items with any `P1435` and `P17 = Q881`.
- Allowed licenses: `LicenseShortName` equal to `Public domain` or `CC0`, or starting with `CC BY ` or `CC BY-SA ` (any version). Attribution required for the `CC BY` / `CC BY-SA` ones.
- Allowed MIME types: `image/jpeg`, `image/png`, `image/webp`. Minimum short side 224 px.
- Thresholds: `min_images_per_item: 3`, `max_images_per_item: 5`, `min_items_per_concept: 3`.
- Relation list (P → name), with `P276` and `P195` both → `kept_at`: `P31 instance_of`, `P1435 heritage_designation`, `P276 kept_at`, `P195 kept_at`, `P131 located_in`, `P186 made_from`, `P2348 time_period`, `P2596 culture`, `P180 depicts`, `P189 discovered_at`, `P361 part_of`, `P527 has_part`, `P138 named_after`, `P170 creator`, `P88 commissioned_by`, `P737 influenced_by`, `P5422 has_pattern`, `P793 significant_event`.
- Review reasons for images: `not_the_object`, `identifiable_person`, `duplicate`, `other`. `review_status` is `ok` | `drop` | `pending`.
- User-Agent `mmxai-course-project/0.1 (research script)` (no personal data); 1.5 s between requests; retry HTTP 429 (honour `Retry-After`) and 5xx, at most 5 attempts.
- Before reporting done: `uv run ruff check` and `uv run pytest` clean, real output pasted.

## Review Focus

- A SPARQL query that times out (HTTP 504) should be retried, then fail with a clear error, never return an empty result. → Task 2 test `test_retries_server_errors_then_raises`.
- A Commons category with more than 500 files must follow `continue` until done, not silently stop at 500. → Task 3 test `test_category_files_follow_continuation`.
- The same file must be recognised whether it comes from `P18` (`.../Special:FilePath/Ngoc%20Lu_drum.jpg`) or from a category (`File:Ngoc Lu drum.jpg`), otherwise sharing/duplicate rules miss it. → Task 3 test `test_file_names_from_p18_and_category_match`.
- Vietnamese labels and author strings with commas, quotes, newlines and HTML entities must survive a CSV round trip unchanged. → Task 1 test `test_round_trip_keeps_vietnamese_quotes_and_newlines`.
- An interrupted collect must not leave a half-written snapshot that a later build reads. → Task 3 test `test_write_snapshot_is_all_or_nothing`.

---

### Task 1: CSV helpers with the 1 MiB shard rule

**Files:**
- Create: `src/mmxai/data/mmkg_csv.py`
- Test: `tests/test_mmkg_csv.py`

**Interfaces:**
- Produces:
  - `write_rows(directory: Path, stem: str, header: list[str], rows: list[dict[str, str]], max_file_bytes: int) -> list[Path]` — writes `<stem>.csv`, or `<stem>_000.csv`, `<stem>_001.csv`, ... when one file would reach `max_file_bytes`; every shard repeats the header. Returns the paths written.
  - `read_rows(directory: Path, stem: str) -> list[dict[str, str]]` — reads `<stem>.csv` or all `<stem>_NNN.csv` in order; raises `FileNotFoundError` if neither exists, `ValueError` if both exist.

- [ ] **Step 1: Write the failing tests** in `tests/test_mmkg_csv.py`:
  - `test_small_table_is_one_file`: 3 rows, `max_file_bytes=1_000_000` → returns `[dir/"t.csv"]`, `read_rows` gives the same 3 dicts.
  - `test_large_table_is_sharded_under_limit`: 200 rows of ~100 bytes, `max_file_bytes=2_000` → more than one `t_NNN.csv`, each `stat().st_size < 2_000`, each starts with the header line, `read_rows` returns all 200 rows in order.
  - `test_round_trip_keeps_vietnamese_quotes_and_newlines`: a row `{"name_vi": 'Trống "Ngọc Lũ", đồng\nthế kỷ 3', "author": "A &amp; B"}` → read back equal (the CSV layer does not unescape; that is Task 3's job).
  - `test_read_rejects_both_plain_and_sharded`: create `t.csv` and `t_000.csv` → `ValueError`.

- [ ] **Step 2: Run** `uv run pytest tests/test_mmkg_csv.py -v` → FAIL (module missing).
- [ ] **Step 3: Implement** both functions in `src/mmxai/data/mmkg_csv.py`. Measure size by encoding each formatted line (`csv.writer` to a `StringIO`, then `.encode("utf-8")`); start a new shard before the line that would reach the limit. A single row larger than the limit is a `ValueError`.
- [ ] **Step 4: Run** the tests → PASS.
- [ ] **Step 5: Commit** `feat(mmkg): CSV helpers that keep every file under 1 MiB`.

---

### Task 2: Polite Wikimedia HTTP client

**Files:**
- Create: `src/mmxai/data/wikimedia.py`
- Test: `tests/test_wikimedia.py`

**Interfaces:**
- Produces:
  - `class WikimediaClient` with `__init__(self, user_agent: str, interval_seconds: float, max_attempts: int, opener=urllib.request.urlopen, sleep=time.sleep, clock=time.monotonic)`.
  - `sparql(self, query: str) -> list[dict[str, str]]` — GET `https://query.wikidata.org/sparql` with `Accept: application/sparql-results+json`; returns each binding flattened to `{var: value}`.
  - `commons(self, params: dict[str, str]) -> dict` — GET `https://commons.wikimedia.org/w/api.php` with `format=json` added; raises `RuntimeError` if the JSON has an `error` key.
  - `download(self, url: str) -> bytes`.
  - Constants `SPARQL_URL`, `COMMONS_API_URL`.

- [ ] **Step 1: Write the failing tests** with a fake opener (a callable recording `Request` objects and returning/raising scripted responses) and a fake `sleep` that records waits:
  - `test_sends_user_agent_and_waits_between_requests`: two `commons` calls → both requests carry the given User-Agent; the second call sleeps so that at least `interval_seconds` passed (fake clock).
  - `test_retries_429_using_retry_after`: first response `HTTPError(429, headers {"Retry-After": "7"})`, then success → result returned, a sleep of 7 recorded.
  - `test_retries_server_errors_then_raises`: 5 × `HTTPError(504)` with `max_attempts=5` → raises `RuntimeError` mentioning `504`; never returns `[]`.
  - `test_does_not_retry_client_errors`: `HTTPError(400)` → raised at once, 1 request made.
  - `test_sparql_flattens_bindings`: canned `{"results": {"bindings": [{"x": {"type": "uri", "value": "http://www.wikidata.org/entity/Q1"}}]}}` → `[{"x": "http://www.wikidata.org/entity/Q1"}]`.
- [ ] **Step 2: Run** `uv run pytest tests/test_wikimedia.py -v` → FAIL.
- [ ] **Step 3: Implement** `WikimediaClient`. Retry on 429 and 5xx; wait `Retry-After` seconds when present, else `interval_seconds * 2**attempt`. Every request goes through one private `_get(url) -> bytes` so the interval and retry rules live in one place.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(mmkg): polite Wikidata/Commons client with retry`.

---

### Task 3: Collector and raw snapshot (`scripts/collect_mmkg.py`)

**Files:**
- Create: `src/mmxai/data/mmkg_collect.py`, `scripts/collect_mmkg.py`, `configs/mmkg_vi_collect.yaml`
- Test: `tests/test_mmkg_collect.py`

**Interfaces:**
- Consumes: `WikimediaClient` (Task 2), `write_rows` (Task 1).
- Produces:
  - `@dataclass(frozen=True) class MmkgCollectConfig`: `output_root: str`, `country_qid: str`, `designation_property: str`, `user_agent: str`, `request_interval_seconds: float`, `max_attempts: int`, `max_file_bytes: int`; `load_mmkg_collect_config(path) -> MmkgCollectConfig`.
  - `normalize_file_name(value: str) -> str` — accepts a `P18` URL, `File:...` title or bare name; returns the bare name with spaces (no `File:` prefix, `_` → space, URL-unquoted).
  - `strip_html(value: str) -> str` — removes tags, unescapes entities, collapses whitespace.
  - `@dataclass(frozen=True) class RawSnapshot`: `entities: list[dict[str, str]]`, `triples: list[dict[str, str]]`, `image_candidates: list[dict[str, str]]`, `image_credits: list[dict[str, str]]`, `meta: dict`.
  - `collect_snapshot(client: WikimediaClient, config: MmkgCollectConfig, today: date) -> RawSnapshot`.
  - `write_snapshot(snapshot: RawSnapshot, output_root: Path, max_file_bytes: int) -> Path` — writes `output_root/<YYYYMMDD>/`; refuses if that folder exists.
- Raw columns (exact):
  - `entities_raw`: `entity_id, name_vi, name_en, in_domain` (`1`/`0`), `commons_category`, `p18_files` (`|`-joined normalized names).
  - `triples_raw`: `head_id, property, tail_id` (only item-valued direct statements).
  - `image_candidates`: `file_name, entity_id, origin` (`p18`/`category`), `mime, width, height, license, sha1`.
  - `image_credits`: `file_name, artist, uploader` (one row per file).
  - `meta.json`: `collected_on`, `queries` (SPARQL text used), `row_counts`, `git_commit`, `user_agent`, `config`.

- [ ] **Step 1: Write the failing tests** with a `FakeClient` exposing `sparql(query)` and `commons(params)` that answer from canned data keyed on the query/params:
  - `test_file_names_from_p18_and_category_match`: `normalize_file_name("http://commons.wikimedia.org/wiki/Special:FilePath/Ngoc%20Lu_drum.jpg") == normalize_file_name("File:Ngoc Lu drum.jpg") == "Ngoc Lu drum.jpg"`.
  - `test_strip_html_unescapes`: `strip_html('<a href="x">Nguyễn&nbsp;Văn&amp;A</a>\n') == "Nguyễn Văn&A"`.
  - `test_collect_keeps_targets_and_concept_triples`: 2 domain items (Q1, Q2) → Q1 `P186` Q9, Q2 `P186` Q9, Q9 `P279` Q8 (Q8 not in the entity set) and Q9 `P361` Q1 → `entities` has Q1, Q2 (`in_domain=1`) and Q9 (`in_domain=0`), no Q8; `triples` contains `Q9 P361 Q1` but not `Q9 P279 Q8`.
  - `test_category_files_follow_continuation`: the category answer comes in two pages (`continue.cmcontinue`) → both pages' files appear in `image_candidates`.
  - `test_shared_file_gets_one_row_per_entity`: one file is Q1's `P18` and in Q2's category → two `image_candidates` rows (Q1/`p18`, Q2/`category`), one `image_credits` row.
  - `test_write_snapshot_is_all_or_nothing`: make `write_rows` raise on the third file (monkeypatch) → `output_root/<date>` does not exist afterwards; a second call on an existing folder raises `FileExistsError`.
- [ ] **Step 2: Run** `uv run pytest tests/test_mmkg_collect.py -v` → FAIL.
- [ ] **Step 3: Implement** in `src/mmxai/data/mmkg_collect.py`:
  - Domain query: items with `wdt:<designation_property> ?h; wdt:P17 wd:<country_qid>` plus optional `vi`/`en` labels, `P373`, `P18`.
  - Triples: one query for every direct item-valued statement of domain items (the survey query: `FILTER(STRSTARTS(STR(?p), "http://www.wikidata.org/prop/direct/P"))`, object starts with `http://www.wikidata.org/entity/Q`), with no label service (it timed out in the survey).
  - Target labels and concept-to-concept statements: `VALUES` batches of 150 ids; keep a statement only if both ends are already in the entity set.
  - Images: `list=categorymembers&cmtype=file&cmlimit=500` per category, following `continue`; then `prop=imageinfo&iiprop=extmetadata|size|sha1|mime|user` in batches of 50 titles. `license` = `LicenseShortName`, `artist` = `strip_html(Artist)`.
  - `write_snapshot`: write into `<date>.partial/`, then `Path.rename` to `<date>/`; delete the partial folder on error.
  - `configs/mmkg_vi_collect.yaml`: `output_root: mmkg/raw`, `country_qid: Q881`, `designation_property: P1435`, user agent / interval / attempts / `max_file_bytes: 1048576` from Global Constraints; a `# Decision (ADR-0050)` comment like other configs.
  - `scripts/collect_mmkg.py --config`: builds the client, calls `collect_snapshot` and `write_snapshot`, prints row counts.
- [ ] **Step 4: Run** → PASS; `uv run ruff check`.
- [ ] **Step 5: Commit** `feat(mmkg): collect a raw Wikidata/Commons snapshot`.

---

### Task 4: Builder, image rules

**Files:**
- Create: `src/mmxai/data/mmkg_build.py`
- Test: `tests/test_mmkg_build_images.py`

**Interfaces:**
- Produces (used by Task 5):
  - `@dataclass(frozen=True) class MmkgBuildConfig`: `snapshot_dir: str`, `review_file: str`, `output_dir: str`, `allowed_mime_types: list[str]`, `allowed_license_names: list[str]`, `allowed_license_prefixes: list[str]`, `attribution_license_prefixes: list[str]`, `min_short_side_px: int`, `min_images_per_item: int`, `max_images_per_item: int`, `min_items_per_concept: int`, `relations: dict[str, str]`, `max_file_bytes: int`; `load_mmkg_build_config(path) -> MmkgBuildConfig`.
  - `@dataclass(frozen=True) class Review`: `status: dict[tuple[str, str], str]` keyed by `(kind, key)`, `fix_name_vi: dict[str, str]`; `read_review(path: Path) -> Review` (missing file → empty review).
  - `select_images(candidates, credits, review: Review, config: MmkgBuildConfig, report: Counter) -> dict[str, list[dict[str, str]]]` — entity id → chosen image rows (columns of `images.csv`), only for entities with ≥ `min_images_per_item` valid images, at most `max_images_per_item` each.
- Rule order (each removal adds 1 to `report["images_dropped_<rule>"]`): `shared_between_entities` → `mime` → `license` → `too_small` → `no_author` → `review_drop` → `duplicate_sha1` → `over_max_per_item`.

- [ ] **Step 1: Write the failing tests** (rows built by a small helper `candidate(file, entity, origin="category", mime="image/jpeg", w=800, h=600, license="CC BY-SA 4.0", sha1=...)`):
  - `test_file_shared_by_two_entities_is_dropped_for_both`.
  - `test_license_rules`: `Public domain`, `CC0`, `CC BY 2.0`, `CC BY-SA 3.0` kept; `CC BY-NC 2.0`, `GFDL`, `""` dropped.
  - `test_short_side_below_224_is_dropped`: 223×1000 dropped, 224×224 kept.
  - `test_author_rule`: CC BY with empty artist and uploader `Bob` → kept with `author == "uploader: Bob"`; CC BY with both empty → dropped; `Public domain` with both empty → kept with `author == "unknown"`.
  - `test_review_drop_brings_in_next_image`: 6 valid files, the 2nd (by order) has review `drop` → 5 chosen, including the 6th.
  - `test_duplicate_sha1_keeps_smallest_name`: `b.jpg` and `a.jpg` same sha1 → only `a.jpg`.
  - `test_p18_first_then_name_order_and_cap`: 7 valid files with `P18 = z.jpg` → chosen `["z.jpg", 4 smallest others]`.
  - `test_entity_with_two_valid_images_gets_none`: 2 valid files → entity absent from the result.
  - `test_read_review_rejects_unknown_values`: a row with `review_status: maybe`, `kind: picture`, or an image `drop` whose `reason` is not one of the four reasons → `ValueError` naming the row; a missing `review.csv` → empty `Review`.
  - `test_image_id_is_stable`: `image_id == f"{entity_id}_{sha1[:8]}"`; `source_url == "https://commons.wikimedia.org/wiki/File:" + quote(name.replace(" ", "_"))`.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** `select_images` and the config/review readers. License check: exact match in `allowed_license_names` or prefix in `allowed_license_prefixes`; attribution needed when prefix in `attribution_license_prefixes`.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(mmkg): image selection rules for the MMKG build`.

---

### Task 5: Builder, entities, triples, review sheet and outputs (`scripts/build_mmkg.py`)

**Files:**
- Modify: `src/mmxai/data/mmkg_build.py`
- Create: `scripts/build_mmkg.py`, `configs/mmkg_vi_build.yaml`
- Test: `tests/test_mmkg_build.py`

**Interfaces:**
- Consumes: `select_images`, `read_review`, `MmkgBuildConfig` (Task 4); `read_rows`, `write_rows` (Task 1).
- Produces:
  - `@dataclass(frozen=True) class BuiltMmkg`: `entities`, `triples`, `images` (lists of row dicts in ADR-0015 column order + extra columns from spec §5.5), `report: dict`, `pending: list[dict[str, str]]` (review rows to append).
  - `build_mmkg(snapshot_dir: Path, review: Review, config: MmkgBuildConfig) -> BuiltMmkg` — pure (reads files, writes nothing).
  - `write_built_mmkg(built: BuiltMmkg, output_dir: Path, review_file: Path, max_file_bytes: int) -> None` — writes the three CSVs + `build_report.json`, **appends** `pending` rows to `review.csv` (never rewrites existing rows).
  - `scripts/build_mmkg.py --config ... [--require-reviewed]` — exits with code 1 and prints the counts when `--require-reviewed` is set and any official row is still `pending`.
- Algorithm (in this order): entity review `drop` removes the entity → items = domain entities with images from `select_images` → triple review `drop` and unknown properties removed, properties renamed via `relations`, duplicates of `(head, relation, tail)` merged (`wikidata_property` = sorted codes joined by `|`) → concepts = non-item entities that are the tail of item-headed triples from ≥ `min_items_per_concept` distinct items → keep triples whose both ends are kept and `head != tail` → `name_vi` = `fix_name_vi` if given → sort every table by its key.

- [ ] **Step 1: Write the failing tests** on a tiny snapshot folder written with `write_rows` in `tmp_path`:
  - `test_concept_needs_three_items`: concept shared by 3 items kept, by 2 items dropped (with its triples).
  - `test_location_and_collection_merge_to_kept_at`: item `P276` M and `P195` M → one triple `kept_at` with `wikidata_property == "P195|P276"`.
  - `test_domain_entity_with_few_images_can_be_concept`: in-domain museum with 1 image, referenced by 3 items → `type == "concept"`, no images.
  - `test_concept_to_concept_triple_kept_when_both_kept`.
  - `test_fix_name_vi_overrides_label` and `test_entity_drop_removes_its_triples`.
  - `test_every_reference_exists`: every `head_id`, `tail_id` and image `entity_id` of the built tables is an `entity_id` of `entities`; every item has 3–5 images.
  - `test_build_is_deterministic`: build twice → identical rows and report.
  - `test_pending_rows_are_appended_not_rewritten`: existing `review.csv` with one `ok` row; build with a higher `min_items_per_concept` → that row still present byte-for-byte; new rows `review_status == "pending"` only for official rows without a review.
  - `test_report_counts_raw_and_official`: report has `raw_entities`, `raw_triples`, `raw_image_candidates`, `entities_item`, `entities_concept`, `triples`, `images`, `pending_*` and the `images_dropped_*` counts.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** `build_mmkg`, `write_built_mmkg`, the script, and `configs/mmkg_vi_build.yaml` with every value from Global Constraints (`snapshot_dir` left as `mmkg/raw/YYYYMMDD` placeholder with a comment to set it after Task 7's collect; the loader rejects the placeholder with a clear message).
- [ ] **Step 4: Run** → PASS; `uv run ruff check`.
- [ ] **Step 5: Commit** `feat(mmkg): build official MMKG files from the raw snapshot`.

---

### Task 6: Download the selected images

**Files:**
- Create: `src/mmxai/data/mmkg_images.py`, `scripts/download_mmkg_images.py`
- Test: `tests/test_mmkg_images.py`

**Interfaces:**
- Consumes: `WikimediaClient` (Task 2), `read_rows` (Task 1), `images.csv` columns (Task 4/5).
- Produces: `download_images(images: list[dict[str, str]], images_dir: Path, client: WikimediaClient) -> dict[str, int]` returning counts `downloaded`, `already_present`, `sha1_mismatch`. Files saved as `<image_id><ext>` (ext from the file name, lower-case). Originals are downloaded (URL from `prop=imageinfo&iiprop=url`), so the Commons sha1 can be checked.

- [ ] **Step 1: Write the failing tests** with a fake client:
  - `test_downloads_and_checks_sha1`: bytes whose sha1 matches → file written, `downloaded == 1`.
  - `test_existing_file_with_right_sha1_is_skipped`: no download call made.
  - `test_sha1_mismatch_is_not_saved`: wrong bytes → no file, `sha1_mismatch == 1`, and the script exits non-zero when this count is > 0.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** the function and script (`--config configs/mmkg_vi_build.yaml`; writes into `mmkg/images/`, already gitignored). Write to `<name>.part` then rename.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(mmkg): download selected images and verify sha1`.

---

### Task 7: Real run, ADR and documents

**Files:**
- Create: `tests/test_mmkg_committed.py`, `mmkg/raw/<YYYYMMDD>/*` (script output), `mmkg/entities.csv`, `mmkg/triples.csv`, `mmkg/images.csv`, `mmkg/review.csv`, `mmkg/build_report.json`, `docs/decisions/0051-<slug>.md`, `data_cards/mmkg_vi.md`
- Modify: `configs/mmkg_vi_build.yaml` (`snapshot_dir`), `mmkg/README.md`, `docs/decisions/README.md`, `data_cards/README.md` (mark `mmkg_vi.md` as present), `docs/ai-usage-log.md`

- [ ] **Step 1: Collect** `uv run python scripts/collect_mmkg.py --config configs/mmkg_vi_collect.yaml`. Expected: about 170 domain entities, roughly 1.9k triples, about 5k image candidates (survey numbers, 2026-11-16). If a count is 0 or far off, stop and report; do not retry with changed queries silently.
- [ ] **Step 2: Build** after setting `snapshot_dir`: `uv run python scripts/build_mmkg.py --config configs/mmkg_vi_build.yaml`. Check `build_report.json`: entities (item + concept) in 50–150, triples in 300–1,000, images in 200–500. If entities fall outside 50–150, **stop and ask** before changing `min_items_per_concept` (spec §5.3: the group adjusts and records it).
- [ ] **Step 3: Add `tests/test_mmkg_committed.py`** (skips when `mmkg/entities.csv` is absent): every file under `mmkg/` except `mmkg/images/` is < 1,048,576 bytes; referential integrity of the committed tables (same check as `test_every_reference_exists`); every image `license` allowed and `author`/`source_url` non-empty. Run `uv run pytest` and `uv run ruff check` → clean.
- [ ] **Step 4: Write ADR-0052** with `/new-adr`: raw snapshot committed as-is + filtering at build time + review stored apart; the thresholds and the relation list; status `Đề xuất`. Add the index row.
- [ ] **Step 5: Write `data_cards/mmkg_vi.md`** with every mandatory section of ADR-0015; numbers "snapshot → official" copied from `build_report.json`; the reason for each rule (spec §5.3, marked as the group's choice, not the assignment's); the relation table with Vietnamese meanings; licence and attribution rule; a "Kết quả duyệt" section left for the reviewers to fill (counts per reason, who reviewed). Update `mmkg/README.md` with the flow collect → build → review → rebuild → download, and how to open a file to review (`https://commons.wikimedia.org/wiki/File:<file_name>`).
- [ ] **Step 6: Log AI use** with `/log-ai-usage` (last column `Chưa`).
- [ ] **Step 7: Commit** in two commits: `data(mmkg): raw snapshot <YYYYMMDD> and first build` and `docs(mmkg): ADR-0052, data card and README for the MMKG build`.
- [ ] **Step 8: Hand over to the team**: the review (`mmkg/review.csv`, all rows `pending`) and image download happen after this plan; `--require-reviewed` must pass before any experiment uses the MMKG.
