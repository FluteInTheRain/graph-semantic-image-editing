# Kế hoạch cài đặt: embedding CLIP cho Visual Genome và Flickr30k

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Trích embedding ảnh và văn bản bằng CLIP đóng băng cho 2 bộ dữ liệu, kiểm tra tự động, và đăng lên Kaggle cho cả nhóm.

**Architecture:** Loader của từng bộ dữ liệu trả về bản ghi chung (`ImageRecord`, `TextRecord`). Encoder biến bản ghi thành mảng, `pipeline.py` nối các bước, `cache.py` ghi `.npz` + `meta.json`, `verify.py` kiểm tra đầu ra. `src/mmxai/` là bản gốc; 1 notebook chứa bản sao được test so khớp.

**Tech Stack:** Python 3.11, numpy, pyyaml, torch, open-clip-torch 3.3.0, pillow, pytest, ruff, uv. Không thêm dependency mới.

**Spec:** `docs/superpowers/specs/2026-11-04-clip-embeddings-design.md`

## Global Constraints

- Code, docstring, comment viết tiếng Anh; tài liệu viết tiếng Việt (`CLAUDE.md`).
- Code đơn giản, giải thích được khi vấn đáp: không one-liner "thông minh", không metaprogramming.
- Sau **mỗi task**: chạy `uv run ruff check`, `uv run ruff format --check`, `uv run pytest`, dán kết quả thật, rồi **dừng cho người dùng duyệt**. Chỉ commit khi người dùng đồng ý.
- Test mặc định không cần torch, GPU, mạng hay `data/`. Test cần torch dùng `pytest.importorskip("torch")`.
- `torch` và `open_clip` chỉ được import trong `features/clip_encoder.py`, hoặc bên trong hàm `run_extraction`.
- **Không cài torch trên máy cá nhân.** Mọi lần chạy CLIP thật diễn ra trên Kaggle, nơi có sẵn torch.
- Không commit `*.npz`, dữ liệu hay token. `CLIP` luôn đóng băng.
- Model `ViT-B-32`, pretrained `laion2b_s34b_b79k`, 5 region mỗi ảnh, `sampling_seed: 0`, ngưỡng ảnh lỗi `0.001`.
- Định dạng đầu ra đúng như spec §4. Ghi bằng `np.savez`, đọc được với `allow_pickle=False`.

## Review Focus

1. **Chạy lại khi đã có kết quả cũ:** phải báo lỗi, không ghi đè âm thầm. Test: `test_save_features_refuses_existing_output` (Task 3).
2. **Bản chạy thử bị nhầm thành bản thật:** ghi vào `<dataset>_partial/`, `is_partial: true`. Test: `test_feature_dir_separates_partial_runs` (Task 3).
3. **Chạy dở thì bị ngắt:** không được để lại thư mục trông như hoàn chỉnh. Test: `test_load_features_requires_meta` (Task 3).
4. **Ảnh có trong JSON nhưng thiếu hoặc hỏng:** ghi vào danh sách lỗi, không làm chết cả lần chạy, và văn bản của ảnh đó cũng bị bỏ. Test: `test_encode_images_reports_missing_and_corrupt` (Task 5), `test_filter_texts_drops_failed_images` (Task 8).
5. **Bản sao trong notebook bị lệch:** test phải thất bại. Test: `test_notebook_cells_match_src` (Task 9).

---

### Task 1: Ghi lại quyết định

**Files:**
- Create: 2 ADR bằng skill `/new-adr`, số 0040 trở đi
- Modify: `docs/decisions/README.md`, `CLAUDE.md`, `data/README.md`, `data_cards/README.md`

**Interfaces:** Không có.

- [ ] **Step 1:** ADR "Chọn Visual Genome và Flickr30k + ConceptNet": lý do, các phương án đã loại (MKG-W/Y, FB15k-237-IMG, MS-COCO), và hệ quả với phương pháp cấp độ 2.
- [ ] **Step 2:** ADR "Định dạng embedding và chia sẻ qua Kaggle": `.npz` + `meta.json`, thêm embedding văn bản, 5 region mỗi ảnh, Kaggle thay Google Drive, notebook `1x` chứa bản sao code. Ghi rõ ADR này thay phần nào của ADR-0008 và là ngoại lệ gì của ADR-0012.
- [ ] **Step 3:** Sửa dòng cache đặc trưng trong `CLAUDE.md` từ `data/` thành `artifacts/`. Thêm Flickr30k vào `data/README.md` và `data_cards/README.md`; ghi Visual Genome v1.2 trong `data/README.md`.
- [ ] **Step 4:** Chạy `uv run ruff check`, `uv run pytest`. Kỳ vọng: pass.
- [ ] **Step 5:** Dừng để duyệt. Commit: `docs(adr): record dataset choice and embedding format`.

### Task 2: Đọc config

**Files:**
- Create: `src/mmxai/utils/config.py`, `tests/test_config.py`, `configs/features_visual_genome.yaml`, `configs/features_flickr30k.yaml`
- Modify: `configs/README.md`

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True)
class FeatureConfig:
    dataset: str  # "visual_genome" | "flickr30k"
    dataset_version: str  # "1.2" | "karpathy"
    data_dir: str
    output_root: str
    model: str
    pretrained: str
    device: str  # "auto" | "cuda" | "cpu"
    image_batch_size: int
    text_batch_size: int
    num_workers: int
    max_failed_image_fraction: float
    images_dir: str | None = None  # default: <data_dir>/images
    regions_per_image: int | None = None
    sampling_seed: int | None = None


def load_feature_config(path: str | Path) -> FeatureConfig: ...
```

- [ ] **Step 1: Viết test.**

```python
def test_loads_both_committed_configs(): ...  # dataset, regions_per_image == 5 for VG, None for Flickr
def test_unknown_key_raises(tmp_path): ...  # pytest.raises(ValueError, match="unknown")
def test_missing_required_key_raises(tmp_path): ...
def test_visual_genome_requires_sampling_fields(tmp_path): ...  # ValueError
def test_flickr30k_rejects_sampling_fields(tmp_path): ...  # ValueError
def test_unknown_dataset_raises(tmp_path): ...
```

- [ ] **Step 2:** Chạy `uv run pytest tests/test_config.py -v`. Kỳ vọng: FAIL vì chưa có module.
- [ ] **Step 3:** Viết code và 2 file YAML. Đọc bằng `yaml.safe_load`, so tên khoá với `dataclasses.fields` (ADR-0007). `image_batch_size: 256`, `text_batch_size: 1024`, `num_workers: 4`.
- [ ] **Step 4:** Chạy lại test. Kỳ vọng: PASS. Dừng để duyệt. Commit: `feat(utils): load feature extraction config`.

### Task 3: Ghi và đọc đặc trưng

**Files:**
- Create: `src/mmxai/features/cache.py`, `tests/test_feature_cache.py`

**Interfaces:**
- Consumes: `FeatureConfig`
- Produces:

```python
def feature_dir(config: FeatureConfig, is_partial: bool) -> Path: ...


# <output_root>/<dataset or dataset_partial>/<model>__<pretrained>


def save_features(
    output_dir: Path,
    image_embeddings: np.ndarray,
    image_ids: np.ndarray,
    text_embeddings: np.ndarray,
    text_image_ids: np.ndarray,
    text_id_name: str,
    text_id_values: np.ndarray,
    meta: dict,
) -> None: ...


# text_id_name is "text_ids" (Visual Genome) or "caption_index" (Flickr30k)


def load_meta(output_dir: Path) -> dict: ...
def load_image_features(
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray]: ...  # embeddings, image_ids
def load_text_features(output_dir: Path) -> dict[str, np.ndarray]: ...
def check_meta_matches_config(
    meta: dict, config: FeatureConfig
) -> None: ...  # ValueError on mismatch
def compute_code_hash(package_root: Path, module_paths: list[str]) -> str: ...  # sha256 hex
```

- [ ] **Step 1: Viết test.**

```python
def test_save_and_load_round_trip(tmp_path): ...  # arrays equal, dtypes float32/int64
def test_saved_arrays_load_without_pickle(tmp_path): ...  # np.load(path, allow_pickle=False)
def test_save_features_refuses_existing_output(tmp_path): ...  # FileExistsError
def test_load_features_requires_meta(tmp_path): ...  # FileNotFoundError when meta.json is absent
def test_feature_dir_separates_partial_runs(): ...  # ".../flickr30k_partial/ViT-B-32__laion2b_s34b_b79k"
def test_check_meta_rejects_other_model(): ...  # pytest.raises(ValueError)
def test_code_hash_changes_with_content(tmp_path): ...  # same content same hash, edited differs
```

- [ ] **Step 2:** Chạy test. Kỳ vọng: FAIL.
- [ ] **Step 3:** Viết code. `save_features` ghi vào `<output_dir>.tmp`, ghi `meta.json` sau cùng, rồi `os.replace` sang tên thật, để một lần chạy bị ngắt không để lại thư mục trông như hoàn chỉnh. Hash theo thứ tự `module_paths` đã cho.
- [ ] **Step 4:** Chạy lại test. Kỳ vọng: PASS. Dừng để duyệt. Commit: `feat(features): save and load embedding cache`.

### Task 4: Kiểm tra đầu ra

**Files:**
- Create: `src/mmxai/features/verify.py`, `tests/test_feature_verify.py`

**Interfaces:**
- Consumes: `load_meta`, `load_image_features`, `load_text_features`
- Produces: `def verify_features(output_dir: Path, seed: int = 0, num_pairs: int = 1000) -> None` (ném `FeatureVerificationError(ValueError)` kèm tất cả lỗi tìm được)

- [ ] **Step 1: Viết test.** Dữ liệu giả: mỗi văn bản là embedding của ảnh tương ứng cộng thêm nhiễu nhỏ, đã chuẩn hoá. Test hợp lệ phải pass; mỗi test còn lại làm hỏng đúng một điều.

```python
def test_valid_output_passes(tmp_path): ...
def test_row_count_mismatch_fails(tmp_path): ...
def test_nan_fails(tmp_path): ...
def test_unnormalized_vector_fails(tmp_path): ...  # np.isclose(norm, 1, atol=1e-3)
def test_duplicate_image_id_fails(tmp_path): ...
def test_text_pointing_to_unknown_image_fails(tmp_path): ...
def test_more_than_five_regions_fails(tmp_path): ...
def test_flickr_requires_exactly_five_captions_when_complete(tmp_path): ...
def test_partial_flickr_allows_fewer_captions(tmp_path): ...
def test_misaligned_rows_fail_sanity_check(
    tmp_path,
): ...  # reversed text rows, matched not > shuffled
```

- [ ] **Step 2:** Chạy test. Kỳ vọng: FAIL.
- [ ] **Step 3:** Viết code theo bảng kiểm tra ở spec §6. Kiểm tra số văn bản mỗi ảnh và kiểm tra ngữ nghĩa dùng `meta["dataset"]`, `meta["is_partial"]`. Cặp xáo trộn dùng `np.random.default_rng(seed)`. Kết quả không được đưa vào quyết định tham số nào.
- [ ] **Step 4:** Chạy lại test. Kỳ vọng: PASS. Dừng để duyệt. Commit: `feat(features): verify extracted embeddings`.

### Task 5: Encoder CLIP

**Files:**
- Create: `src/mmxai/data/records.py`, `src/mmxai/features/clip_encoder.py`, `tests/test_clip_encoder.py`

**Interfaces:**
- Produces:

```python
# data/records.py
@dataclass(frozen=True)
class ImageRecord:
    image_id: int
    path: Path


@dataclass(frozen=True)
class TextRecord:
    image_id: int
    text_id: int  # region_id (Visual Genome) or caption_index (Flickr30k)
    text: str


# features/clip_encoder.py
@dataclass
class ClipBundle:
    model: torch.nn.Module
    preprocess: Callable
    tokenizer: Callable
    device: str


def freeze_model(model: torch.nn.Module) -> torch.nn.Module: ...
def load_frozen_clip(model_name: str, pretrained: str, device: str) -> ClipBundle: ...
def encode_images(
    bundle: ClipBundle, records: list[ImageRecord], batch_size: int, num_workers: int
) -> tuple[np.ndarray, np.ndarray, list[int]]: ...
# embeddings (float32, L2-normalized), encoded image_ids in input order, failed image_ids
def encode_texts(bundle: ClipBundle, texts: list[str], batch_size: int) -> np.ndarray: ...
```

- [ ] **Step 1: Viết test.** Đầu file: `torch = pytest.importorskip("torch")`. Dùng model giả, không tải trọng số.

```python
def test_freeze_model_disables_gradients_and_sets_eval(): ...
def test_encode_images_returns_unit_float32(tmp_path): ...  # norm == 1, dtype float32
def test_encode_images_keeps_input_order(tmp_path): ...  # num_workers=0 and 2
def test_encode_images_reports_missing_and_corrupt(
    tmp_path,
): ...  # failed ids listed, others still encoded
def test_encode_images_handles_grayscale(tmp_path): ...  # preprocess receives RGB
def test_encode_texts_preserves_row_count_across_batches(): ...
```

- [ ] **Step 2:** Chạy `uv run pytest tests/test_clip_encoder.py -v`. Kỳ vọng: báo skipped, vì máy này không có torch.
- [ ] **Step 3:** Viết code. Tải bằng `open_clip.create_model_and_transforms(model_name, pretrained=pretrained)` và `open_clip.get_tokenizer(model_name)`. Dùng `torch.no_grad()`, không autocast, ép `.float()`, chuẩn hoá L2. `device="auto"` chọn `cuda` nếu có. Dataset bắt `OSError` và `ValueError` khi mở ảnh rồi trả `None`, mở bằng `Image.open(path).convert("RGB")`; `collate_fn` tách phần lỗi; `shuffle=False`.
- [ ] **Step 4:** Chạy `uv run pytest`. Kỳ vọng: test này skipped, các test khác PASS. Encoder **chưa được chạy thật** ở bước này; lần chạy thử trên Kaggle ở Task 9 sẽ kiểm tra. Dừng để duyệt. Commit: `feat(features): add frozen CLIP encoder`.

### Task 6: Loader Flickr30k

**Files:**
- Create: `src/mmxai/data/flickr30k.py`, `tests/test_data_flickr30k.py`

**Interfaces:**
- Consumes: `ImageRecord`, `TextRecord`
- Produces:

```python
def load_image_records(data_dir: Path, images_dir: Path | None = None) -> list[ImageRecord]: ...
def load_captions(data_dir: Path) -> list[TextRecord]: ...
```

Đã xác nhận trên file thật: `images[]` có `filename`, `imgid`, `split`, `sentences[]` với `raw`; 31.014 ảnh; 155.070 caption.

- [ ] **Step 1: Viết test** với `dataset_flickr30k.json` giả nhỏ.

```python
def test_image_id_comes_from_filename(tmp_path): ...  # "1000092795.jpg" -> 1000092795, not imgid
def test_records_sorted_by_image_id(tmp_path): ...
def test_extra_images_in_folder_are_ignored(tmp_path): ...
def test_caption_index_runs_zero_to_four(tmp_path): ...  # caption text comes from "raw"
def test_non_numeric_filename_raises(tmp_path): ...  # ValueError
def test_images_dir_override(tmp_path): ...
```

- [ ] **Step 2:** Chạy test. Kỳ vọng: FAIL.
- [ ] **Step 3:** Viết code. Không đọc trường `split`: split chính thức nằm trong `splits/`.
- [ ] **Step 4:** Chạy lại test. Kỳ vọng: PASS. Dừng để duyệt. Commit: `feat(data): load Flickr30k images and captions`.

### Task 7: Loader Visual Genome và chọn region

**Files:**
- Create: `src/mmxai/data/visual_genome.py`, `tests/test_data_visual_genome.py`

**Interfaces:**
- Consumes: `ImageRecord`, `TextRecord`
- Produces:

```python
def load_image_records(data_dir: Path, images_dir: Path | None = None) -> list[ImageRecord]: ...
def load_region_texts(data_dir: Path) -> list[TextRecord]: ...
def sample_regions(
    regions: list[TextRecord], regions_per_image: int, seed: int
) -> list[TextRecord]: ...
```

Đã xác nhận trên file thật:
- `image_data.json`: danh sách có `image_id`, `url`. Thư mục ảnh là phần áp chót của `url` (`VG_100K` 64.345, `VG_100K_2` 43.732).
- `region_descriptions.json`: danh sách có `id`, `regions[]`; mỗi region có `region_id`, `image_id`, `phrase`.
- 108.077 ảnh, 5.408.689 region, ít nhất 3 region mỗi ảnh, 7 `phrase` rỗng.

- [ ] **Step 1: Viết test** với JSON giả nhỏ.

```python
def test_image_path_uses_folder_from_url(tmp_path): ...  # images/VG_100K_2/1.jpg
def test_empty_phrases_are_dropped(tmp_path): ...
def test_sampling_is_deterministic(): ...  # same seed gives identical records
def test_sampling_changes_with_seed(): ...
def test_sampling_takes_at_most_k_without_repeats(): ...
def test_image_with_fewer_regions_keeps_all(): ...
def test_sampling_output_sorted_by_image_then_region(): ...
```

- [ ] **Step 2:** Chạy test. Kỳ vọng: FAIL.
- [ ] **Step 3:** Viết code. Chỉ dùng một `rng = np.random.default_rng(seed)`, duyệt ảnh tăng dần, region tăng dần, và `rng.choice(n, size=k, replace=False)` khi `n > k`. Bỏ `phrase` rỗng **trước** khi chọn, vì vector của chuỗi rỗng không phải một truy vấn. Không có lọc nội dung nào khác.
- [ ] **Step 4:** Chạy lại test. Kỳ vọng: PASS. Dừng để duyệt. Commit: `feat(data): load Visual Genome and sample regions`.

### Task 8: Nối pipeline và script

**Files:**
- Create: `src/mmxai/features/pipeline.py`, `scripts/extract_features.py`, `tests/test_feature_pipeline.py`
- Modify: `scripts/README.md`

**Interfaces:**
- Consumes: mọi hàm ở Task 2–7
- Produces:

```python
CODE_MODULES: list[str]  # fixed order, relative to the mmxai package:
# utils/config.py, data/records.py, data/flickr30k.py, data/visual_genome.py,
# features/cache.py, features/verify.py, features/clip_encoder.py, features/pipeline.py


def check_failed_fraction(num_failed: int, num_total: int, max_fraction: float) -> None: ...
def filter_texts_for_images(
    texts: list[TextRecord], kept_image_ids: set[int]
) -> list[TextRecord]: ...
def load_dataset_records(config: FeatureConfig) -> tuple[list[ImageRecord], list[TextRecord]]: ...
def build_meta(
    config: FeatureConfig,
    num_images: int,
    num_texts: int,
    failed_image_ids: list[int],
    is_partial: bool,
    versions: dict[str, str],
    code_hash: str,
) -> dict: ...
def run_extraction(
    config: FeatureConfig, limit_images: int | None = None, limit_texts: int | None = None
) -> Path: ...
```

Script: `--config`, `--data-dir`, `--images-dir`, `--output-root`, `--limit-images`, `--limit-texts`. Ghi đè bằng `dataclasses.replace`.

- [ ] **Step 1: Viết test.** Test không import torch.

```python
def test_failed_fraction_at_threshold_passes(): ...  # 1 of 1000 with 0.001
def test_failed_fraction_above_threshold_raises(): ...  # 2 of 1000
def test_filter_texts_drops_failed_images(): ...
def test_load_dataset_records_samples_visual_genome(tmp_path): ...
def test_meta_matches_spec_fields(): ...
def test_meta_marks_partial_run(): ...
def test_importing_pipeline_does_not_import_torch(): ...  # run in a subprocess, inspect sys.modules
```

- [ ] **Step 2:** Chạy test. Kỳ vọng: FAIL.
- [ ] **Step 3:** Viết code. Thứ tự trong `run_extraction`: nạp bản ghi (đã chọn region), áp `limit_images` rồi lọc văn bản theo ảnh, áp `limit_texts`, kiểm tra thư mục đích chưa tồn tại **trước khi mã hoá**, mã hoá ảnh, kiểm tra ngưỡng lỗi, lọc văn bản, mã hoá văn bản, lưu, `verify_features`. Import `clip_encoder` bên trong hàm này. Tính hash bằng `compute_code_hash(Path(mmxai.features.__file__).parent.parent, CODE_MODULES)`, hoặc đường dẫn tương đương chạy được cả khi không có `__init__.py`.
- [ ] **Step 4:** Chạy lại test. Kỳ vọng: PASS.
- [ ] **Step 5:** Dừng để duyệt. Commit: `feat(features): add extraction pipeline and CLI`.

### Task 9: Notebook Kaggle

**Files:**
- Create: `notebooks/10_prepare_clip_embeddings.ipynb`, `scripts/sync_notebook_code.py`, `tests/test_notebooks_match_src.py`
- Modify: `notebooks/README.md`

**Interfaces:**
- Consumes: `CODE_MODULES`, `run_extraction`, `FeatureConfig`
- Produces: Mỗi module một ô code, dòng đầu là `%%writefile /kaggle/temp/mmxai_src/mmxai/<module>`. Các ô cuối cài `open-clip-torch==3.3.0`, thêm `/kaggle/temp/mmxai_src` vào `sys.path`, tạo `FeatureConfig`, chạy thử rồi chạy đầy đủ.

`scripts/sync_notebook_code.py` chỉ dùng `json` để ghi lại các ô đó từ `src/mmxai/`, tránh phải chép tay 8 module.

- [ ] **Step 1: Viết test.** Đọc `.ipynb` bằng `json`.

```python
def test_notebook_cells_match_src(): ...  # each module: cell minus magic line == src file
def test_notebook_contains_every_code_module(): ...
def test_notebook_sections_in_order(): ...  # config, code, smoke, dataset 1, dataset 2
def test_only_one_dataset_setting(): ...  # exactly one DATASET assignment, in the first code cell
def test_notebook_has_no_outputs(): ...
```

- [ ] **Step 2:** Chạy test. Kỳ vọng: FAIL.
- [ ] **Step 3:** Viết script và tạo notebook theo thứ tự config, code, smoke, dataset 1, dataset 2. Chỉ phần ứng với `DATASET` được chạy; giá trị khác hai lựa chọn thì báo lỗi ngay ở ô config. Smoke ghi vào `/kaggle/temp/smoke`. Phần Visual Genome tải dữ liệu vào `/kaggle/temp` nếu chưa có bản sao. Phần Flickr30k gắn `eeshawn/flickr30k`, dùng `images_dir=/kaggle/input/flickr30k/flickr30k_images` (xác nhận lại đường dẫn này trên Kaggle) và tải `dataset_flickr30k.json`. Output ghi vào `/kaggle/working/features`.
- [ ] **Step 4:** Chạy lại test. Kỳ vọng: PASS. Sửa thử một ký tự trong một ô: test phải FAIL; khôi phục lại.
- [ ] **Step 5:** Người dùng tải notebook lên Kaggle, đặt `DATASET = "flickr30k"` và chạy tới hết phần smoke (1.000 ảnh + 10.000 văn bản). Kỳ vọng: `verify_features` pass, có `.npz` và `meta.json`. Đây là lần đầu encoder chạy thật; nếu lỗi thì sửa trong `src/mmxai/`, đồng bộ lại notebook và chạy lại.
- [ ] **Step 6:** Dừng để duyệt. Commit: `feat(notebooks): add Kaggle embedding notebook`.

### Task 10: Chạy thật và đăng

Người dùng thực hiện các bước cần tài khoản Kaggle.

**Files:**
- Create: `data_cards/visual_genome.md`, `data_cards/flickr30k.md`
- Modify: `docs/guides/chuan-bi-embedding-clip.md`, `README.md`, `docs/ai-usage-log.md`

**Interfaces:** Đầu ra đúng spec §4.

- [ ] **Step 1:** Gọi subagent `leakage-auditor` trên nhánh, sửa vi phạm nếu có.
- [ ] **Step 2:** Người dùng chạy notebook hai lần trên Kaggle, mỗi lần một giá trị `DATASET`, có thể song song. Kỳ vọng: `verify_features` pass, `code_hash` và `open_clip_version` giống nhau ở hai bộ dữ liệu. Xác nhận `%%writefile` và đường dẫn chạy đúng; ghi lại tốc độ thật.
- [ ] **Step 3:** Viết 2 data card theo mục tối thiểu trong `data_cards/README.md`, dùng số đếm thật và các điều kiện giấy phép ở spec §8.
- [ ] **Step 4:** Người dùng đăng `clip-features-visual-genome` (công khai, CC BY 4.0, ghi nguồn) và `clip-features-flickr30k` (riêng tư, chia sẻ cho nhóm).
- [ ] **Step 5:** Sửa hướng dẫn: thay `TODO` bằng link Kaggle, bỏ cảnh báo "code chưa được viết", điền thời gian đo được. Thử tải về và đọc theo §2 của hướng dẫn. Kỳ vọng: shape đúng với `meta.json`.
- [ ] **Step 6:** Chạy lint và test đầy đủ, dừng để duyệt. Commit: `docs(data): add data cards and publish embeddings`. Mở PR theo ADR-0014 sau khi người dùng đồng ý.
