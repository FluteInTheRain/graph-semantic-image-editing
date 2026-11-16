"""Build the Vietnamese heritage MMKG from a raw snapshot: image selection rules (no network).

Every rule removes images and counts each removal in a report, so the team can
justify every dropped image.
"""

import csv
import urllib.parse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from mmxai.utils.yaml_config import load_yaml_dataclass

IMAGE_COLUMNS = [
    "image_id",
    "entity_id",
    "file_name",
    "source_url",
    "license",
    "author",
    "sha1",
    "width",
    "height",
]

REVIEW_KINDS = {"image", "triple", "entity"}
REVIEW_STATUSES = {"ok", "drop", "pending"}
IMAGE_DROP_REASONS = {"not_the_object", "identifiable_person", "duplicate", "other"}
COMMONS_FILE_URL = "https://commons.wikimedia.org/wiki/File:"


@dataclass(frozen=True)
class MmkgBuildConfig:
    snapshot_dir: str
    review_file: str
    output_dir: str
    allowed_mime_types: list[str]
    allowed_license_names: list[str]
    allowed_license_prefixes: list[str]
    attribution_license_prefixes: list[str]
    min_short_side_px: int
    min_images_per_item: int
    max_images_per_item: int
    min_items_per_concept: int
    relations: dict[str, str]
    max_file_bytes: int


def load_mmkg_build_config(path: str | Path) -> MmkgBuildConfig:
    """Read the build config YAML (a mistyped key is an error)."""
    return load_yaml_dataclass(path, MmkgBuildConfig)


@dataclass(frozen=True)
class Review:
    """Human decisions from review.csv.

    status is keyed by (kind, key); fix_name_vi maps an entity id to a corrected label.
    """

    status: dict[tuple[str, str], str]
    fix_name_vi: dict[str, str]


def read_review(path: Path) -> Review:
    """Read and validate review.csv. A missing file means no decisions yet."""
    if not path.exists():
        return Review({}, {})

    status: dict[tuple[str, str], str] = {}
    fix_name_vi: dict[str, str] = {}
    with open(path, encoding="utf-8", newline="") as file:
        # Row 1 is the header, so the first data row is row 2 (matches a spreadsheet).
        for row_number, row in enumerate(csv.DictReader(file), start=2):
            kind = row["kind"]
            key = row["key"]
            review_status = row["review_status"]
            reason = row["reason"]
            fix = row["fix_name_vi"]
            where = f"{path.name} row {row_number} (key {key!r})"

            if kind not in REVIEW_KINDS:
                raise ValueError(f"{where}: unknown kind {kind!r}")
            if review_status not in REVIEW_STATUSES:
                raise ValueError(f"{where}: unknown review_status {review_status!r}")
            if kind == "image" and review_status == "drop" and reason not in IMAGE_DROP_REASONS:
                raise ValueError(
                    f"{where}: image drop needs a reason in {sorted(IMAGE_DROP_REASONS)}"
                )
            if fix and kind != "entity":
                raise ValueError(f"{where}: fix_name_vi is only allowed on entity rows")
            if (kind, key) in status:
                # Two rows for one item would be conflicting decisions; make that visible.
                raise ValueError(f"{where}: repeated (kind, key)")

            status[(kind, key)] = review_status
            if fix:
                fix_name_vi[key] = fix
    return Review(status, fix_name_vi)


def drop_where(
    rows: list[dict[str, str]], is_dropped, rule: str, report: Counter
) -> list[dict[str, str]]:
    """Remove rows for which `is_dropped(row)` is true and count each removal under `rule`."""
    kept: list[dict[str, str]] = []
    for row in rows:
        if is_dropped(row):
            report[f"images_dropped_{rule}"] += 1
        else:
            kept.append(row)
    return kept


def starts_with_any(text: str, prefixes: list[str]) -> bool:
    return any(text.startswith(prefix) for prefix in prefixes)


def license_is_allowed(license_name: str, config: MmkgBuildConfig) -> bool:
    if license_name in config.allowed_license_names:
        return True
    return starts_with_any(license_name, config.allowed_license_prefixes)


def short_side(row: dict[str, str]) -> int:
    """Smaller of width and height; 0 when either is not a number (counts as too small)."""
    try:
        return min(int(row["width"]), int(row["height"]))
    except ValueError:
        return 0


def pick_author(license_name: str, credit: dict[str, str], config: MmkgBuildConfig) -> str | None:
    """Author text for the image, or None when attribution is required but unknown."""
    artist = credit.get("artist", "").strip()
    if artist:
        return artist
    if starts_with_any(license_name, config.attribution_license_prefixes):
        uploader = credit.get("uploader", "").strip()
        if uploader:
            return f"uploader: {uploader}"
        return None
    return "unknown"


def drop_shared_files(rows: list[dict[str, str]], report: Counter) -> list[dict[str, str]]:
    """Drop a file listed under more than one entity: we cannot tell which one it shows."""
    entities_of_file: dict[str, set[str]] = {}
    for row in rows:
        entities_of_file.setdefault(row["file_name"], set()).add(row["entity_id"])

    def is_shared(row: dict[str, str]) -> bool:
        return len(entities_of_file[row["file_name"]]) > 1

    return drop_where(rows, is_shared, "shared_between_entities", report)


def drop_duplicate_sha1(rows: list[dict[str, str]], report: Counter) -> list[dict[str, str]]:
    """Among identical files (same sha1) keep only the smallest file name, across all entities."""
    smallest_name: dict[str, str] = {}
    for row in rows:
        sha1 = row["sha1"]
        if sha1 not in smallest_name or row["file_name"] < smallest_name[sha1]:
            smallest_name[sha1] = row["file_name"]

    def is_duplicate(row: dict[str, str]) -> bool:
        return row["file_name"] != smallest_name[row["sha1"]]

    return drop_where(rows, is_duplicate, "duplicate_sha1", report)


def make_image_row(row: dict[str, str], author: str) -> dict[str, str]:
    file_name = row["file_name"]
    quoted_name = urllib.parse.quote(file_name.replace(" ", "_"))
    return {
        "image_id": f"{row['entity_id']}_{row['sha1'][:8]}",
        "entity_id": row["entity_id"],
        "file_name": file_name,
        "source_url": COMMONS_FILE_URL + quoted_name,
        "license": row["license"],
        "author": author,
        "sha1": row["sha1"],
        "width": row["width"],
        "height": row["height"],
    }


def select_images(
    candidates: list[dict[str, str]],
    credits: list[dict[str, str]],
    review: Review,
    config: MmkgBuildConfig,
    report: Counter,
) -> dict[str, list[dict[str, str]]]:
    """Apply the filter rules in order and return entity id -> chosen image rows.

    Rule order: shared_between_entities, mime, license, too_small, no_author,
    review_drop, duplicate_sha1, then per entity below_min / over_max.
    """
    credit_of_file = {credit["file_name"]: credit for credit in credits}

    rows = drop_shared_files(candidates, report)
    rows = drop_where(rows, lambda r: r["mime"] not in config.allowed_mime_types, "mime", report)
    rows = drop_where(
        rows, lambda r: not license_is_allowed(r["license"], config), "license", report
    )
    rows = drop_where(rows, lambda r: short_side(r) < config.min_short_side_px, "too_small", report)

    # Author is decided here (not later) because a missing author is itself a drop rule.
    author_of_row: dict[tuple[str, str], str] = {}
    has_author: list[dict[str, str]] = []
    for row in rows:
        credit = credit_of_file.get(row["file_name"], {})
        author = pick_author(row["license"], credit, config)
        if author is None:
            report["images_dropped_no_author"] += 1
        else:
            author_of_row[(row["entity_id"], row["file_name"])] = author
            has_author.append(row)

    def is_review_dropped(row: dict[str, str]) -> bool:
        return review.status.get(("image", row["file_name"])) == "drop"

    rows = drop_where(has_author, is_review_dropped, "review_drop", report)
    rows = drop_duplicate_sha1(rows, report)

    valid_of_entity: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        valid_of_entity.setdefault(row["entity_id"], []).append(row)

    chosen: dict[str, list[dict[str, str]]] = {}
    for entity_id, valid in valid_of_entity.items():
        # The P18 (main image) of an item is the most trusted, so it comes first.
        valid.sort(key=lambda r: (r["origin"] != "p18", r["file_name"]))
        if len(valid) < config.min_images_per_item:
            report["images_dropped_below_min_per_item"] += len(valid)
            continue
        kept = valid[: config.max_images_per_item]
        report["images_dropped_over_max_per_item"] += len(valid) - len(kept)
        chosen[entity_id] = [
            make_image_row(r, author_of_row[(entity_id, r["file_name"])]) for r in kept
        ]
    return chosen
