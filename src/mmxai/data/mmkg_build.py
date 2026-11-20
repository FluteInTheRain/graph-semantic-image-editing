"""Build the Vietnamese heritage MMKG from a raw snapshot (no network).

Every rule removes rows and counts each removal in a report, so the team can
justify every dropped image, triple and entity.
"""

import csv
import json
import re
import urllib.parse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from mmxai.data.mmkg_csv import read_rows, write_rows
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
ENTITY_COLUMNS = ["entity_id", "name_vi", "name_en", "type", "source_url"]
TRIPLE_COLUMNS = ["head_id", "relation", "tail_id", "source", "wikidata_property"]
REVIEW_COLUMNS = ["kind", "key", "review_status", "reason", "fix_name_vi", "reviewer", "date"]
WIKIDATA_ENTITY_URL = "https://www.wikidata.org/wiki/"
# Every counter the report always shows (0 when nothing was dropped).
DROP_COUNTERS = [
    "images_dropped_shared_between_entities",
    "images_dropped_mime",
    "images_dropped_license",
    "images_dropped_too_small",
    "images_dropped_no_author",
    "images_dropped_review_drop",
    "images_dropped_duplicate_sha1",
    "images_dropped_below_min_per_item",
    "images_dropped_over_max_per_item",
    "images_dropped_entity_review_drop",
    "entities_dropped_review",
    "triples_dropped_unknown_relation",
    "triples_dropped_review",
    "triples_dropped_end_not_kept",
]
COMMONS_FILE_URL = "https://commons.wikimedia.org/wiki/File:"
# Artist texts (after normalize_artist, lower case) that name nobody. Under an
# attribution licence they cannot serve as the credit the licence requires.
NO_AUTHOR_TEXTS = {"unknown", "unknown author", "author unknown", "anonymous"}
# Licence boilerplate pasted into the Artist field ("... hereby publishes it under ...").
NO_AUTHOR_PHRASE = "hereby publish"
# Commons' self-published template reads "<name>, the copyright holder of this work,
# hereby publishes it ...": the text before this marker is the author to credit.
COPYRIGHT_HOLDER_MARKER = ", the copyright holder of this work"
TRAILING_PUNCTUATION = " .,;:"


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
    download_width: int


def load_mmkg_build_config(path: str | Path) -> MmkgBuildConfig:
    """Read the build config YAML (a mistyped key is an error)."""
    config = load_yaml_dataclass(path, MmkgBuildConfig)
    if "YYYYMMDD" in config.snapshot_dir:
        raise ValueError(
            f"snapshot_dir {config.snapshot_dir!r} is still the placeholder; "
            "set it to the collected snapshot folder in the build config"
        )
    return config


@dataclass(frozen=True)
class Review:
    """Human decisions from review.csv.

    status is keyed by (kind, key); fix_name_vi maps an entity id to a corrected label.
    """

    status: dict[tuple[str, str], str]
    fix_name_vi: dict[str, str]


def check_review_header(path: Path, header: list[str]) -> None:
    """Fail unless the header has exactly the REVIEW_COLUMNS (in any order)."""
    missing = sorted(set(REVIEW_COLUMNS) - set(header))
    extra = sorted(set(header) - set(REVIEW_COLUMNS))
    if missing or extra:
        raise ValueError(
            f"{path}: wrong review header; missing columns {missing}, extra columns {extra} "
            f"(expected {REVIEW_COLUMNS})"
        )


def read_review_header(path: Path) -> list[str]:
    """Column names of an existing review file, stripped and without a byte order mark."""
    # utf-8-sig drops the byte order mark Excel writes before the first column name.
    with open(path, encoding="utf-8-sig", newline="") as file:
        first_row = next(csv.reader(file), [])
    return [name.strip() for name in first_row]


def read_review(path: Path) -> Review:
    """Read and validate review.csv. A missing or empty file means no decisions yet.

    A sheet re-saved by Excel or Google Sheets may gain a byte order mark, stray
    spaces or a new column order; all three are accepted. A missing or unknown
    column is an error, because every decision would otherwise be read wrongly.
    """
    if not path.exists() or path.stat().st_size == 0:
        return Review({}, {})

    header = read_review_header(path)
    check_review_header(path, header)
    status: dict[tuple[str, str], str] = {}
    fix_name_vi: dict[str, str] = {}
    with open(path, encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, fieldnames=header)
        next(reader)  # the header row, already read and checked above
        # Row 1 is the header, so the first data row is row 2 (matches a spreadsheet).
        for row_number, raw_row in enumerate(reader, start=2):
            if None in raw_row:
                raise ValueError(f"{path.name} row {row_number}: more cells than columns")
            # A short row gives None for its last cells; treat them as empty.
            row = {column: (raw_row[column] or "").strip() for column in REVIEW_COLUMNS}
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


def normalize_artist(text: str) -> str:
    """Clean an Artist text: drop zero-width spaces, collapse a doubled phrase, strip end dots.

    Snapshots collected before strip_html put a space per tag hold
    "Unknown authorUnknown author." (the same phrase twice, glued).
    """
    cleaned = " ".join(text.replace("\u200b", "").split())
    cleaned = cleaned.rstrip(TRAILING_PUNCTUATION)
    # A phrase of at least 4 characters followed by itself; the minimum keeps a
    # short user name such as "nana" from being cut in half.
    doubled = re.fullmatch(r"(.{4,}?)\s*\1", cleaned)
    if doubled is not None:
        cleaned = doubled.group(1).rstrip(TRAILING_PUNCTUATION)
    return cleaned


def name_before_licence_sentence(artist: str) -> str:
    """Keep only "<name>" from "<name>, the copyright holder of this work, hereby ...".

    Text without the marker, or with nothing in front of it, is returned unchanged,
    so names_an_author still rejects a licence sentence that names nobody.
    """
    position = artist.lower().find(COPYRIGHT_HOLDER_MARKER)
    if position <= 0:
        return artist
    return artist[:position].strip()


def names_an_author(artist: str) -> bool:
    """False for empty text, "unknown"-like texts and licence boilerplate."""
    lowered = artist.lower()
    if lowered == "" or lowered in NO_AUTHOR_TEXTS:
        return False
    return NO_AUTHOR_PHRASE not in lowered


def pick_author(license_name: str, credit: dict[str, str], config: MmkgBuildConfig) -> str | None:
    """Author text for the image, or None when attribution is required but unknown.

    A licence like CC BY needs a real name in the credit; when the Artist field
    names nobody, the uploader is the best name we have, else the image is dropped.
    Public domain and CC0 need no credit, so "unknown" is enough there.
    """
    artist = name_before_licence_sentence(normalize_artist(credit.get("artist", "")))
    if names_an_author(artist):
        return artist
    if starts_with_any(license_name, config.attribution_license_prefixes):
        uploader = credit.get("uploader", "").strip()
        if uploader:
            return f"uploader: {uploader}"
        return None
    return "unknown"


def check_no_repeated_pairs(candidates: list[dict[str, str]]) -> None:
    """Fail on a repeated (entity_id, file_name): the collector never writes one."""
    seen: set[tuple[str, str]] = set()
    for row in candidates:
        pair = (row["entity_id"], row["file_name"])
        if pair in seen:
            raise ValueError(
                f"snapshot repeats entity {pair[0]!r} with file {pair[1]!r}; "
                "the collector writes one row per pair, so the snapshot is corrupt"
            )
        seen.add(pair)


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
    check_no_repeated_pairs(candidates)
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


@dataclass(frozen=True)
class BuiltMmkg:
    """The official tables, the report, and review rows still to be appended."""

    entities: list[dict[str, str]]
    triples: list[dict[str, str]]
    images: list[dict[str, str]]
    report: dict
    pending: list[dict[str, str]]


def review_dropped_entities(
    raw_entities: list[dict[str, str]], review: Review, report: Counter
) -> set[str]:
    """Entity ids a member removed; they can be neither an item nor a concept."""
    dropped: set[str] = set()
    for row in raw_entities:
        if review.status.get(("entity", row["entity_id"])) == "drop":
            dropped.add(row["entity_id"])
    report["entities_dropped_review"] = len(dropped)
    return dropped


def merge_triples(
    raw_triples: list[dict[str, str]], review: Review, config: MmkgBuildConfig, report: Counter
) -> dict[tuple[str, str, str], set[str]]:
    """Drop unknown/review-dropped rows, rename properties, merge on (head, relation, tail)."""
    codes_of_triple: dict[tuple[str, str, str], set[str]] = {}
    for row in raw_triples:
        code = row["property"]
        head = row["head_id"]
        tail = row["tail_id"]
        if code not in config.relations:
            report["triples_dropped_unknown_relation"] += 1
            continue
        if review.status.get(("triple", f"{head}|{code}|{tail}")) == "drop":
            report["triples_dropped_review"] += 1
            continue
        key = (head, config.relations[code], tail)
        codes_of_triple.setdefault(key, set()).add(code)
    return codes_of_triple


def find_concepts(
    codes_of_triple: dict[tuple[str, str, str], set[str]],
    items: set[str],
    excluded: set[str],
    min_items: int,
) -> set[str]:
    """Non-item entities that are the tail of triples from at least `min_items` distinct items."""
    heads_of_tail: dict[str, set[str]] = {}
    for head, _relation, tail in codes_of_triple:
        if head in items and tail not in items and tail not in excluded and head != tail:
            heads_of_tail.setdefault(tail, set()).add(head)
    return {tail for tail, heads in heads_of_tail.items() if len(heads) >= min_items}


def keep_triples(
    codes_of_triple: dict[tuple[str, str, str], set[str]], kept_ids: set[str], report: Counter
) -> dict[tuple[str, str, str], set[str]]:
    """Keep triples whose both ends are kept entities and that are not self-loops."""
    kept: dict[tuple[str, str, str], set[str]] = {}
    for (head, relation, tail), codes in codes_of_triple.items():
        if head == tail or head not in kept_ids or tail not in kept_ids:
            report["triples_dropped_end_not_kept"] += 1
        else:
            kept[(head, relation, tail)] = codes
    return kept


def make_entity_rows(
    raw_by_id: dict[str, dict[str, str]],
    items: set[str],
    concepts: set[str],
    review: Review,
    report: Counter,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for entity_id in sorted(items | concepts):
        raw = raw_by_id[entity_id]
        name_vi = review.fix_name_vi.get(entity_id, raw["name_vi"])
        if not name_vi:
            report["entities_missing_name_vi"] += 1
        rows.append(
            {
                "entity_id": entity_id,
                "name_vi": name_vi,
                "name_en": raw["name_en"],
                "type": "item" if entity_id in items else "concept",
                "source_url": WIKIDATA_ENTITY_URL + entity_id,
            }
        )
    return rows


def make_triple_rows(kept: dict[tuple[str, str, str], set[str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for head, relation, tail in sorted(kept):
        rows.append(
            {
                "head_id": head,
                "relation": relation,
                "tail_id": tail,
                "source": WIKIDATA_ENTITY_URL + head,
                "wikidata_property": "|".join(sorted(kept[(head, relation, tail)])),
            }
        )
    return rows


def official_review_keys(
    entities: list[dict[str, str]],
    kept: dict[tuple[str, str, str], set[str]],
    images: list[dict[str, str]],
) -> list[tuple[str, str]]:
    """Every (kind, key) a member should review, sorted."""
    keys: set[tuple[str, str]] = set()
    for row in images:
        keys.add(("image", row["file_name"]))
    for row in entities:
        keys.add(("entity", row["entity_id"]))
    for (head, _relation, tail), codes in kept.items():
        for code in codes:
            keys.add(("triple", f"{head}|{code}|{tail}"))
    return sorted(keys)


def check_entities_exist(
    raw_by_id: dict[str, dict[str, str]],
    candidates: list[dict[str, str]],
    triples: list[dict[str, str]],
) -> None:
    """Fail when a candidate or triple names an entity missing from entities_raw.

    The collector always writes these entities, so a gap means a corrupt snapshot;
    dropping the rows silently would break the counts in the report.
    """
    referenced: set[str] = set()
    for row in candidates:
        referenced.add(row["entity_id"])
    for row in triples:
        referenced.add(row["head_id"])
        referenced.add(row["tail_id"])
    missing = sorted(referenced - set(raw_by_id))
    if missing:
        raise ValueError(
            f"{len(missing)} entity ids are used but missing from entities_raw "
            f"(first 10: {missing[:10]}); the snapshot is corrupt"
        )


def unmatched_review_keys(
    review: Review,
    raw_entities: list[dict[str, str]],
    raw_triples: list[dict[str, str]],
    raw_candidates: list[dict[str, str]],
) -> list[str]:
    """Review rows whose key names nothing in the raw snapshot, as sorted "kind:key".

    Such a row is usually a typo in a file name or QID: its decision would be
    ignored without a word, so the build lists it instead.
    """
    known: set[tuple[str, str]] = set()
    for row in raw_candidates:
        known.add(("image", row["file_name"]))
    for row in raw_triples:
        known.add(("triple", f"{row['head_id']}|{row['property']}|{row['tail_id']}"))
    for row in raw_entities:
        known.add(("entity", row["entity_id"]))
    unmatched = []
    for kind, key in review.status:
        if (kind, key) not in known:
            unmatched.append(f"{kind}:{key}")
    return sorted(unmatched)


def build_mmkg(snapshot_dir: Path, review: Review, config: MmkgBuildConfig) -> BuiltMmkg:
    """Build the official tables from a raw snapshot. Reads files, writes nothing."""
    raw_entities = read_rows(snapshot_dir, "entities_raw")
    raw_triples = read_rows(snapshot_dir, "triples_raw")
    raw_candidates = read_rows(snapshot_dir, "image_candidates")
    credits = read_rows(snapshot_dir, "image_credits")
    report: Counter = Counter()

    dropped_entities = review_dropped_entities(raw_entities, review, report)
    candidates = [r for r in raw_candidates if r["entity_id"] not in dropped_entities]
    report["images_dropped_entity_review_drop"] = len(raw_candidates) - len(candidates)

    raw_by_id = {row["entity_id"]: row for row in raw_entities}
    check_entities_exist(raw_by_id, raw_candidates, raw_triples)
    chosen = select_images(candidates, credits, review, config, report)
    items = set(chosen)

    codes_of_triple = merge_triples(raw_triples, review, config, report)
    concepts = find_concepts(codes_of_triple, items, dropped_entities, config.min_items_per_concept)
    kept = keep_triples(codes_of_triple, items | concepts, report)

    entities = make_entity_rows(raw_by_id, items, concepts, review, report)
    triples = make_triple_rows(kept)
    images: list[dict[str, str]] = []
    for entity_id in sorted(items):
        images.extend(chosen[entity_id])

    pending: list[dict[str, str]] = []
    pending_count = {"image": 0, "triple": 0, "entity": 0}
    for kind, key in official_review_keys(entities, kept, images):
        status = review.status.get((kind, key))
        if status is None:
            pending.append(
                {
                    "kind": kind,
                    "key": key,
                    "review_status": "pending",
                    "reason": "",
                    "fix_name_vi": "",
                    "reviewer": "",
                    "date": "",
                }
            )
        if status is None or status == "pending":
            pending_count[kind] += 1

    full_report: dict = {name: report[name] for name in DROP_COUNTERS}
    full_report.update(
        {
            "snapshot_dir": str(snapshot_dir),
            "raw_entities": len(raw_entities),
            "raw_triples": len(raw_triples),
            "raw_image_candidates": len(raw_candidates),
            "entities_item": len(items),
            "entities_concept": len(concepts),
            "triples": len(triples),
            "images": len(images),
            "pending_image": pending_count["image"],
            "pending_triple": pending_count["triple"],
            "pending_entity": pending_count["entity"],
            "entities_missing_name_vi": report["entities_missing_name_vi"],
            "review_keys_unmatched": unmatched_review_keys(
                review, raw_entities, raw_triples, raw_candidates
            ),
        }
    )
    return BuiltMmkg(entities, triples, images, full_report, pending)


def append_pending(review_file: Path, pending: list[dict[str, str]]) -> None:
    """Append pending rows to review.csv; existing bytes are never rewritten.

    Rows follow the file's own column order, so a sheet re-saved with its columns
    moved stays aligned. A missing or empty (0-byte) file first gets the header.
    """
    is_new = not review_file.exists() or review_file.stat().st_size == 0
    review_file.parent.mkdir(parents=True, exist_ok=True)
    columns = REVIEW_COLUMNS
    # A hand-saved file may lack the final newline; without one, our first row
    # would be glued onto the reviewer's last row and change their last cell.
    needs_terminator = False
    if not is_new:
        columns = read_review_header(review_file)
        check_review_header(review_file, columns)
        with open(review_file, "rb") as existing:
            existing.seek(-1, 2)
            needs_terminator = existing.read(1) != b"\n"
    with open(review_file, "a", encoding="utf-8", newline="") as file:
        if needs_terminator:
            file.write("\r\n")
        writer = csv.DictWriter(file, fieldnames=columns)
        if is_new:
            writer.writeheader()
        writer.writerows(pending)


def write_built_mmkg(
    built: BuiltMmkg, output_dir: Path, review_file: Path, max_file_bytes: int
) -> None:
    """Write the three tables and build_report.json, then append pending review rows."""
    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(output_dir, "entities", ENTITY_COLUMNS, built.entities, max_file_bytes)
    write_rows(output_dir, "triples", TRIPLE_COLUMNS, built.triples, max_file_bytes)
    write_rows(output_dir, "images", IMAGE_COLUMNS, built.images, max_file_bytes)
    report_text = json.dumps(built.report, indent=2, sort_keys=True, ensure_ascii=False)
    (output_dir / "build_report.json").write_text(report_text + "\n", encoding="utf-8")
    append_pending(review_file, built.pending)
