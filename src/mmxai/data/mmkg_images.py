"""Download the selected MMKG images (960 px thumbnails) and record a sha1 manifest.

Wikimedia throttled the download of originals, and its robot policy allows
"originals or one of our pregenerated thumbnail sizes". So each image is fetched
as a thumbnail of `download_width` px (CLIP only needs 224 px). Commons has no sha1
for a thumbnail, so we hash the bytes ourselves and write them to
`images_manifest.csv`; a teammate can check a cloud copy against that file.

The original is downloaded instead, and checked against the Commons sha1 in
images.csv, when it is no wider than `download_width` (Commons then serves the
original itself as the "thumbnail"), when Commons gives no thumbnail, or when the
thumbnail's short side would be under 224 px. The manifest column `why` says which.
"""

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from mmxai.data.mmkg_collect import normalize_file_name
from mmxai.data.mmkg_csv import read_rows, write_rows
from mmxai.data.wikimedia import WikimediaClient

LOOKUP_BATCH_SIZE = 50
PROGRESS_EVERY = 25
# Same as min_short_side_px in the build config: CLIP ViT-B-32 input size.
MIN_THUMBNAIL_SHORT_SIDE = 224
MANIFEST_STEM = "images_manifest"
MANIFEST_COLUMNS = [
    "image_id",
    "file_name",
    "kind",
    "why",
    "url",
    "width",
    "height",
    "bytes",
    "sha1",
]
# Why a file is what it is (manifest column `why`).
WHY_THUMBNAIL = "thumbnail"
WHY_NOT_WIDER = "original_not_wider"
WHY_NO_THUMBURL = "original_no_thumburl"
WHY_THUMB_TOO_SMALL = "original_thumb_too_small"
# An original found on disk and verified, but the reason it was chosen is unknown.
WHY_ALREADY_PRESENT = "original_already_present"


@dataclass(frozen=True)
class DownloadResult:
    """Counts, plus every failure named so it can be checked by hand."""

    counts: dict[str, int]
    # {image_id, file_name, expected_sha1, actual_sha1} per original with a wrong sha1.
    sha1_mismatches: list[dict[str, str]]
    # {image_id, file_name, reason} per file Commons could not describe.
    missing: list[dict[str, str]]


def _image_path(images_dir: Path, image: dict[str, str]) -> Path:
    """Return `<image_id><ext>` with the extension taken from the Commons file name."""
    extension = Path(image["file_name"]).suffix.lower()
    return images_dir / (image["image_id"] + extension)


def _sha1_of_file(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def _is_not_wider(image: dict[str, str], download_width: int) -> bool:
    """True when the original is at most `download_width` px wide (images.csv width)."""
    return int(image["width"]) <= download_width


def find_stray_files(images_dir: Path, images: list[dict[str, str]]) -> list[Path]:
    """Files in `images_dir` whose stem is not an image_id of `images`, sorted.

    A stray file is usually an image the build has since dropped (for example an
    identifiable person) or a `.part` left by a crash; it must not reach the shared copy.
    """
    if not images_dir.exists():
        return []
    image_ids = {image["image_id"] for image in images}
    strays = []
    for path in images_dir.iterdir():
        if path.is_file() and path.stem not in image_ids:
            strays.append(path)
    return sorted(strays)


def _look_up_info(
    file_names: list[str], width: int, client: WikimediaClient
) -> tuple[dict[str, dict], dict[str, str]]:
    """Return ({file name: imageinfo}, {file name: reason it has no imageinfo})."""
    titles = "|".join("File:" + name for name in file_names)
    params = {
        "action": "query",
        "prop": "imageinfo",
        "iiprop": "url|size|sha1",
        "iiurlwidth": str(width),
        "titles": titles,
    }
    response = client.commons(params)

    info: dict[str, dict] = {}
    reasons: dict[str, str] = {}
    for page in response["query"]["pages"].values():
        name = normalize_file_name(page["title"])
        if "missing" in page:
            reasons[name] = "missing"
        elif "invalid" in page:
            reasons[name] = "invalid"
        elif "imageinfo" not in page:
            reasons[name] = "no_imageinfo"
        else:
            info[name] = page["imageinfo"][0]
    return info, reasons


def _choose_source(info: dict, download_width: int) -> tuple[str, str, str, str, str]:
    """Return (kind, why, url, width, height) to download for one imageinfo."""
    thumb_url = info.get("thumburl")
    thumb_width = info.get("thumbwidth", 0)
    thumb_height = info.get("thumbheight", 0)
    if not thumb_url:
        why = WHY_NO_THUMBURL
    elif info["width"] <= download_width:
        # Commons cannot scale up, so its "thumbnail" is the original file itself;
        # downloading it as an original lets us check it against the Commons sha1.
        why = WHY_NOT_WIDER
    elif min(thumb_width, thumb_height) < MIN_THUMBNAIL_SHORT_SIDE:
        why = WHY_THUMB_TOO_SMALL
    else:
        return "thumbnail", WHY_THUMBNAIL, thumb_url, str(thumb_width), str(thumb_height)
    return "original", why, info["url"], str(info["width"]), str(info["height"])


def _read_manifest(directory: Path) -> dict[str, dict[str, str]]:
    """Return the existing manifest rows keyed by image_id (empty if none yet)."""
    try:
        rows = read_rows(directory, MANIFEST_STEM)
    except FileNotFoundError:
        return {}
    return {row["image_id"]: row for row in rows}


def _write_manifest(directory: Path, rows: dict[str, dict[str, str]], max_file_bytes: int) -> None:
    sorted_rows = [rows[image_id] for image_id in sorted(rows)]
    write_rows(directory, MANIFEST_STEM, MANIFEST_COLUMNS, sorted_rows, max_file_bytes)


def _save(path: Path, data: bytes) -> None:
    # Write to a temporary name first so an interrupted run never leaves
    # a half-written file that looks like a finished image.
    part_path = path.with_name(path.name + ".part")
    part_path.write_bytes(data)
    part_path.replace(path)


def _original_row(image: dict[str, str], url: str, path: Path, why: str) -> dict[str, str]:
    """Manifest row of a verified original on disk (sizes come from images.csv)."""
    return {
        "image_id": image["image_id"],
        "file_name": image["file_name"],
        "kind": "original",
        "why": why,
        "url": url,
        "width": image["width"],
        "height": image["height"],
        "bytes": str(path.stat().st_size),
        "sha1": image["sha1"],
    }


def _check_file_on_disk(
    image: dict[str, str],
    path: Path,
    old_row: dict[str, str] | None,
    download_width: int,
    counts: dict[str, int],
    log: Callable[[str], None],
) -> dict[str, str] | None:
    """Return the manifest row of a good file already on disk, or None to download it.

    A file whose sha1 matches neither the Commons sha1 nor its manifest row is
    removed, so a half-written or outdated file never stays next to good ones.
    """
    if not path.exists():
        return None
    sha1 = _sha1_of_file(path)
    if sha1 == image["sha1"]:
        # The original itself, verified against Commons.
        if old_row is not None and old_row["kind"] == "original" and old_row.get("why"):
            return old_row
        url = image["source_url"]
        if old_row is not None:
            url = old_row["url"]
            if old_row["kind"] == "thumbnail":
                # Saved as a "thumbnail" before the not-wider rule existed.
                counts["relabelled_as_original"] += 1
        why = WHY_NOT_WIDER if _is_not_wider(image, download_width) else WHY_ALREADY_PRESENT
        return _original_row(image, url, path, why)
    is_valid_thumbnail = (
        old_row is not None
        and old_row["kind"] == "thumbnail"
        and sha1 == old_row["sha1"]
        and not _is_not_wider(image, download_width)
    )
    if is_valid_thumbnail:
        row = dict(old_row)
        row["why"] = WHY_THUMBNAIL
        return row
    log(f"removing {path.name}: its sha1 {sha1} matches neither the manifest nor Commons")
    path.unlink()
    counts["removed_bad_file"] += 1
    return None


def download_images(
    images: list[dict[str, str]],
    images_dir: Path,
    client: WikimediaClient,
    manifest_dir: Path,
    download_width: int,
    max_file_bytes: int,
    log: Callable[[str], None] = print,
) -> DownloadResult:
    """Download thumbnails (or verified originals) of `images` into `images_dir`.

    Files already on disk are checked first (see _check_file_on_disk). The manifest
    then holds exactly the images of `images` that are on disk, and it is rewritten
    after every saved file so an interrupted run loses no row.
    A downloaded original with the wrong Commons sha1 is not saved.
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    counts = {
        "downloaded_thumbnail": 0,
        "downloaded_original": 0,
        "already_present": 0,
        "relabelled_as_original": 0,
        "removed_bad_file": 0,
        "sha1_mismatch": 0,
        "missing_on_commons": 0,
    }
    sha1_mismatches: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []
    old_manifest = _read_manifest(manifest_dir)
    manifest: dict[str, dict[str, str]] = {}

    to_fetch: list[dict[str, str]] = []
    for image in images:
        path = _image_path(images_dir, image)
        old_row = old_manifest.get(image["image_id"])
        row = _check_file_on_disk(image, path, old_row, download_width, counts, log)
        if row is None:
            to_fetch.append(image)
        else:
            manifest[image["image_id"]] = row
            counts["already_present"] += 1
    _write_manifest(manifest_dir, manifest, max_file_bytes)

    processed = 0
    for start in range(0, len(to_fetch), LOOKUP_BATCH_SIZE):
        batch = to_fetch[start : start + LOOKUP_BATCH_SIZE]
        names = [normalize_file_name(image["file_name"]) for image in batch]
        infos, reasons = _look_up_info(names, download_width, client)

        for image, name in zip(batch, names, strict=True):
            processed += 1
            if processed % PROGRESS_EVERY == 0:
                log(f"progress: {processed}/{len(to_fetch)} files handled")
            if name not in infos:
                counts["missing_on_commons"] += 1
                # A title Commons did not answer about at all gets "not_returned".
                reason = reasons.get(name, "not_returned")
                missing.append(
                    {
                        "image_id": image["image_id"],
                        "file_name": image["file_name"],
                        "reason": reason,
                    }
                )
                continue
            kind, why, url, width, height = _choose_source(infos[name], download_width)
            data = client.download(url)
            sha1 = hashlib.sha1(data).hexdigest()
            # Commons only knows the sha1 of the original, so only an original
            # can be compared; a thumbnail is trusted and recorded as downloaded.
            if kind == "original" and sha1 != image["sha1"]:
                counts["sha1_mismatch"] += 1
                sha1_mismatches.append(
                    {
                        "image_id": image["image_id"],
                        "file_name": image["file_name"],
                        "expected_sha1": image["sha1"],
                        "actual_sha1": sha1,
                    }
                )
                continue
            _save(_image_path(images_dir, image), data)
            counts["downloaded_" + kind] += 1
            manifest[image["image_id"]] = {
                "image_id": image["image_id"],
                "file_name": image["file_name"],
                "kind": kind,
                "why": why,
                "url": url,
                "width": width,
                "height": height,
                "bytes": str(len(data)),
                "sha1": sha1,
            }
            _write_manifest(manifest_dir, manifest, max_file_bytes)
    return DownloadResult(counts, sha1_mismatches, missing)


def write_download_report(path: Path, result: DownloadResult) -> None:
    """Write the counts and every named failure as JSON."""
    text = json.dumps(asdict(result), indent=2, sort_keys=True, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8")
