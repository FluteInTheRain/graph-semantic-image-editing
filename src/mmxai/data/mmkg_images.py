"""Download the selected MMKG images (960 px thumbnails) and record a sha1 manifest.

Wikimedia throttled the download of originals, and its robot policy allows
"originals or one of our pregenerated thumbnail sizes". So each image is fetched
as a thumbnail of `download_width` px (CLIP only needs 224 px). Commons has no sha1
for a thumbnail, so we hash the bytes ourselves and write them to
`images_manifest.csv`; a teammate can check a cloud copy against that file.
When a thumbnail would be too small, the original is downloaded and checked
against the Commons sha1 in images.csv.
"""

import hashlib
from collections.abc import Callable
from pathlib import Path

from mmxai.data.mmkg_collect import normalize_file_name
from mmxai.data.mmkg_csv import read_rows, write_rows
from mmxai.data.wikimedia import WikimediaClient

LOOKUP_BATCH_SIZE = 50
PROGRESS_EVERY = 25
# Same as min_short_side_px in the build config: CLIP ViT-B-32 input size.
MIN_THUMBNAIL_SHORT_SIDE = 224
MANIFEST_STEM = "images_manifest"
MANIFEST_COLUMNS = ["image_id", "file_name", "kind", "url", "width", "height", "bytes", "sha1"]


def _image_path(images_dir: Path, image: dict[str, str]) -> Path:
    """Return `<image_id><ext>` with the extension taken from the Commons file name."""
    extension = Path(image["file_name"]).suffix.lower()
    return images_dir / (image["image_id"] + extension)


def _sha1_of_file(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def _look_up_info(file_names: list[str], width: int, client: WikimediaClient) -> dict[str, dict]:
    """Return {file name: imageinfo dict}. Names Commons does not know are left out."""
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
    for page in response["query"]["pages"].values():
        if "missing" in page or "imageinfo" not in page:
            continue
        info[normalize_file_name(page["title"])] = page["imageinfo"][0]
    return info


def _choose_source(info: dict) -> tuple[str, str, str, str]:
    """Return (kind, url, width, height) to download for one imageinfo.

    A thumbnail is used when Commons gives one whose short side is at least
    MIN_THUMBNAIL_SHORT_SIDE. Otherwise (no thumbnail, or too small) the original.
    """
    thumb_url = info.get("thumburl")
    thumb_width = info.get("thumbwidth", 0)
    thumb_height = info.get("thumbheight", 0)
    if thumb_url and min(thumb_width, thumb_height) >= MIN_THUMBNAIL_SHORT_SIDE:
        return "thumbnail", thumb_url, str(thumb_width), str(thumb_height)
    return "original", info["url"], str(info["width"]), str(info["height"])


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


def download_images(
    images: list[dict[str, str]],
    images_dir: Path,
    client: WikimediaClient,
    manifest_dir: Path,
    download_width: int,
    max_file_bytes: int,
    log: Callable[[str], None] | None = None,
) -> dict[str, int]:
    """Download thumbnails (or originals when too small) of `images` into `images_dir`.

    A file is already present when `images_manifest.csv` has a row for it and the
    file's sha1 matches that row; a file without a row counts when its sha1 equals
    the Commons sha1 (an original downloaded earlier) and is added to the manifest.
    A downloaded original with the wrong Commons sha1 is not saved. The manifest is
    rewritten after every batch so an interrupted run keeps its progress.

    Returns counts: downloaded_thumbnail, downloaded_original, already_present,
    sha1_mismatch, missing_on_commons.
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    counts = {
        "downloaded_thumbnail": 0,
        "downloaded_original": 0,
        "already_present": 0,
        "sha1_mismatch": 0,
        "missing_on_commons": 0,
    }
    old_manifest = _read_manifest(manifest_dir)
    manifest: dict[str, dict[str, str]] = {}

    to_fetch: list[dict[str, str]] = []
    for image in images:
        path = _image_path(images_dir, image)
        row = old_manifest.get(image["image_id"])
        if path.exists() and row is not None and _sha1_of_file(path) == row["sha1"]:
            manifest[image["image_id"]] = row
            counts["already_present"] += 1
        elif path.exists() and row is None and _sha1_of_file(path) == image["sha1"]:
            manifest[image["image_id"]] = {
                "image_id": image["image_id"],
                "file_name": image["file_name"],
                "kind": "original",
                "url": image["source_url"],
                "width": image["width"],
                "height": image["height"],
                "bytes": str(path.stat().st_size),
                "sha1": image["sha1"],
            }
            counts["already_present"] += 1
        else:
            to_fetch.append(image)
    _write_manifest(manifest_dir, manifest, max_file_bytes)

    processed = 0
    for start in range(0, len(to_fetch), LOOKUP_BATCH_SIZE):
        batch = to_fetch[start : start + LOOKUP_BATCH_SIZE]
        names = [normalize_file_name(image["file_name"]) for image in batch]
        infos = _look_up_info(names, download_width, client)

        for image, name in zip(batch, names, strict=True):
            processed += 1
            if log is not None and processed % PROGRESS_EVERY == 0:
                log(f"progress: {processed}/{len(to_fetch)} files handled")
            if name not in infos:
                counts["missing_on_commons"] += 1
                continue
            kind, url, width, height = _choose_source(infos[name])
            data = client.download(url)
            sha1 = hashlib.sha1(data).hexdigest()
            # Commons only knows the sha1 of the original, so only an original
            # can be compared; a thumbnail is trusted and recorded as downloaded.
            if kind == "original" and sha1 != image["sha1"]:
                counts["sha1_mismatch"] += 1
                continue
            _save(_image_path(images_dir, image), data)
            counts["downloaded_" + kind] += 1
            manifest[image["image_id"]] = {
                "image_id": image["image_id"],
                "file_name": image["file_name"],
                "kind": kind,
                "url": url,
                "width": width,
                "height": height,
                "bytes": str(len(data)),
                "sha1": sha1,
            }
        _write_manifest(manifest_dir, manifest, max_file_bytes)
    return counts
