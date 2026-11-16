"""Download the selected MMKG images and check each against the Commons sha1.

Only originals are downloaded (never thumbnails), so the sha1 recorded in
images.csv can be compared with the bytes on disk. A copy that passes the check
is exactly the image the team reviewed.
"""

import hashlib
from pathlib import Path

from mmxai.data.mmkg_collect import normalize_file_name
from mmxai.data.wikimedia import WikimediaClient

LOOKUP_BATCH_SIZE = 50


def _image_path(images_dir: Path, image: dict[str, str]) -> Path:
    """Return `<image_id><ext>` with the extension taken from the Commons file name."""
    extension = Path(image["file_name"]).suffix.lower()
    return images_dir / (image["image_id"] + extension)


def _sha1_of_file(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def _look_up_urls(file_names: list[str], client: WikimediaClient) -> dict[str, str]:
    """Return {file name: original URL}. Names Commons does not know are left out."""
    titles = "|".join("File:" + name for name in file_names)
    params = {"action": "query", "prop": "imageinfo", "iiprop": "url", "titles": titles}
    response = client.commons(params)

    urls: dict[str, str] = {}
    for page in response["query"]["pages"].values():
        if "missing" in page or "imageinfo" not in page:
            continue
        name = normalize_file_name(page["title"])
        urls[name] = page["imageinfo"][0]["url"]
    return urls


def download_images(
    images: list[dict[str, str]], images_dir: Path, client: WikimediaClient
) -> dict[str, int]:
    """Download the originals of `images` into `images_dir`, verifying sha1.

    A file already on disk with the right sha1 is kept. A file with the wrong sha1
    is treated as absent and downloaded again. Downloaded bytes with the wrong sha1
    are not saved. Returns counts: downloaded, already_present, sha1_mismatch,
    missing_on_commons.
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    counts = {
        "downloaded": 0,
        "already_present": 0,
        "sha1_mismatch": 0,
        "missing_on_commons": 0,
    }

    to_fetch: list[dict[str, str]] = []
    for image in images:
        path = _image_path(images_dir, image)
        if path.exists() and _sha1_of_file(path) == image["sha1"]:
            counts["already_present"] += 1
        else:
            to_fetch.append(image)

    for start in range(0, len(to_fetch), LOOKUP_BATCH_SIZE):
        batch = to_fetch[start : start + LOOKUP_BATCH_SIZE]
        names = [normalize_file_name(image["file_name"]) for image in batch]
        urls = _look_up_urls(names, client)

        for image, name in zip(batch, names, strict=True):
            if name not in urls:
                counts["missing_on_commons"] += 1
                continue
            data = client.download(urls[name])
            if hashlib.sha1(data).hexdigest() != image["sha1"]:
                counts["sha1_mismatch"] += 1
                continue
            # Write to a temporary name first so an interrupted run never leaves
            # a half-written file that looks like a finished image.
            path = _image_path(images_dir, image)
            part_path = path.with_name(path.name + ".part")
            part_path.write_bytes(data)
            part_path.replace(path)
            counts["downloaded"] += 1
    return counts
