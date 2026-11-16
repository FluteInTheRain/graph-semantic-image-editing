"""Tests for the MMKG image downloader. A fake client answers; no network."""

import hashlib
from pathlib import Path

from mmxai.data.mmkg_csv import read_rows
from mmxai.data.mmkg_images import download_images

ORIGINAL_BYTES = b"original image bytes"
ORIGINAL_SHA1 = hashlib.sha1(ORIGINAL_BYTES).hexdigest()
THUMB_BYTES = b"thumbnail image bytes"
THUMB_SHA1 = hashlib.sha1(THUMB_BYTES).hexdigest()
WIDTH = 960
MAX_FILE_BYTES = 100000


class FakeClient:
    """Answers imageinfo queries and downloads.

    `thumb_sizes` maps a file name to the (width, height) of its thumbnail; a name
    not in it gets a 960x640 thumbnail. Thumbnail URLs contain "/thumb/".
    """

    def __init__(
        self,
        originals: dict[str, bytes],
        missing: tuple[str, ...] = (),
        thumb_sizes: dict[str, tuple[int, int]] | None = None,
    ):
        self.originals = originals
        self.missing = missing
        self.thumb_sizes = thumb_sizes or {}
        self.download_calls: list[str] = []
        self.commons_calls: list[dict[str, str]] = []

    def commons(self, params: dict[str, str]) -> dict:
        self.commons_calls.append(params)
        pages = {}
        for index, title in enumerate(params["titles"].split("|")):
            name = title[len("File:") :]
            if name in self.missing:
                pages[str(-1 - index)] = {"title": title, "missing": ""}
                continue
            slug = name.replace(" ", "_")
            thumb_width, thumb_height = self.thumb_sizes.get(name, (960, 640))
            info = {
                "url": "https://upload.example/" + slug,
                "width": 4000,
                "height": 3000,
                "sha1": ORIGINAL_SHA1,
                "thumburl": "https://upload.example/thumb/" + slug,
                "thumbwidth": thumb_width,
                "thumbheight": thumb_height,
            }
            pages[str(index + 1)] = {"title": title, "imageinfo": [info]}
        return {"query": {"pages": pages}}

    def download(self, url: str) -> bytes:
        self.download_calls.append(url)
        name = url.rsplit("/", 1)[-1].replace("_", " ")
        if "/thumb/" in url:
            return THUMB_BYTES
        return self.originals[name]


def run(images, images_dir: Path, client: FakeClient) -> dict[str, int]:
    return download_images(images, images_dir, client, images_dir.parent, WIDTH, MAX_FILE_BYTES)


def manifest(images_dir: Path) -> dict[str, dict[str, str]]:
    rows = read_rows(images_dir.parent, "images_manifest")
    return {row["image_id"]: row for row in rows}


def image_row(image_id="img1", file_name="Ngoc Lu.JPG", sha1=ORIGINAL_SHA1):
    return {
        "image_id": image_id,
        "file_name": file_name,
        "sha1": sha1,
        "source_url": "https://commons.example/File:" + file_name,
        "width": "4000",
        "height": "3000",
    }


def make_dir(tmp_path: Path) -> Path:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    return images_dir


def test_thumbnail_is_chosen_and_its_own_sha1_recorded(tmp_path: Path):
    images_dir = make_dir(tmp_path)
    client = FakeClient({})

    counts = run([image_row()], images_dir, client)

    assert counts == {
        "downloaded_thumbnail": 1,
        "downloaded_original": 0,
        "already_present": 0,
        "sha1_mismatch": 0,
        "missing_on_commons": 0,
    }
    assert client.commons_calls[0]["iiurlwidth"] == "960"
    assert "sha1" in client.commons_calls[0]["iiprop"]
    assert (images_dir / "img1.jpg").read_bytes() == THUMB_BYTES
    assert not list(images_dir.glob("*.part"))
    row = manifest(images_dir)["img1"]
    assert row["kind"] == "thumbnail"
    assert row["sha1"] == THUMB_SHA1
    assert (row["width"], row["height"]) == ("960", "640")
    assert row["bytes"] == str(len(THUMB_BYTES))
    assert "/thumb/" in row["url"]


def test_small_thumbnail_falls_back_to_verified_original(tmp_path: Path):
    images_dir = make_dir(tmp_path)
    client = FakeClient({"Ngoc Lu.JPG": ORIGINAL_BYTES}, thumb_sizes={"Ngoc Lu.JPG": (960, 200)})

    counts = run([image_row()], images_dir, client)

    assert counts["downloaded_original"] == 1
    assert counts["downloaded_thumbnail"] == 0
    assert (images_dir / "img1.jpg").read_bytes() == ORIGINAL_BYTES
    row = manifest(images_dir)["img1"]
    assert row["kind"] == "original"
    assert row["sha1"] == ORIGINAL_SHA1
    assert (row["width"], row["height"]) == ("4000", "3000")


def test_original_with_wrong_commons_sha1_is_not_saved(tmp_path: Path):
    images_dir = make_dir(tmp_path)
    client = FakeClient(
        {"Ngoc Lu.JPG": b"different bytes"}, thumb_sizes={"Ngoc Lu.JPG": (960, 200)}
    )

    counts = run([image_row()], images_dir, client)

    assert counts["sha1_mismatch"] == 1
    assert counts["downloaded_original"] == 0
    assert list(images_dir.iterdir()) == []


def test_existing_original_without_manifest_row_is_present_and_recorded(tmp_path: Path):
    images_dir = make_dir(tmp_path)
    (images_dir / "img1.jpg").write_bytes(ORIGINAL_BYTES)
    client = FakeClient({})

    counts = run([image_row()], images_dir, client)

    assert counts["already_present"] == 1
    assert client.download_calls == []
    assert client.commons_calls == []
    row = manifest(images_dir)["img1"]
    assert row["kind"] == "original"
    assert row["sha1"] == ORIGINAL_SHA1


def test_second_run_downloads_nothing(tmp_path: Path):
    images_dir = make_dir(tmp_path)
    run([image_row()], images_dir, FakeClient({}))
    client = FakeClient({})

    counts = run([image_row()], images_dir, client)

    assert counts["already_present"] == 1
    assert counts["downloaded_thumbnail"] == 0
    assert client.download_calls == []
    assert client.commons_calls == []
    assert manifest(images_dir)["img1"]["kind"] == "thumbnail"


def test_corrupted_thumbnail_is_downloaded_again(tmp_path: Path):
    images_dir = make_dir(tmp_path)
    run([image_row()], images_dir, FakeClient({}))
    (images_dir / "img1.jpg").write_bytes(b"corrupt")

    counts = run([image_row()], images_dir, FakeClient({}))

    assert counts["downloaded_thumbnail"] == 1
    assert (images_dir / "img1.jpg").read_bytes() == THUMB_BYTES


def test_missing_on_commons_is_counted_and_not_saved(tmp_path: Path):
    images_dir = make_dir(tmp_path)
    client = FakeClient({}, missing=("Ngoc Lu.JPG",))

    counts = run([image_row()], images_dir, client)

    assert counts["missing_on_commons"] == 1
    assert client.download_calls == []
    assert list(images_dir.iterdir()) == []


def test_manifest_is_sorted_by_image_id(tmp_path: Path):
    images_dir = make_dir(tmp_path)
    rows = [
        image_row(image_id="b2", file_name="B.png"),
        image_row(image_id="a1", file_name="A.png"),
    ]

    run(rows, images_dir, FakeClient({}))

    ids = [row["image_id"] for row in read_rows(tmp_path, "images_manifest")]
    assert ids == ["a1", "b2"]


def test_lookups_are_batched_by_fifty(tmp_path: Path):
    images_dir = make_dir(tmp_path)
    rows = [image_row(image_id=f"img{n}", file_name=f"File {n}.png") for n in range(51)]
    client = FakeClient({})

    counts = run(rows, images_dir, client)

    assert counts["downloaded_thumbnail"] == 51
    assert len(client.commons_calls) == 2
