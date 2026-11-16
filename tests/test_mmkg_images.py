"""Tests for the MMKG image downloader. A fake client answers; no network."""

import hashlib
from pathlib import Path

from mmxai.data.mmkg_images import download_images

GOOD_BYTES = b"original image bytes"
GOOD_SHA1 = hashlib.sha1(GOOD_BYTES).hexdigest()


class FakeClient:
    """Answers imageinfo queries and downloads from a dict {file name: bytes}."""

    def __init__(self, files: dict[str, bytes], missing: tuple[str, ...] = ()):
        self.files = files
        self.missing = missing
        self.download_calls: list[str] = []
        self.commons_calls: list[dict[str, str]] = []

    def commons(self, params: dict[str, str]) -> dict:
        self.commons_calls.append(params)
        pages = {}
        for index, title in enumerate(params["titles"].split("|")):
            name = title[len("File:") :]
            if name in self.missing:
                pages[str(-1 - index)] = {"title": title, "missing": ""}
            else:
                url = "https://upload.example/" + name.replace(" ", "_")
                pages[str(index + 1)] = {"title": title, "imageinfo": [{"url": url}]}
        return {"query": {"pages": pages}}

    def download(self, url: str) -> bytes:
        self.download_calls.append(url)
        name = url.rsplit("/", 1)[-1].replace("_", " ")
        return self.files[name]


def image_row(image_id="img1", file_name="Ngoc Lu.JPG", sha1=GOOD_SHA1):
    return {"image_id": image_id, "file_name": file_name, "sha1": sha1}


def test_downloads_and_checks_sha1(tmp_path: Path):
    client = FakeClient({"Ngoc Lu.JPG": GOOD_BYTES})

    counts = download_images([image_row()], tmp_path, client)

    assert counts == {
        "downloaded": 1,
        "already_present": 0,
        "sha1_mismatch": 0,
        "missing_on_commons": 0,
    }
    assert (tmp_path / "img1.jpg").read_bytes() == GOOD_BYTES
    assert not list(tmp_path.glob("*.part"))


def test_existing_file_with_right_sha1_is_skipped(tmp_path: Path):
    (tmp_path / "img1.jpg").write_bytes(GOOD_BYTES)
    client = FakeClient({})

    counts = download_images([image_row()], tmp_path, client)

    assert counts["already_present"] == 1
    assert counts["downloaded"] == 0
    assert client.download_calls == []
    assert client.commons_calls == []


def test_existing_file_with_wrong_sha1_is_downloaded_again(tmp_path: Path):
    (tmp_path / "img1.jpg").write_bytes(b"corrupt")
    client = FakeClient({"Ngoc Lu.JPG": GOOD_BYTES})

    counts = download_images([image_row()], tmp_path, client)

    assert counts["downloaded"] == 1
    assert counts["already_present"] == 0
    assert (tmp_path / "img1.jpg").read_bytes() == GOOD_BYTES


def test_sha1_mismatch_is_not_saved(tmp_path: Path):
    client = FakeClient({"Ngoc Lu.JPG": b"different bytes"})

    counts = download_images([image_row()], tmp_path, client)

    assert counts["sha1_mismatch"] == 1
    assert counts["downloaded"] == 0
    assert list(tmp_path.iterdir()) == []


def test_missing_on_commons_is_counted_and_not_saved(tmp_path: Path):
    client = FakeClient({}, missing=("Ngoc Lu.JPG",))

    counts = download_images([image_row()], tmp_path, client)

    assert counts["missing_on_commons"] == 1
    assert client.download_calls == []
    assert list(tmp_path.iterdir()) == []


def test_lookups_are_batched_by_fifty(tmp_path: Path):
    files = {}
    rows = []
    for number in range(51):
        name = f"File {number}.png"
        files[name] = GOOD_BYTES
        rows.append(image_row(image_id=f"img{number}", file_name=name))
    client = FakeClient(files)

    counts = download_images(rows, tmp_path, client)

    assert counts["downloaded"] == 51
    assert len(client.commons_calls) == 2
