"""Tests for the MMKG image downloader. A fake client answers; no network."""

import hashlib
import json
from pathlib import Path

import pytest

from mmxai.data.mmkg_csv import read_rows, write_rows
from mmxai.data.mmkg_images import (
    MANIFEST_COLUMNS,
    download_images,
    find_stray_files,
    write_download_report,
)

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
    `original_widths` maps a file name to the width of its original (default 4000).
    Like Commons, an original no wider than the asked width gets the original itself
    as `thumburl`, and `no_thumb` names get no `thumburl` at all.
    """

    def __init__(
        self,
        originals: dict[str, bytes],
        missing: tuple[str, ...] = (),
        thumb_sizes: dict[str, tuple[int, int]] | None = None,
        original_widths: dict[str, int] | None = None,
        no_thumb: tuple[str, ...] = (),
    ):
        self.originals = originals
        self.missing = missing
        self.thumb_sizes = thumb_sizes or {}
        self.original_widths = original_widths or {}
        self.no_thumb = no_thumb
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
            width = self.original_widths.get(name, 4000)
            thumb_width, thumb_height = self.thumb_sizes.get(name, (960, 640))
            info = {
                "url": "https://upload.example/" + slug,
                "width": width,
                "height": 3000,
                "sha1": ORIGINAL_SHA1,
                "thumburl": "https://upload.example/thumb/" + slug,
                "thumbwidth": thumb_width,
                "thumbheight": thumb_height,
            }
            if width <= int(params["iiurlwidth"]):
                # Commons answers with the original file, labelled as a thumbnail.
                info["thumburl"] = info["url"]
                info["thumbwidth"] = int(params["iiurlwidth"])
            if name in self.no_thumb:
                del info["thumburl"]
            pages[str(index + 1)] = {"title": title, "imageinfo": [info]}
        return {"query": {"pages": pages}}

    def download(self, url: str) -> bytes:
        self.download_calls.append(url)
        name = url.rsplit("/", 1)[-1].replace("_", " ")
        if "/thumb/" in url:
            return THUMB_BYTES
        return self.originals[name]


def run(images, images_dir: Path, client: FakeClient, log=None):
    log = log if log is not None else [].append
    return download_images(
        images, images_dir, client, images_dir.parent, WIDTH, MAX_FILE_BYTES, log=log
    )


def manifest(images_dir: Path) -> dict[str, dict[str, str]]:
    rows = read_rows(images_dir.parent, "images_manifest")
    return {row["image_id"]: row for row in rows}


def image_row(image_id="img1", file_name="Ngoc Lu.JPG", sha1=ORIGINAL_SHA1, width="4000"):
    return {
        "image_id": image_id,
        "file_name": file_name,
        "sha1": sha1,
        "source_url": "https://commons.example/File:" + file_name,
        "width": width,
        "height": "3000",
    }


def make_dir(tmp_path: Path) -> Path:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    return images_dir


def test_thumbnail_is_chosen_and_its_own_sha1_recorded(tmp_path: Path):
    images_dir = make_dir(tmp_path)
    client = FakeClient({})

    counts = run([image_row()], images_dir, client).counts

    assert counts == {
        "downloaded_thumbnail": 1,
        "downloaded_original": 0,
        "already_present": 0,
        "relabelled_as_original": 0,
        "removed_bad_file": 0,
        "sha1_mismatch": 0,
        "missing_on_commons": 0,
    }
    assert client.commons_calls[0]["iiurlwidth"] == "960"
    assert "sha1" in client.commons_calls[0]["iiprop"]
    assert (images_dir / "img1.jpg").read_bytes() == THUMB_BYTES
    assert not list(images_dir.glob("*.part"))
    row = manifest(images_dir)["img1"]
    assert row["kind"] == "thumbnail"
    assert row["why"] == "thumbnail"
    assert row["sha1"] == THUMB_SHA1
    assert (row["width"], row["height"]) == ("960", "640")
    assert row["bytes"] == str(len(THUMB_BYTES))
    assert "/thumb/" in row["url"]


def test_small_thumbnail_falls_back_to_verified_original(tmp_path: Path):
    images_dir = make_dir(tmp_path)
    client = FakeClient({"Ngoc Lu.JPG": ORIGINAL_BYTES}, thumb_sizes={"Ngoc Lu.JPG": (960, 200)})

    counts = run([image_row()], images_dir, client).counts

    assert counts["downloaded_original"] == 1
    assert counts["downloaded_thumbnail"] == 0
    assert (images_dir / "img1.jpg").read_bytes() == ORIGINAL_BYTES
    row = manifest(images_dir)["img1"]
    assert row["kind"] == "original"
    assert row["why"] == "original_thumb_too_small"
    assert row["sha1"] == ORIGINAL_SHA1
    assert (row["width"], row["height"]) == ("4000", "3000")


def test_original_with_wrong_commons_sha1_is_not_saved(tmp_path: Path):
    images_dir = make_dir(tmp_path)
    client = FakeClient(
        {"Ngoc Lu.JPG": b"different bytes"}, thumb_sizes={"Ngoc Lu.JPG": (960, 200)}
    )

    counts = run([image_row()], images_dir, client).counts

    assert counts["sha1_mismatch"] == 1
    assert counts["downloaded_original"] == 0
    assert list(images_dir.iterdir()) == []


def test_existing_original_without_manifest_row_is_present_and_recorded(tmp_path: Path):
    images_dir = make_dir(tmp_path)
    (images_dir / "img1.jpg").write_bytes(ORIGINAL_BYTES)
    client = FakeClient({})

    counts = run([image_row()], images_dir, client).counts

    assert counts["already_present"] == 1
    assert client.download_calls == []
    assert client.commons_calls == []
    row = manifest(images_dir)["img1"]
    assert row["kind"] == "original"
    assert row["why"] == "original_already_present"
    assert row["sha1"] == ORIGINAL_SHA1


def test_second_run_downloads_nothing(tmp_path: Path):
    images_dir = make_dir(tmp_path)
    run([image_row()], images_dir, FakeClient({}))
    client = FakeClient({})

    counts = run([image_row()], images_dir, client).counts

    assert counts["already_present"] == 1
    assert counts["downloaded_thumbnail"] == 0
    assert client.download_calls == []
    assert client.commons_calls == []
    assert manifest(images_dir)["img1"]["kind"] == "thumbnail"


def test_corrupted_thumbnail_is_downloaded_again(tmp_path: Path):
    images_dir = make_dir(tmp_path)
    run([image_row()], images_dir, FakeClient({}))
    (images_dir / "img1.jpg").write_bytes(b"corrupt")

    counts = run([image_row()], images_dir, FakeClient({})).counts

    assert counts["downloaded_thumbnail"] == 1
    assert (images_dir / "img1.jpg").read_bytes() == THUMB_BYTES


def test_missing_on_commons_is_counted_and_not_saved(tmp_path: Path):
    images_dir = make_dir(tmp_path)
    client = FakeClient({}, missing=("Ngoc Lu.JPG",))

    counts = run([image_row()], images_dir, client).counts

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

    counts = run(rows, images_dir, client).counts

    assert counts["downloaded_thumbnail"] == 51
    assert len(client.commons_calls) == 2


def test_original_not_wider_than_download_width_is_verified_original(tmp_path: Path):
    images_dir = make_dir(tmp_path)
    name = "Ngoc Lu.JPG"
    client = FakeClient({name: ORIGINAL_BYTES}, original_widths={name: 800})

    counts = run([image_row(width="800")], images_dir, client).counts

    assert counts["downloaded_original"] == 1
    assert client.download_calls == ["https://upload.example/Ngoc_Lu.JPG"]
    row = manifest(images_dir)["img1"]
    assert (row["kind"], row["why"]) == ("original", "original_not_wider")
    assert row["sha1"] == ORIGINAL_SHA1
    assert row["width"] == "800"


def test_original_as_wide_as_download_width_is_original(tmp_path: Path):
    images_dir = make_dir(tmp_path)
    name = "Ngoc Lu.JPG"
    client = FakeClient({name: ORIGINAL_BYTES}, original_widths={name: WIDTH})

    run([image_row(width=str(WIDTH))], images_dir, client)

    assert manifest(images_dir)["img1"]["why"] == "original_not_wider"


def test_not_wider_original_with_wrong_sha1_is_a_mismatch(tmp_path: Path):
    images_dir = make_dir(tmp_path)
    name = "Ngoc Lu.JPG"
    client = FakeClient({name: b"other bytes"}, original_widths={name: 800})

    result = run([image_row(width="800")], images_dir, client)

    assert result.counts["sha1_mismatch"] == 1
    assert result.sha1_mismatches == [
        {
            "image_id": "img1",
            "file_name": name,
            "expected_sha1": ORIGINAL_SHA1,
            "actual_sha1": hashlib.sha1(b"other bytes").hexdigest(),
        }
    ]
    assert list(images_dir.iterdir()) == []


def test_no_thumburl_downloads_the_original(tmp_path: Path):
    images_dir = make_dir(tmp_path)
    name = "Ngoc Lu.JPG"
    client = FakeClient({name: ORIGINAL_BYTES}, no_thumb=(name,))

    run([image_row()], images_dir, client)

    row = manifest(images_dir)["img1"]
    assert (row["kind"], row["why"]) == ("original", "original_no_thumburl")


def old_manifest_row(image_id, kind, sha1, width="960"):
    """A manifest row as written before the `why` column existed."""
    return {
        "image_id": image_id,
        "file_name": "Ngoc Lu.JPG",
        "kind": kind,
        "url": "https://upload.example/Ngoc_Lu.JPG?utm_content=thumbnail_unscaled",
        "width": width,
        "height": "640",
        "bytes": "20",
        "sha1": sha1,
    }


def write_old_manifest(tmp_path: Path, rows) -> None:
    header = [column for column in MANIFEST_COLUMNS if column != "why"]
    write_rows(tmp_path, "images_manifest", header, rows, MAX_FILE_BYTES)


def test_old_thumbnail_of_not_wider_original_is_relabelled_without_download(tmp_path: Path):
    images_dir = make_dir(tmp_path)
    (images_dir / "img1.jpg").write_bytes(ORIGINAL_BYTES)
    write_old_manifest(tmp_path, [old_manifest_row("img1", "thumbnail", ORIGINAL_SHA1)])
    client = FakeClient({})

    counts = run([image_row(width="800")], images_dir, client).counts

    assert counts["relabelled_as_original"] == 1
    assert counts["already_present"] == 1
    assert client.download_calls == []
    assert client.commons_calls == []
    row = manifest(images_dir)["img1"]
    assert (row["kind"], row["why"]) == ("original", "original_not_wider")
    assert row["width"] == "800"
    assert row["sha1"] == ORIGINAL_SHA1


def test_old_thumbnail_of_not_wider_original_with_wrong_sha1_is_downloaded_again(
    tmp_path: Path,
):
    images_dir = make_dir(tmp_path)
    (images_dir / "img1.jpg").write_bytes(THUMB_BYTES)
    write_old_manifest(tmp_path, [old_manifest_row("img1", "thumbnail", THUMB_SHA1)])
    client = FakeClient({"Ngoc Lu.JPG": ORIGINAL_BYTES}, original_widths={"Ngoc Lu.JPG": 800})
    messages = []

    counts = run([image_row(width="800")], images_dir, client, log=messages.append).counts

    assert counts["removed_bad_file"] == 1
    assert counts["downloaded_original"] == 1
    assert (images_dir / "img1.jpg").read_bytes() == ORIGINAL_BYTES
    assert any("removing img1.jpg" in message for message in messages)
    assert manifest(images_dir)["img1"]["why"] == "original_not_wider"


def test_old_rows_without_why_get_one(tmp_path: Path):
    images_dir = make_dir(tmp_path)
    (images_dir / "a.jpg").write_bytes(THUMB_BYTES)
    (images_dir / "b.jpg").write_bytes(ORIGINAL_BYTES)
    write_old_manifest(
        tmp_path,
        [
            old_manifest_row("a", "thumbnail", THUMB_SHA1),
            old_manifest_row("b", "original", ORIGINAL_SHA1, width="4000"),
        ],
    )
    rows = [image_row(image_id="a"), image_row(image_id="b")]

    counts = run(rows, images_dir, FakeClient({})).counts

    assert counts["already_present"] == 2
    whys = {image_id: row["why"] for image_id, row in manifest(images_dir).items()}
    assert whys == {"a": "thumbnail", "b": "original_already_present"}


def test_bad_file_on_disk_is_removed_and_logged_before_download(tmp_path: Path):
    images_dir = make_dir(tmp_path)
    (images_dir / "img1.jpg").write_bytes(b"half a file")

    class FailingClient(FakeClient):
        def download(self, url: str) -> bytes:
            # The bad file must already be gone when the new download starts.
            assert not (images_dir / "img1.jpg").exists()
            return super().download(url)

    messages = []
    counts = run([image_row()], images_dir, FailingClient({}), log=messages.append).counts

    assert counts["removed_bad_file"] == 1
    assert counts["downloaded_thumbnail"] == 1
    assert any("removing img1.jpg" in message for message in messages)


def test_missing_page_is_named_with_a_reason(tmp_path: Path):
    images_dir = make_dir(tmp_path)
    client = FakeClient({}, missing=("Ngoc Lu.JPG",))

    result = run([image_row()], images_dir, client)

    assert result.missing == [{"image_id": "img1", "file_name": "Ngoc Lu.JPG", "reason": "missing"}]


def test_manifest_is_written_after_each_file(tmp_path: Path):
    images_dir = make_dir(tmp_path)
    rows = [image_row(image_id="a", file_name="A.png"), image_row(image_id="b", file_name="B.png")]

    class InterruptedClient(FakeClient):
        def download(self, url: str) -> bytes:
            if "B.png" in url:
                raise KeyboardInterrupt
            return super().download(url)

    with pytest.raises(KeyboardInterrupt):
        run(rows, images_dir, InterruptedClient({}))

    assert set(manifest(images_dir)) == {"a"}


def test_stray_files_are_those_not_named_after_an_image_id(tmp_path: Path):
    images_dir = make_dir(tmp_path)
    for name in ["img1.jpg", "old_dropped.jpg", "img1.jpg.part", ".DS_Store"]:
        (images_dir / name).write_bytes(b"x")
    (images_dir / "subfolder").mkdir()

    strays = find_stray_files(images_dir, [image_row()])

    assert [path.name for path in strays] == [".DS_Store", "img1.jpg.part", "old_dropped.jpg"]


def test_stray_check_of_missing_folder_is_empty(tmp_path: Path):
    assert find_stray_files(tmp_path / "nothing", [image_row()]) == []


def test_download_report_names_every_failure(tmp_path: Path):
    images_dir = make_dir(tmp_path)
    client = FakeClient({}, missing=("Ngoc Lu.JPG",))
    result = run([image_row()], images_dir, client)

    write_download_report(tmp_path / "download_report.json", result)

    report = json.loads((tmp_path / "download_report.json").read_text(encoding="utf-8"))
    assert report["counts"]["missing_on_commons"] == 1
    assert report["missing"] == [
        {"image_id": "img1", "file_name": "Ngoc Lu.JPG", "reason": "missing"}
    ]
    assert report["sha1_mismatches"] == []
