from collections import Counter
from pathlib import Path
from urllib.parse import quote

import pytest

from mmxai.data.mmkg_build import (
    IMAGE_COLUMNS,
    MmkgBuildConfig,
    Review,
    pick_author,
    read_review,
    select_images,
)


def make_config() -> MmkgBuildConfig:
    return MmkgBuildConfig(
        snapshot_dir="snap",
        review_file="review.csv",
        output_dir="out",
        allowed_mime_types=["image/jpeg", "image/png"],
        allowed_license_names=["Public domain", "CC0"],
        allowed_license_prefixes=["CC BY ", "CC BY-SA "],
        attribution_license_prefixes=["CC BY ", "CC BY-SA "],
        min_short_side_px=224,
        min_images_per_item=3,
        max_images_per_item=5,
        min_items_per_concept=2,
        relations={},
        max_file_bytes=1000,
        download_width=960,
    )


def candidate(
    file: str,
    entity: str,
    origin: str = "category",
    mime: str = "image/jpeg",
    w: int | str = 800,
    h: int | str = 600,
    license: str = "CC BY-SA 4.0",
    sha1: str | None = None,
) -> dict[str, str]:
    return {
        "file_name": file,
        "entity_id": entity,
        "origin": origin,
        "mime": mime,
        "width": str(w),
        "height": str(h),
        "license": license,
        "sha1": sha1 if sha1 is not None else f"sha-{file}".ljust(12, "0"),
    }


def credits_for(*files: str, artist: str = "Alice", uploader: str = "") -> list[dict[str, str]]:
    return [{"file_name": f, "artist": artist, "uploader": uploader} for f in files]


def run(candidates, credits, review=None, config=None):
    report: Counter = Counter()
    result = select_images(
        candidates, credits, review or Review({}, {}), config or make_config(), report
    )
    return result, report


def names(result, entity="Q1"):
    return [row["file_name"] for row in result[entity]]


def filler(entity: str, count: int, prefix: str = "f") -> list[dict[str, str]]:
    return [candidate(f"{prefix}{i}.jpg", entity) for i in range(count)]


def test_file_shared_by_two_entities_is_dropped_for_both():
    rows = filler("Q1", 3) + filler("Q2", 3, "g")
    rows += [candidate("shared.jpg", "Q1"), candidate("shared.jpg", "Q2")]
    files = [r["file_name"] for r in rows]
    result, report = run(rows, credits_for(*files))
    assert "shared.jpg" not in names(result, "Q1")
    assert "shared.jpg" not in names(result, "Q2")
    assert report["images_dropped_shared_between_entities"] == 2


def test_license_rules():
    good = ["Public domain", "CC0", "CC BY 2.0", "CC BY-SA 3.0"]
    bad = ["CC BY-NC 2.0", "GFDL", ""]
    rows = [candidate(f"ok{i}.jpg", "Q1", license=lic) for i, lic in enumerate(good)]
    rows += [candidate(f"bad{i}.jpg", "Q1", license=lic) for i, lic in enumerate(bad)]
    files = [r["file_name"] for r in rows]
    result, report = run(rows, credits_for(*files))
    assert sorted(names(result)) == ["ok0.jpg", "ok1.jpg", "ok2.jpg", "ok3.jpg"]
    assert report["images_dropped_license"] == 3


def test_mime_is_checked():
    rows = filler("Q1", 3) + [candidate("x.gif", "Q1", mime="image/gif")]
    result, report = run(rows, credits_for(*[r["file_name"] for r in rows]))
    assert "x.gif" not in names(result)
    assert report["images_dropped_mime"] == 1


def test_short_side_below_224_is_dropped():
    rows = filler("Q1", 3)
    rows += [candidate("thin.jpg", "Q1", w=223, h=1000), candidate("edge.jpg", "Q1", w=224, h=224)]
    rows += [candidate("junk.jpg", "Q1", w="abc")]
    result, report = run(rows, credits_for(*[r["file_name"] for r in rows]))
    assert "thin.jpg" not in names(result)
    assert "junk.jpg" not in names(result)
    assert "edge.jpg" in names(result)
    assert report["images_dropped_too_small"] == 2


def test_author_rule():
    rows = filler("Q1", 2)
    rows += [
        candidate("a.jpg", "Q1", license="CC BY 4.0"),
        candidate("b.jpg", "Q1", license="CC BY 4.0"),
        candidate("c.jpg", "Q1", license="Public domain"),
    ]
    credits = credits_for("f0.jpg", "f1.jpg")
    credits.append({"file_name": "a.jpg", "artist": "", "uploader": "Bob"})
    credits.append({"file_name": "b.jpg", "artist": "  ", "uploader": ""})
    credits.append({"file_name": "c.jpg", "artist": "", "uploader": ""})
    result, report = run(rows, credits)
    authors = {row["file_name"]: row["author"] for row in result["Q1"]}
    assert authors["a.jpg"] == "uploader: Bob"
    assert authors["c.jpg"] == "unknown"
    assert "b.jpg" not in authors
    assert authors["f0.jpg"] == "Alice"
    assert report["images_dropped_no_author"] == 1


def test_review_drop_brings_in_next_image():
    rows = [candidate(f"v{i}.jpg", "Q1") for i in range(6)]
    review = Review({("image", "v1.jpg"): "drop"}, {})
    result, report = run(rows, credits_for(*[r["file_name"] for r in rows]), review)
    assert names(result) == ["v0.jpg", "v2.jpg", "v3.jpg", "v4.jpg", "v5.jpg"]
    assert report["images_dropped_review_drop"] == 1


def test_duplicate_sha1_keeps_smallest_name():
    rows = filler("Q1", 3)
    rows += [candidate("b.jpg", "Q1", sha1="same" * 3), candidate("a.jpg", "Q1", sha1="same" * 3)]
    result, report = run(rows, credits_for(*[r["file_name"] for r in rows]))
    chosen = names(result)
    assert "a.jpg" in chosen
    assert "b.jpg" not in chosen
    assert report["images_dropped_duplicate_sha1"] == 1


def test_p18_first_then_name_order_and_cap():
    rows = [candidate(f"{c}.jpg", "Q1") for c in "abcdef"]
    rows.append(candidate("z.jpg", "Q1", origin="p18"))
    result, report = run(rows, credits_for(*[r["file_name"] for r in rows]))
    assert names(result) == ["z.jpg", "a.jpg", "b.jpg", "c.jpg", "d.jpg"]
    assert report["images_dropped_over_max_per_item"] == 2


def test_entity_with_two_valid_images_gets_none():
    rows = filler("Q1", 2)
    result, report = run(rows, credits_for(*[r["file_name"] for r in rows]))
    assert "Q1" not in result
    assert report["images_dropped_below_min_per_item"] == 2


def write_review(path: Path, *lines: str) -> Path:
    header = "kind,key,review_status,reason,fix_name_vi,reviewer,date"
    path.write_text("\n".join([header, *lines]) + "\n", encoding="utf-8")
    return path


def test_read_review_rejects_unknown_values(tmp_path):
    bad_lines = [
        "image,a.jpg,maybe,,,k,2026-01-01",
        "picture,a.jpg,ok,,,k,2026-01-01",
        "image,a.jpg,drop,blurry,,k,2026-01-01",
        "image,a.jpg,ok,,Tên,k,2026-01-01",
    ]
    for line in bad_lines:
        path = write_review(tmp_path / "review.csv", line)
        with pytest.raises(ValueError, match="row 2"):
            read_review(path)
    assert read_review(tmp_path / "missing.csv") == Review({}, {})


def test_read_review_rejects_repeated_key(tmp_path):
    path = write_review(
        tmp_path / "review.csv",
        "image,a.jpg,ok,,,k,2026-01-01",
        "image,a.jpg,drop,other,,k,2026-01-01",
    )
    with pytest.raises(ValueError, match="a.jpg"):
        read_review(path)


def test_read_review_reads_status_and_fixes(tmp_path):
    path = write_review(
        tmp_path / "review.csv",
        "image,a.jpg,drop,duplicate,,k,2026-01-01",
        "entity,Q5,ok,,Chùa Một Cột,k,2026-01-01",
        "triple,Q1|P17|Q2,pending,,,k,2026-01-01",
    )
    review = read_review(path)
    assert review.status[("image", "a.jpg")] == "drop"
    assert review.status[("triple", "Q1|P17|Q2")] == "pending"
    assert review.fix_name_vi == {"Q5": "Chùa Một Cột"}


def test_image_id_is_stable():
    rows = [candidate(f"Chua {i}.jpg", "Q1") for i in range(3)]
    result, _ = run(rows, credits_for(*[r["file_name"] for r in rows]))
    row = result["Q1"][0]
    assert list(row) == IMAGE_COLUMNS
    assert row["image_id"] == f"Q1_{rows[0]['sha1'][:8]}"
    assert row["source_url"] == (
        "https://commons.wikimedia.org/wiki/File:" + quote(row["file_name"].replace(" ", "_"))
    )


def test_repeated_entity_file_pair_is_an_error():
    rows = filler("Q1", 3)
    rows.append(candidate("f0.jpg", "Q1", origin="p18"))
    with pytest.raises(ValueError, match="Q1.*f0.jpg"):
        run(rows, credits_for("f0.jpg", "f1.jpg", "f2.jpg"))


# Real Artist strings from the 20261116 snapshot (image_credits.csv).
GLUED_UNKNOWN = "Unknown authorUnknown author"
HEREBY = (
    "​Vietnames Wikipedia user Rungbachduong, the copyright holder of this work, "
    "hereby publishes it under the following license:"
)


@pytest.mark.parametrize(
    "artist",
    [
        GLUED_UNKNOWN,
        GLUED_UNKNOWN + ".",
        "Unknown author Unknown author .",
        "Unknown",
        "unknown",
        "Author unknown",
        "Anonymous",
        HEREBY,
    ],
)
def test_no_real_author_under_attribution_falls_back_to_uploader(artist):
    credit = {"file_name": "a.jpg", "artist": artist, "uploader": "Bob"}
    assert pick_author("CC BY-SA 3.0", credit, make_config()) == "uploader: Bob"


@pytest.mark.parametrize("artist", [GLUED_UNKNOWN, GLUED_UNKNOWN + ".", HEREBY])
def test_no_real_author_and_no_uploader_under_attribution_is_none(artist):
    credit = {"file_name": "a.jpg", "artist": artist, "uploader": ""}
    assert pick_author("CC BY 4.0", credit, make_config()) is None


@pytest.mark.parametrize("artist", [GLUED_UNKNOWN, GLUED_UNKNOWN + ".", "Unknown", "Anonymous"])
def test_no_real_author_in_public_domain_is_unknown(artist):
    credit = {"file_name": "a.jpg", "artist": artist, "uploader": "Bob"}
    assert pick_author("Public domain", credit, make_config()) == "unknown"
    assert pick_author("CC0", credit, make_config()) == "unknown"


def test_real_author_is_kept_without_trailing_punctuation():
    credit = {"file_name": "a.jpg", "artist": "Chainwit.", "uploader": "Bob"}
    assert pick_author("CC BY 4.0", credit, make_config()) == "Chainwit"


def test_unknown_photographer_is_a_real_statement_and_kept():
    # Only the listed phrases mean "no author"; anything else is the author text.
    credit = {"file_name": "a.jpg", "artist": "Unknown photographer", "uploader": "Bob"}
    assert pick_author("Public domain", credit, make_config()) == "Unknown photographer"


def test_glued_unknown_author_drops_image_without_uploader():
    rows = filler("Q1", 3) + [candidate("u.jpg", "Q1", license="CC BY-SA 3.0")]
    credits = credits_for("f0.jpg", "f1.jpg", "f2.jpg")
    credits.append({"file_name": "u.jpg", "artist": GLUED_UNKNOWN + ".", "uploader": ""})
    result, report = run(rows, credits)
    assert "u.jpg" not in names(result)
    assert report["images_dropped_no_author"] == 1


def test_read_review_accepts_excel_bom_and_strips_cells(tmp_path):
    path = tmp_path / "review.csv"
    text = (
        "kind,key,review_status,reason,fix_name_vi,reviewer,date\r\n"
        " entity , Q5 , ok ,, Chùa Một Cột ,k,2026-01-01\r\n"
    )
    # Excel saves "CSV UTF-8" with a byte order mark before the first header.
    path.write_bytes(b"\xef\xbb\xbf" + text.encode("utf-8"))
    review = read_review(path)
    assert review.status == {("entity", "Q5"): "ok"}
    assert review.fix_name_vi == {"Q5": "Chùa Một Cột"}


def test_read_review_accepts_reordered_columns(tmp_path):
    path = tmp_path / "review.csv"
    path.write_text(
        "key,kind,review_status,reason,fix_name_vi,reviewer,date\na.jpg,image,ok,,,k,\n",
        encoding="utf-8",
    )
    assert read_review(path).status == {("image", "a.jpg"): "ok"}


def test_read_review_names_missing_and_extra_columns(tmp_path):
    path = tmp_path / "review.csv"
    path.write_text("kind,key,status,reason,fix_name_vi,reviewer,date\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"review\.csv.*review_status.*status"):
        read_review(path)


def test_read_review_of_empty_file_has_no_decisions(tmp_path):
    path = tmp_path / "review.csv"
    path.write_bytes(b"")
    assert read_review(path) == Review({}, {})


def test_read_review_rejects_row_with_extra_cells(tmp_path):
    path = write_review(tmp_path / "review.csv", "image,a.jpg,ok,,,k,2026-01-01,surplus")
    with pytest.raises(ValueError, match="row 2.*more cells"):
        read_review(path)
