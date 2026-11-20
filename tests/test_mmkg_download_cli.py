"""Tests for scripts/download_mmkg_images.py: exit codes and stray files (no network)."""

import importlib.util
import json
from pathlib import Path

from mmxai.data.mmkg_build import IMAGE_COLUMNS
from mmxai.data.mmkg_csv import write_rows
from test_mmkg_images import ORIGINAL_BYTES, FakeClient, image_row

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script(fake_client: FakeClient):
    """Import the script (scripts/ is not a package) with a fake client in place."""
    path = REPO_ROOT / "scripts" / "download_mmkg_images.py"
    spec = importlib.util.spec_from_file_location("download_mmkg_images_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.WikimediaClient = lambda **kwargs: fake_client
    return module


def setup(tmp_path: Path, images: list[dict[str, str]]) -> list[str]:
    """Write images.csv and both configs; return the script arguments."""
    out = tmp_path / "out"
    rows = []
    for image in images:
        row = dict.fromkeys(IMAGE_COLUMNS, "")
        row.update(image)
        rows.append(row)
    write_rows(out, "images", IMAGE_COLUMNS, rows, 100000)
    collect = tmp_path / "collect.yaml"
    collect.write_text(
        "output_root: raw\n"
        "sparql_url: https://example.org/api/wikidata\n"
        "country_qid: Q881\n"
        "designation_property: P1435\n"
        "user_agent: test-agent\n"
        "request_interval_seconds: 0\n"
        "max_attempts: 1\n"
        "max_retry_wait_seconds: 1\n"
        "max_file_bytes: 1048576\n",
        encoding="utf-8",
    )
    build = tmp_path / "build.yaml"
    build.write_text(
        "snapshot_dir: raw/20261116\n"
        "review_file: review.csv\n"
        f"output_dir: {out}\n"
        "allowed_mime_types: [image/jpeg]\n"
        "allowed_license_names: [CC0]\n"
        "allowed_license_prefixes: []\n"
        "attribution_license_prefixes: []\n"
        "min_short_side_px: 224\n"
        "min_images_per_item: 3\n"
        "max_images_per_item: 5\n"
        "min_items_per_concept: 3\n"
        "max_file_bytes: 100000\n"
        "download_width: 960\n"
        "relations: {P131: located_in}\n",
        encoding="utf-8",
    )
    return ["--collect-config", str(collect), "--build-config", str(build)]


def test_all_downloaded_exits_zero_and_writes_report(tmp_path):
    argv = setup(tmp_path, [image_row()])

    code = load_script(FakeClient({})).main(argv)

    assert code == 0
    report = json.loads((tmp_path / "out" / "download_report.json").read_text(encoding="utf-8"))
    assert report["counts"]["downloaded_thumbnail"] == 1


def test_sha1_mismatch_exits_one_and_is_named(tmp_path, capsys):
    argv = setup(tmp_path, [image_row(width="800")])
    client = FakeClient({"Ngoc Lu.JPG": b"other"}, original_widths={"Ngoc Lu.JPG": 800})

    code = load_script(client).main(argv)

    assert code == 1
    err = capsys.readouterr().err
    assert "sha1 mismatch" in err
    assert "img1" in err


def test_missing_on_commons_exits_one_and_is_named(tmp_path, capsys):
    argv = setup(tmp_path, [image_row()])

    code = load_script(FakeClient({}, missing=("Ngoc Lu.JPG",))).main(argv)

    assert code == 1
    assert "img1" in capsys.readouterr().err


def test_stray_file_exits_one_before_any_request(tmp_path, capsys):
    argv = setup(tmp_path, [image_row()])
    images_dir = tmp_path / "out" / "images"
    images_dir.mkdir()
    (images_dir / "Q9_dropped.jpg").write_bytes(ORIGINAL_BYTES)
    client = FakeClient({})

    code = load_script(client).main(argv)

    assert code == 1
    assert "Q9_dropped.jpg" in capsys.readouterr().err
    assert (images_dir / "Q9_dropped.jpg").exists()
    assert client.commons_calls == []


def test_delete_stray_removes_and_logs_then_downloads(tmp_path, capsys):
    argv = setup(tmp_path, [image_row()])
    images_dir = tmp_path / "out" / "images"
    images_dir.mkdir()
    (images_dir / "Q9_dropped.jpg").write_bytes(ORIGINAL_BYTES)

    code = load_script(FakeClient({})).main(argv + ["--delete-stray"])

    assert code == 0
    assert not (images_dir / "Q9_dropped.jpg").exists()
    assert "deleted stray file" in capsys.readouterr().out
    assert (images_dir / "img1.jpg").exists()
