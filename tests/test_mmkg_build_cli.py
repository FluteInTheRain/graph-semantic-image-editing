"""Tests for scripts/build_mmkg.py: exit codes and messages (no network)."""

import importlib.util
from pathlib import Path

from test_mmkg_build import REVIEW_HEADER, three_items_snapshot

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script():
    """Import scripts/build_mmkg.py as a module (scripts/ is not a package)."""
    path = REPO_ROOT / "scripts" / "build_mmkg.py"
    spec = importlib.util.spec_from_file_location("build_mmkg_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_config(tmp_path: Path) -> Path:
    config_file = tmp_path / "build.yaml"
    config_file.write_text(
        f"snapshot_dir: {tmp_path / 'snap'}\n"
        f"review_file: {tmp_path / 'review.csv'}\n"
        f"output_dir: {tmp_path / 'out'}\n"
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
    return config_file


def mark_all_ok(review_file: Path) -> None:
    text = review_file.read_text(encoding="utf-8").replace(",pending,", ",ok,")
    review_file.write_text(text, encoding="utf-8")


def test_require_reviewed_fails_while_rows_are_pending(tmp_path, capsys):
    three_items_snapshot(tmp_path / "snap")
    config = write_config(tmp_path)

    code = load_script().main(["--config", str(config), "--require-reviewed"])

    assert code == 1
    assert "still pending" in capsys.readouterr().out


def test_require_reviewed_passes_when_everything_is_reviewed(tmp_path):
    three_items_snapshot(tmp_path / "snap")
    config = write_config(tmp_path)
    script = load_script()
    script.main(["--config", str(config)])
    mark_all_ok(tmp_path / "review.csv")

    assert script.main(["--config", str(config), "--require-reviewed"]) == 0


def test_require_reviewed_fails_on_unmatched_review_key(tmp_path, capsys):
    three_items_snapshot(tmp_path / "snap")
    config = write_config(tmp_path)
    script = load_script()
    script.main(["--config", str(config)])
    mark_all_ok(tmp_path / "review.csv")
    with open(tmp_path / "review.csv", "a", encoding="utf-8", newline="") as file:
        file.write("image,Typo.jpg,drop,other,,k,2026-11-16\r\n")
    capsys.readouterr()

    code = script.main(["--config", str(config), "--require-reviewed"])

    assert code == 1
    assert "image:Typo.jpg" in capsys.readouterr().out


def test_without_require_reviewed_pending_rows_exit_zero(tmp_path):
    three_items_snapshot(tmp_path / "snap")
    config = write_config(tmp_path)

    assert load_script().main(["--config", str(config)]) == 0


def test_missing_review_file_is_announced(tmp_path, capsys):
    three_items_snapshot(tmp_path / "snap")
    config = write_config(tmp_path)

    load_script().main(["--config", str(config)])

    out = capsys.readouterr().out
    assert f"a new one will be created at {tmp_path / 'review.csv'}" in out
    header = (tmp_path / "review.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header == ",".join(REVIEW_HEADER)


def test_prints_nonzero_drop_counters_report_path_and_name_warning(tmp_path, capsys):
    three_items_snapshot(tmp_path / "snap")
    # Q1 loses its Vietnamese label and one triple uses an unknown property.
    entities_file = tmp_path / "snap" / "entities_raw.csv"
    text = entities_file.read_text(encoding="utf-8").replace("vi Q1", "")
    entities_file.write_text(text, encoding="utf-8")
    with open(tmp_path / "snap" / "triples_raw.csv", "a", encoding="utf-8", newline="") as file:
        file.write("Q1,P999,Q9\r\n")
    config = write_config(tmp_path)

    load_script().main(["--config", str(config)])

    out = capsys.readouterr().out
    assert "triples_dropped_unknown_relation: 1" in out
    assert "images_dropped_mime" not in out
    assert str(tmp_path / "out" / "build_report.json") in out
    assert "warning: 1 entities have no name_vi" in out
