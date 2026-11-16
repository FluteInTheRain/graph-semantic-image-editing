"""Tests for the MMKG CSV helpers (shards stay under the size limit)."""

from pathlib import Path

import pytest

from mmxai.data.mmkg_csv import read_rows, write_rows

HEADER = ["id", "text"]


def test_small_table_is_one_file(tmp_path: Path) -> None:
    rows = [{"id": str(i), "text": f"row {i}"} for i in range(3)]

    paths = write_rows(tmp_path, "t", HEADER, rows, max_file_bytes=1_000_000)

    assert paths == [tmp_path / "t.csv"]
    assert read_rows(tmp_path, "t") == rows


def test_large_table_is_sharded_under_limit(tmp_path: Path) -> None:
    rows = [{"id": str(i), "text": "x" * 90} for i in range(200)]

    paths = write_rows(tmp_path, "t", HEADER, rows, max_file_bytes=2_000)

    assert len(paths) > 1
    assert paths[0].name == "t_000.csv"
    for path in paths:
        assert path.stat().st_size < 2_000
        assert path.read_bytes().startswith(b"id,text\r\n")
    assert read_rows(tmp_path, "t") == rows


def test_round_trip_keeps_vietnamese_quotes_and_newlines(tmp_path: Path) -> None:
    rows = [{"name_vi": 'Trống "Ngọc Lũ", đồng\nthế kỷ 3', "author": "A &amp; B"}]

    write_rows(tmp_path, "t", ["name_vi", "author"], rows, max_file_bytes=1_000_000)

    assert read_rows(tmp_path, "t") == rows


def test_empty_rows_write_header_only(tmp_path: Path) -> None:
    paths = write_rows(tmp_path, "t", HEADER, [], max_file_bytes=1_000)

    assert paths == [tmp_path / "t.csv"]
    assert read_rows(tmp_path, "t") == []


def test_row_larger_than_limit_is_rejected(tmp_path: Path) -> None:
    rows = [{"id": "1", "text": "x" * 500}]

    with pytest.raises(ValueError):
        write_rows(tmp_path, "t", HEADER, rows, max_file_bytes=100)


def test_read_rejects_both_plain_and_sharded(tmp_path: Path) -> None:
    (tmp_path / "t.csv").write_text("id,text\r\n", encoding="utf-8")
    (tmp_path / "t_000.csv").write_text("id,text\r\n", encoding="utf-8")

    with pytest.raises(ValueError):
        read_rows(tmp_path, "t")


def test_read_missing_table_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_rows(tmp_path, "t")
