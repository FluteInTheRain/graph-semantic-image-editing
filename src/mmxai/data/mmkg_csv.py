"""Write and read the MMKG CSV tables, splitting big ones into shards.

The repo refuses any committed file of 1 MiB or more, so a table that would be too
big is written as `<stem>_000.csv`, `<stem>_001.csv`, ... and each shard repeats
the header. A small table stays one file `<stem>.csv`.
"""

import csv
import io
from pathlib import Path


def _format_line(values: list[str]) -> str:
    """Format one CSV line (with its line ending) exactly as it will be written."""
    buffer = io.StringIO()
    csv.writer(buffer).writerow(values)
    return buffer.getvalue()


def _size_in_bytes(text: str) -> int:
    return len(text.encode("utf-8"))


def write_rows(
    directory: Path,
    stem: str,
    header: list[str],
    rows: list[dict[str, str]],
    max_file_bytes: int,
) -> list[Path]:
    """Write rows to `<stem>.csv`, or to numbered shards if one file would be too big.

    Every written file is strictly smaller than `max_file_bytes`. Returns the paths
    written. A single row that cannot fit in a file by itself raises ValueError.
    """
    header_line = _format_line(header)
    header_size = _size_in_bytes(header_line)

    # Each shard is a list of formatted row lines (without the header).
    shards: list[list[str]] = [[]]
    current_size = header_size
    for row in rows:
        line = _format_line([row[column] for column in header])
        line_size = _size_in_bytes(line)
        if header_size + line_size >= max_file_bytes:
            raise ValueError(
                f"One row of {stem} needs {header_size + line_size} bytes "
                f"with the header, which is not under the limit of {max_file_bytes}."
            )
        # Start a new shard before the line that would reach the limit.
        if current_size + line_size >= max_file_bytes:
            shards.append([])
            current_size = header_size
        shards[-1].append(line)
        current_size += line_size

    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, lines in enumerate(shards):
        if len(shards) == 1:
            path = directory / f"{stem}.csv"
        else:
            path = directory / f"{stem}_{index:03d}.csv"
        # newline="" keeps the "\r\n" line endings that csv produced, so the size
        # we measured is the size on disk.
        with path.open("w", encoding="utf-8", newline="") as file:
            file.write(header_line)
            file.writelines(lines)
        paths.append(path)
    return paths


def _read_one_file(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def read_rows(directory: Path, stem: str) -> list[dict[str, str]]:
    """Read `<stem>.csv`, or all `<stem>_NNN.csv` shards in order.

    Raises FileNotFoundError if neither form exists, and ValueError if both do
    (it would be unclear which one is current).
    """
    plain_path = directory / f"{stem}.csv"
    # Three zero-padded digits sort in numeric order as plain strings.
    shard_paths = sorted(directory.glob(f"{stem}_[0-9][0-9][0-9].csv"))

    if plain_path.exists() and shard_paths:
        raise ValueError(f"Both {plain_path.name} and shards of {stem} exist in {directory}.")
    if plain_path.exists():
        return _read_one_file(plain_path)
    if not shard_paths:
        raise FileNotFoundError(f"No {stem}.csv or {stem}_NNN.csv in {directory}.")

    rows: list[dict[str, str]] = []
    for path in shard_paths:
        rows.extend(_read_one_file(path))
    return rows
