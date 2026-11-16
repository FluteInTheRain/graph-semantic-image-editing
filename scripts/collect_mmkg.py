"""Collect a raw Vietnamese heritage snapshot from Wikidata and Commons (ADR-0050).

What it does:
    Asks Wikidata for heritage items in Vietnam, their statements and their Commons
    images, and saves the answers as they are (no filtering) in a dated folder
    mmkg/raw/<YYYYMMDD>/. A later step builds the official tables from it.

Before running:
    uv sync
    Internet access. The script waits between requests, so it takes a while.
    It refuses to run twice on the same day (the snapshot is never overwritten).

Run (from the repo root):
    uv run python scripts/collect_mmkg.py --config configs/mmkg_vi_collect.yaml

Output:
    Prints the snapshot folder, the row count of each table and the number of
    missing (deleted) files. Writes entities_raw, triples_raw, image_candidates,
    image_credits (CSV) and meta.json.
"""

import argparse
from datetime import date
from pathlib import Path

from mmxai.data.mmkg_collect import collect_snapshot, load_mmkg_collect_config, write_snapshot
from mmxai.data.wikimedia import WikimediaClient


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="YAML file in configs/")
    args = parser.parse_args()

    config = load_mmkg_collect_config(args.config)
    client = WikimediaClient(
        config.user_agent, config.request_interval_seconds, config.max_attempts
    )
    snapshot = collect_snapshot(client, config, date.today())
    folder = write_snapshot(snapshot, Path(config.output_root), config.max_file_bytes)

    print(f"saved {folder}")
    for table, count in snapshot.meta["row_counts"].items():
        print(f"  {table}: {count} rows")
    print(f"missing files: {len(snapshot.meta['missing_files'])}")


if __name__ == "__main__":
    main()
