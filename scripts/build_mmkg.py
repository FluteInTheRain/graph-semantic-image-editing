"""Build the official MMKG files from a raw snapshot (ADR-0050). Logic: mmxai.data.mmkg_build.

What it does:
    Reads the raw snapshot, applies the image/entity/triple rules and your review
    decisions, writes mmkg/entities*.csv, triples*.csv, images*.csv and
    build_report.json, and APPENDS a `pending` row to mmkg/review.csv for every
    official image, triple and entity that has no review row yet.

Before running:
    uv sync
    A raw snapshot from scripts/collect_mmkg.py, and snapshot_dir in the config
    set to its folder (the placeholder YYYYMMDD is refused).

Run (from the repo root):
    uv run python scripts/build_mmkg.py --config configs/mmkg_vi_build.yaml
    uv run python scripts/build_mmkg.py --config configs/mmkg_vi_build.yaml --require-reviewed

Output:
    Prints the main counts. With --require-reviewed, exits with code 1 and prints
    the pending counts when any official row is still `pending`.
"""

import argparse
import sys
from pathlib import Path

from mmxai.data.mmkg_build import (
    build_mmkg,
    load_mmkg_build_config,
    read_review,
    write_built_mmkg,
)

MAIN_COUNTS = ["entities_item", "entities_concept", "triples", "images"]
PENDING_COUNTS = ["pending_image", "pending_triple", "pending_entity"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="YAML file in configs/")
    parser.add_argument(
        "--require-reviewed",
        action="store_true",
        help="exit with code 1 when any official row is still pending",
    )
    args = parser.parse_args()

    config = load_mmkg_build_config(args.config)
    review = read_review(Path(config.review_file))
    built = build_mmkg(Path(config.snapshot_dir), review, config)
    write_built_mmkg(
        built, Path(config.output_dir), Path(config.review_file), config.max_file_bytes
    )

    for name in MAIN_COUNTS:
        print(f"{name}: {built.report[name]}")
    pending = {name: built.report[name] for name in PENDING_COUNTS}
    if args.require_reviewed and any(count > 0 for count in pending.values()):
        print(f"still pending: {pending}")
        sys.exit(1)


if __name__ == "__main__":
    main()
