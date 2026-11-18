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
    Prints the main counts, every drop counter above 0, the path of
    build_report.json, a warning when entities lack a Vietnamese name, and the
    review keys that match nothing in the raw snapshot (usually typos).
    With --require-reviewed, exits with code 1 when any official row is still
    `pending` or any review key matches nothing.
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


def main(argv: list[str] | None = None) -> int:
    """Run the build; return the exit code (argv=None reads the command line)."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="YAML file in configs/")
    parser.add_argument(
        "--require-reviewed",
        action="store_true",
        help="exit with code 1 when any official row is pending or a review key matches nothing",
    )
    args = parser.parse_args(argv)

    config = load_mmkg_build_config(args.config)
    review_file = Path(config.review_file)
    if not review_file.exists():
        print(f"no review file yet; a new one will be created at {review_file}")
    review = read_review(review_file)
    built = build_mmkg(Path(config.snapshot_dir), review, config)
    output_dir = Path(config.output_dir)
    write_built_mmkg(built, output_dir, review_file, config.max_file_bytes)

    report = built.report
    for name in MAIN_COUNTS:
        print(f"{name}: {report[name]}")
    for name in sorted(report):
        if "_dropped_" in name and report[name] > 0:
            print(f"{name}: {report[name]}")
    print(f"report: {output_dir / 'build_report.json'}")
    if report["entities_missing_name_vi"] > 0:
        print(
            f"warning: {report['entities_missing_name_vi']} entities have no name_vi; "
            "add fix_name_vi in the review file"
        )
    unmatched = report["review_keys_unmatched"]
    if unmatched:
        print(f"review keys matching nothing in the snapshot ({len(unmatched)}): {unmatched}")

    pending = {name: report[name] for name in PENDING_COUNTS}
    if args.require_reviewed:
        if any(count > 0 for count in pending.values()):
            print(f"still pending: {pending}")
            return 1
        if unmatched:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
