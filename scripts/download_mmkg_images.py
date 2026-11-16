"""Download the selected MMKG images and verify each against images.csv (ADR-0050).

Logic: mmxai.data.mmkg_images.

What it does:
    Reads images.csv from the build config's output_dir and downloads every image
    that is not already on disk, as a thumbnail of download_width px (960) from
    Wikimedia's pregenerated sizes. The original is downloaded instead, and checked
    against the Commons sha1 in images.csv, when it is no wider than download_width,
    when Commons has no thumbnail, or when the thumbnail's short side would be under
    224 px. Saves <output_dir>/images/<image_id><ext> and writes
    <output_dir>/images_manifest.csv (sha1 of every file on disk, column `why`).
    A file on disk with a wrong sha1 is removed and downloaded again.

Before running:
    uv sync
    Run it only after the team has reviewed the official tables
    (scripts/build_mmkg.py); this step uses the network.

Run (from the repo root):
    uv run python scripts/download_mmkg_images.py \
        --collect-config configs/mmkg_vi_collect.yaml \
        --build-config configs/mmkg_vi_build.yaml
    Add --delete-stray to delete files of images.csv's folder that are no longer
    in images.csv (for example an image the review dropped).

Output:
    Stops with code 1 before any request when the images folder holds stray files
    (and --delete-stray is not given); lists them on stderr.
    Prints the counts, a progress line every 25 files and the total bytes of the
    manifest; writes <output_dir>/download_report.json with every failure named.
    Exits with code 1 when any sha1 mismatch or missing file remains (listed on stderr).
"""

import argparse
import sys
from pathlib import Path

from mmxai.data.mmkg_build import load_mmkg_build_config
from mmxai.data.mmkg_collect import load_mmkg_collect_config
from mmxai.data.mmkg_csv import read_rows
from mmxai.data.mmkg_images import download_images, find_stray_files, write_download_report
from mmxai.data.wikimedia import WikimediaClient


def main(argv: list[str] | None = None) -> int:
    """Run the download; return the exit code (argv=None reads the command line)."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--collect-config", required=True, help="YAML: client settings")
    parser.add_argument("--build-config", required=True, help="YAML: where images.csv is")
    parser.add_argument(
        "--delete-stray",
        action="store_true",
        help="delete files in the images folder that are not in images.csv",
    )
    args = parser.parse_args(argv)

    collect_config = load_mmkg_collect_config(args.collect_config)
    build_config = load_mmkg_build_config(args.build_config)
    output_dir = Path(build_config.output_dir)
    images_dir = output_dir / "images"
    images = read_rows(output_dir, "images")

    strays = find_stray_files(images_dir, images)
    if strays and not args.delete_stray:
        print(f"{len(strays)} stray files (not in images.csv):", file=sys.stderr)
        for path in strays:
            print(f"  {path}", file=sys.stderr)
        print("rerun with --delete-stray to delete them", file=sys.stderr)
        return 1
    for path in strays:
        path.unlink()
        print(f"deleted stray file {path}")

    client = WikimediaClient(
        user_agent=collect_config.user_agent,
        interval_seconds=collect_config.request_interval_seconds,
        max_attempts=collect_config.max_attempts,
        max_retry_wait_seconds=collect_config.max_retry_wait_seconds,
    )
    result = download_images(
        images,
        images_dir,
        client,
        output_dir,
        build_config.download_width,
        build_config.max_file_bytes,
        log=print,
    )

    for name, count in result.counts.items():
        print(f"{name}: {count}")
    manifest = read_rows(output_dir, "images_manifest")
    print(f"total_bytes: {sum(int(row['bytes']) for row in manifest)}")
    report_path = output_dir / "download_report.json"
    write_download_report(report_path, result)
    print(f"report: {report_path}")

    for failure in result.sha1_mismatches:
        print(
            f"sha1 mismatch: {failure['image_id']} {failure['file_name']!r} "
            f"expected {failure['expected_sha1']}, got {failure['actual_sha1']}",
            file=sys.stderr,
        )
    for failure in result.missing:
        print(
            f"missing on Commons ({failure['reason']}): {failure['image_id']} "
            f"{failure['file_name']!r}",
            file=sys.stderr,
        )
    if result.sha1_mismatches or result.missing:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
