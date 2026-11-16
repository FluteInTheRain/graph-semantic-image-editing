"""Download the selected MMKG images and verify each against images.csv (ADR-0050).

Logic: mmxai.data.mmkg_images.

What it does:
    Reads images.csv from the build config's output_dir, downloads the ORIGINAL
    file of every image that is not already on disk with the right sha1, and saves
    it as <output_dir>/images/<image_id><ext>. A download whose sha1 differs from
    images.csv is not saved.

Before running:
    uv sync
    Run it only after the team has reviewed the official tables
    (scripts/build_mmkg.py); this step uses the network.

Run (from the repo root):
    uv run python scripts/download_mmkg_images.py \
        --collect-config configs/mmkg_vi_collect.yaml \
        --build-config configs/mmkg_vi_build.yaml

Output:
    Prints downloaded / already_present / sha1_mismatch / missing_on_commons.
    Exits with code 1 when sha1_mismatch or missing_on_commons is above 0.
"""

import argparse
import sys
from pathlib import Path

from mmxai.data.mmkg_build import load_mmkg_build_config
from mmxai.data.mmkg_collect import load_mmkg_collect_config
from mmxai.data.mmkg_csv import read_rows
from mmxai.data.mmkg_images import download_images
from mmxai.data.wikimedia import WikimediaClient


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--collect-config", required=True, help="YAML: client settings")
    parser.add_argument("--build-config", required=True, help="YAML: where images.csv is")
    args = parser.parse_args()

    collect_config = load_mmkg_collect_config(args.collect_config)
    build_config = load_mmkg_build_config(args.build_config)
    output_dir = Path(build_config.output_dir)

    images = read_rows(output_dir, "images")
    client = WikimediaClient(
        user_agent=collect_config.user_agent,
        interval_seconds=collect_config.request_interval_seconds,
        max_attempts=collect_config.max_attempts,
    )
    counts = download_images(images, output_dir / "images", client)

    for name, count in counts.items():
        print(f"{name}: {count}")
    if counts["sha1_mismatch"] > 0 or counts["missing_on_commons"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
