"""Make the fixed train/val/test split file (ADR-0042). Logic lives in mmxai.data.splits.

What it does:
    Groups near-duplicate images, then splits Visual Genome images into
    5000 test, 5000 val and the rest train, and writes splits/visual_genome/split_seed0.json.

You normally do NOT run this: the split is already committed and frozen. The script
refuses to overwrite it. Run it only to check that it is reproducible (write to
another folder by copying the config and changing output_dir) or for a new dataset.

Before running:
    uv sync --group ml
    Visual Genome features in artifacts/features/visual_genome/ViT-B-32__laion2b_s34b_b79k/
    (image_features.npz, text_features.npz, meta.json); see
    docs/guides/chuan-bi-embedding-clip.md section 2, or:
        kaggle kernels output qphulong/prepare-clip-embeddings-for-mmkg-xai \
            -p artifacts/kaggle_download
        mkdir -p artifacts/features
        mv artifacts/kaggle_download/features/visual_genome artifacts/features/
    (`kaggle kernels pull` downloads only the notebook code, not the data.)

Run (from the repo root, about 1 minute on a laptop CPU):
    uv run python scripts/make_splits.py --config configs/splits_visual_genome.yaml

Output:
    Prints the group sizes and the set sizes; writes the split JSON.
"""

import argparse
from collections import Counter

from mmxai.data.splits import (
    SPLIT_NAMES,
    group_near_duplicate_images,
    load_split_config,
    make_split,
    save_split,
    split_path,
)
from mmxai.features.cache import load_image_features


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="YAML file in configs/")
    args = parser.parse_args()

    config = load_split_config(args.config)
    embeddings, image_ids = load_image_features(config.features_dir)
    groups = group_near_duplicate_images(embeddings, image_ids, config.duplicate_threshold)

    # Printed so the person running it can record it in the data card.
    group_sizes = Counter(len(group) for group in groups)
    print(f"{len(image_ids)} images, groups by size: {dict(sorted(group_sizes.items()))}")

    split = make_split(groups, config.dataset, config.num_val, config.num_test, config.split_seed)
    path = split_path(config.output_dir, config.split_seed)
    save_split(split, path)
    sizes = ", ".join(f"{name} {len(split[name])}" for name in SPLIT_NAMES)
    print(f"saved {path}: {sizes}")


if __name__ == "__main__":
    main()
