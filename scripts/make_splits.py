"""Make the fixed train/val/test split file. All logic lives in mmxai.data.splits."""

import argparse
from collections import Counter

from mmxai.data.splits import (
    SPLIT_NAMES,
    group_duplicate_images,
    load_split_config,
    make_split,
    save_split,
    split_path,
)
from mmxai.features.cache import load_image_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML file in configs/")
    args = parser.parse_args()

    config = load_split_config(args.config)
    embeddings, image_ids = load_image_features(config.features_dir)

    # Printed so the person running it can compare with the data card.
    group_sizes = Counter(len(group) for group in group_duplicate_images(embeddings, image_ids))
    print(f"{len(image_ids)} images, duplicate groups by size: {dict(sorted(group_sizes.items()))}")

    split = make_split(
        embeddings, image_ids, config.dataset, config.num_val, config.num_test, config.split_seed
    )
    path = split_path(config.output_dir, config.split_seed)
    save_split(split, path)
    sizes = ", ".join(f"{name} {len(split[name])}" for name in SPLIT_NAMES)
    print(f"saved {path}: {sizes}")


if __name__ == "__main__":
    main()
