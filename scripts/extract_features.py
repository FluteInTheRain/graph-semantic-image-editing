"""Extract CLIP embeddings for one dataset. All logic lives in mmxai.features.pipeline."""

import argparse

from mmxai.features.pipeline import run_extraction, with_path_overrides
from mmxai.utils.config import load_feature_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML file in configs/")
    parser.add_argument("--data-dir", help="override data_dir")
    parser.add_argument("--images-dir", help="override images_dir")
    parser.add_argument("--output-root", help="override output_root")
    parser.add_argument("--limit-images", type=int, help="trial run: encode only this many images")
    parser.add_argument("--limit-texts", type=int, help="trial run: encode only this many texts")
    args = parser.parse_args()

    config = load_feature_config(args.config)
    config = with_path_overrides(config, args.data_dir, args.images_dir, args.output_root)
    run_extraction(config, args.limit_images, args.limit_texts)


if __name__ == "__main__":
    main()
