"""Build the plain-CLIP image vector stores. All logic lives in mmxai.index.vector_store."""

import argparse

from mmxai.index.vector_store import build_vector_stores, load_vector_store_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML file in configs/")
    args = parser.parse_args()

    config = load_vector_store_config(args.config)
    for folder in build_vector_stores(config):
        print(f"saved {folder}")


if __name__ == "__main__":
    main()
