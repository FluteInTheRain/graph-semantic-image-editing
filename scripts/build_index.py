"""Build the plain-CLIP image vector stores. Logic lives in mmxai.index.vector_store.

What it does:
    Builds one exact FAISS index per split (val and test by default) from the CLIP
    image embeddings, and saves it with its row-to-image-id map and a meta.json
    that names the split hash and the feature code hash.

Before running:
    uv sync --group ml
    Visual Genome features in artifacts/features/visual_genome/ViT-B-32__laion2b_s34b_b79k/
    (how to download: see the docstring of scripts/make_splits.py).

Run (from the repo root, a few seconds):
    uv run python scripts/build_index.py --config configs/vector_store_vg.yaml
    The script refuses to overwrite an existing store; delete the folder to rebuild.

Output:
    artifacts/vector_store/visual_genome/ViT-B-32__laion2b_s34b_b79k/<split>/
        images.faiss, image_ids.npy, meta.json   (about 10 MB per split, gitignored)

Use it in code:
    from mmxai.index.image_index import load_image_index, search_images
    folder = "artifacts/vector_store/visual_genome/ViT-B-32__laion2b_s34b_b79k/test"
    store = load_image_index(folder)
    scores, image_ids = search_images(store, query_embeddings, top_k=10)
    # query_embeddings: shape (n, 512), L2-normalised CLIP text embeddings
"""

import argparse

from mmxai.index.vector_store import build_vector_stores, load_vector_store_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="YAML file in configs/")
    args = parser.parse_args()

    config = load_vector_store_config(args.config)
    for folder in build_vector_stores(config):
        print(f"saved {folder}")


if __name__ == "__main__":
    main()
