"""Build a CLIP top-k contains graph reusing another graph's concepts (ADR-0050).

Logic: mmxai.graph.vocabulary_transfer.

What it does:
    Copies the concept nodes (keys, names, CLIP text vectors) of an existing graph,
    links every image of the split to its k nearest concepts by CLIP cosine, and
    writes graph.npz + meta.json. Used for Flickr30k, which has no object annotations.

Before running:
    uv sync --group ml --group graph
    the source graph (configs/graph_vg_clip_topk.yaml), the split file and the cached
    CLIP image embeddings of the new dataset

Run (from the repo root):
    uv run python scripts/build_graph_from_vocabulary.py \
        --config configs/graph_flickr30k_clip_topk.yaml

Output:
    <output_dir>/graph.npz and <output_dir>/meta.json. Refuses to overwrite.
"""

import argparse
import json

from mmxai.graph.vocabulary_transfer import (
    load_vocabulary_transfer_config,
    run_vocabulary_transfer,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="YAML file in configs/")
    args = parser.parse_args()

    config = load_vocabulary_transfer_config(args.config)
    meta = run_vocabulary_transfer(config)
    summary = {key: value for key, value in meta.items() if key != "config"}
    print(json.dumps(summary, indent=2))
    print(f"saved {config.output_dir}")


if __name__ == "__main__":
    main()
