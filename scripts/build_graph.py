"""Build the image-concept graph ("contains" edges) of one dataset (ADR-0045).

Logic: mmxai.graph.pipeline (shared), mmxai.graph.visual_genome (dataset-specific).

What it does:
    Reads the objects of every image, keeps concepts (WordNet synsets) seen in at
    least min_train_images TRAIN images, links each image to the concepts it
    contains, encodes each concept name with the frozen CLIP text encoder, and
    writes graph.npz + meta.json. Image node vectors are not copied: they are the
    cached CLIP image embeddings, joined when the graph is turned into a PyG Data.

Before running:
    uv sync --group ml --group graph
    data/visual_genome/objects.json (see docs/guides/xay-do-thi.md, section 2)

Run (from the repo root):
    uv run python scripts/build_graph.py --config configs/graph_vg.yaml

Output:
    <output_dir>/graph.npz and <output_dir>/meta.json (output_dir set in the config,
    gitignored under artifacts/). Refuses to overwrite an existing graph.
"""

import argparse
import json

from mmxai.graph.pipeline import load_graph_config, run_build_graph


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="YAML file in configs/")
    args = parser.parse_args()

    config = load_graph_config(args.config)
    meta = run_build_graph(config)
    summary = {key: value for key, value in meta.items() if key != "config"}
    print(json.dumps(summary, indent=2))
    print(f"saved {config.output_dir}")


if __name__ == "__main__":
    main()
