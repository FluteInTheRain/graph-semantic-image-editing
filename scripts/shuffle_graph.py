"""Shuffle the contains edges of a built graph, keeping degrees (ADR-0046 section 6 ablation).

Logic: mmxai.graph.shuffle.

What it does:
    Loads a graph folder made by scripts/build_graph.py, rewires every image ->
    concept edge at random inside its own split (train, val, test) while every
    image keeps its number of edges and every concept keeps its number of edges in
    each split, and writes the result as a new graph folder. Concept names and
    vectors are copied unchanged; meta.json is copied with a "shuffle" entry (seed,
    source folder and hash). A model trained on it with the same settings tells
    whether the gain of CLIP + Graph comes from the real image-concept relations.

Before running:
    the source graph (configs/graph_vg_clip_topk.yaml, built by scripts/build_graph.py)

Run (from the repo root):
    uv run python scripts/shuffle_graph.py --config configs/graph_vg_clip_topk_shuffled.yaml

Output:
    <output_dir>/graph.npz and <output_dir>/meta.json (output_dir set in the config,
    gitignored under artifacts/). Refuses to overwrite an existing folder.
"""

import argparse
import json

from mmxai.graph.shuffle import load_shuffle_graph_config, run_shuffle_graph


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="YAML file in configs/")
    args = parser.parse_args()

    config = load_shuffle_graph_config(args.config)
    meta = run_shuffle_graph(config)
    summary = {key: value for key, value in meta.items() if key != "config"}
    print(json.dumps(summary, indent=2))
    print(f"saved {config.output_dir}")


if __name__ == "__main__":
    main()
