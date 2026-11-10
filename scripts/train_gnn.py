"""Train the one-layer GNN that re-ranks CLIP's top 100 (ADR-0046). Logic: mmxai.models.training.

What it does:
    Trains W_self, W_nb, b on TRAIN images only: for each region description, its own
    image against 15 wrong images from CLIP's top 100 (listwise InfoNCE). Keeps the
    epoch with the best val MRR@100 of the graph score alone (early stopping).

Before running:
    uv sync --group ml --group graph
    CLIP features in features_dir (docs/guides/chuan-bi-embedding-clip.md, section 2)
    the graph in graph_dir (scripts/build_graph.py --config configs/graph_vg_clip_topk.yaml)

Run (from the repo root), once per seed (ADR-0011):
    for s in 0 1 2; do
        uv run python scripts/train_gnn.py --config configs/gnn_vg.yaml --seed $s
    done

Output:
    <checkpoint_dir>/<experiment>_seed<k>/model.pt and meta.json (val history, best
    epoch, git state). Refuses to overwrite an existing checkpoint.
"""

import argparse

from mmxai.models.training import load_gnn_config, train_gnn


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="YAML file in configs/")
    parser.add_argument("--seed", type=int, required=True, help="training seed")
    args = parser.parse_args()

    output_dir = train_gnn(load_gnn_config(args.config), args.seed)
    print(f"saved {output_dir}")


if __name__ == "__main__":
    main()
