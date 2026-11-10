"""Train the GNN and choose alpha on val, for every seed (ADR-0046). Slow; run once.

What it does (each step is skipped when its output already exists, so re-running is safe):
    1. train the GNN for each seed (train images only, early stopping on val)
    2. re-rank val for each seed and choose alpha, stored next to its checkpoint
Needs the graph built by scripts/setup.py.

Run (from the repo root; about 9 minutes per epoch on a 14-core CPU, faster on a GPU):
    uv run --group ml --group graph python scripts/train.py
    uv run --group ml --group graph python scripts/train.py --seeds 0     # one seed
With training curves on Weights & Biases (ADR-0047; `uv run --group tracking wandb login` once):
    uv run --group ml --group graph --group tracking python scripts/train.py

Then:
    uv run --group ml --group graph python scripts/run_pipeline.py --mode clip_graph
"""

import argparse
import json
from pathlib import Path

from mmxai.models.training import (
    ALPHA_FILE,
    CHECKPOINT_FILE,
    checkpoint_folder,
    load_gnn_config,
    train_gnn,
)
from mmxai.rerank.evaluate import run_rerank_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/gnn_vg.yaml", help="GNN config")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    config = load_gnn_config(args.config)
    if not (Path(config.graph_dir) / "graph.npz").exists():
        raise SystemExit(f"No graph in {config.graph_dir}; run scripts/setup.py first.")

    for seed in args.seeds:
        folder = checkpoint_folder(config, seed)
        print(f"== 1. Train seed {seed}")
        if (folder / CHECKPOINT_FILE).exists():
            print(f"already trained: {folder}")
        else:
            train_gnn(config, seed)

        print(f"== 2. Choose alpha on val, seed {seed}")
        if (folder / ALPHA_FILE).exists():
            print(f"already chosen: {json.loads((folder / ALPHA_FILE).read_text())}")
        else:
            run_dir = run_rerank_evaluation(config, seed, "val")
            print(f"val results: {run_dir}")

    print("Done. Try:")
    print("  uv run --group ml --group graph python scripts/run_pipeline.py --mode clip_graph")


if __name__ == "__main__":
    main()
