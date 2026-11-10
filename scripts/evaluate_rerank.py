"""Re-rank CLIP's top 100 with the trained GNN and score it (ADR-0046).

Logic: mmxai.rerank.evaluate.

What it does:
    val:  re-ranks with every alpha of alpha_grid, reports all, and picks the best (MRR).
    test: re-ranks with the alpha written in the config (the one chosen on val).
    final = alpha * cos(q, image) + (1 - alpha) * cos(q, h_image).

Before running:
    scripts/train.py for the same config and seed.

Run (from the repo root):
    Val, as often as you like:
        uv run python scripts/evaluate_rerank.py --config configs/gnn_vg.yaml --seed 0 --split val
    Then write the chosen alpha into the config, commit, and run test ONCE per seed (ADR-0009):
        for s in 0 1 2; do
            uv run python scripts/evaluate_rerank.py --config configs/gnn_vg.yaml \
                --seed $s --split test
        done

Output:
    results/runs/<YYYYMMDD-HHMM>_<experiment>_seed<k>/ with config.yaml, metrics.json,
    env.json and scores.npz (per query: text id, candidate image ids, cosine, graph score).
"""

import argparse
import json

from mmxai.models.training import load_gnn_config
from mmxai.rerank.evaluate import run_rerank_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="YAML file in configs/")
    parser.add_argument("--seed", type=int, required=True, help="seed of the trained model")
    parser.add_argument("--split", choices=["val", "test"], required=True)
    args = parser.parse_args()

    run_dir = run_rerank_evaluation(load_gnn_config(args.config), args.seed, args.split)
    with open(run_dir / "metrics.json", encoding="utf-8") as file:
        print(json.dumps(json.load(file), indent=2))
    print(f"saved {run_dir}")


if __name__ == "__main__":
    main()
