"""Train CLIP + Graph and choose epoch and alpha on VAL (ADR-0046, ADR-0047).

Logic: mmxai.eval.graph_rerank.train_and_select. Never touches test.

What it does:
    1. Builds the train subgraph and finds 100 CLIP candidates (hard negatives) for
       every train text, among train images only (about 10 min, once for all seeds).
    2. Per seed: trains the one-layer GraphSAGE, early-stops on val, chooses alpha on
       val, saves the weights and the choice, and writes a val run folder.

Before running:
    uv sync --group ml --group graph
    Visual Genome features in artifacts/features/visual_genome/ViT-B-32__laion2b_s34b_b79k/
    The graph: uv run python scripts/build_graph.py --config configs/graph_vg_clip_topk.yaml

Run (from the repo root; about 30-60 min per seed on a 4-thread laptop CPU):
    uv run python scripts/train_graph_rerank.py --config configs/clip_graph_vg.yaml --seeds 0 1 2
    uv run python scripts/train_graph_rerank.py --config configs/clip_nograph_vg.yaml --seeds 0 1 2

Output:
    artifacts/models/visual_genome/<experiment>/seed<k>/model.pt and selection.json
    results/runs/<YYYYMMDD-HHMM>_<experiment>_seed<k>/ (split: val)
    Then, once, on committed code: scripts/evaluate_graph_rerank.py for test.
"""

import argparse

import torch

from mmxai.eval.graph_rerank import load_graph_rerank_config, prepare, train_and_select


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="YAML file in configs/")
    parser.add_argument("--seeds", type=int, nargs="+", required=True, help="e.g. 0 1 2")
    parser.add_argument("--threads", type=int, help="CPU threads for torch (default: torch's)")
    args = parser.parse_args()

    if args.threads is not None:
        torch.set_num_threads(args.threads)
    config = load_graph_rerank_config(args.config)
    prepared = prepare(config)
    for seed in args.seeds:
        run_dir = train_and_select(config, prepared, seed)
        print(f"saved {run_dir}")


if __name__ == "__main__":
    main()
