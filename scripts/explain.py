"""Explain 5 success + 5 failure VAL queries of CLIP + Graph (PDF section 7, ADR-0016).

Logic: mmxai.explain.query_analysis. Never touches test.

What it does:
    1. Loads the weights and the alpha that scripts/train_graph_rerank.py saved for
       this seed, and scores every val text exactly as training did.
    2. Per query: rank of the first relevant image under plain CLIP and under CLIP + Graph.
    3. Picks 5 success (largest rank gain) and 5 failure (largest rank loss) queries,
       by the rule in mmxai.eval.query_ranks (see its docstring for when each rule was fixed).
    4. Explains the relevant image (and, for a failure, the image CLIP + Graph put first)
       with Occlusion on the trained model: the concept -> image edges whose removal
       changes the graph score most. Then removes the top --num-edges supporting edges
       together and reports graph score and rank before and after (deletion check),
       and the same without any edge into the image (what its own vector gives).
    Stops if the val ranks do not give exactly the val metric saved by training.

Before running:
    uv sync --group ml --group graph
    The trained model: scripts/train_graph_rerank.py with the same --config and seed.
    Optional, for the query text: Visual Genome region_descriptions.json (several GB).
    The feature cache holds only region ids, so without it the text shows as an id.

Run (from the repo root; a few minutes on a laptop CPU):
    uv run python scripts/explain.py --config configs/clip_graph_vg_lr1e4.yaml --seed 0 \
        --split val --num-edges 3 --vg-data-dir data/visual_genome

Output:
    results/runs/<YYYYMMDD-HHMM>_<experiment>_explain_seed<k>/ with analysis.json (all
    numbers), analysis.md (readable), config.yaml and env.json. No metrics.json, so
    scripts/summarize_runs.py does not count it as a run.
"""

import argparse
from datetime import datetime

import torch

from mmxai.eval.graph_rerank import load_graph_rerank_config
from mmxai.explain.query_analysis import analyse_split, write_analysis


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="YAML file in configs/")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", default="val", help="only val is allowed (ADR-0009)")
    parser.add_argument("--num-edges", type=int, default=3, help="relations shown and removed")
    parser.add_argument("--vg-data-dir", help="folder with region_descriptions.json (optional)")
    parser.add_argument("--threads", type=int, help="CPU threads for torch (default: torch's)")
    args = parser.parse_args()

    if args.threads is not None:
        torch.set_num_threads(args.threads)
    config = load_graph_rerank_config(args.config)
    analysis = analyse_split(config, args.seed, args.split, args.num_edges, args.vg_data_dir)
    cli_settings = {
        "seed": args.seed,
        "split": args.split,
        "num_edges": args.num_edges,
        "vg_data_dir": args.vg_data_dir,
    }
    run_dir = write_analysis(config, analysis, cli_settings, datetime.now())
    print(f"saved {run_dir}")


if __name__ == "__main__":
    main()
