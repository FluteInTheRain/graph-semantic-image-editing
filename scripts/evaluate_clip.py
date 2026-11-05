"""Evaluate plain CLIP text-to-image retrieval (ADR-0043). Logic: mmxai.eval.clip_baseline.

What it does:
    Takes the images and region descriptions of one split, searches the images with
    every text (FAISS, top 100), and reports Recall@1/5/10 and MRR@100.

Before running:
    uv sync --group ml
    Visual Genome features in artifacts/features/visual_genome/ViT-B-32__laion2b_s34b_b79k/
    (how to download: see the docstring of scripts/make_splits.py).

Run (from the repo root, about 4 seconds):
    Val, as often as you like while developing:
        uv run python scripts/evaluate_clip.py --config configs/clip_vg.yaml --seed 0
    Test, ONCE per seed, on committed code, changing nothing afterwards (ADR-0009):
        for s in 0 1 2; do
            uv run python scripts/evaluate_clip.py --config configs/clip_vg.yaml \
                --seed $s --split test
        done
    Test was already run on 2026-11-05 (commit c1678ef); do not run it again,
    scripts/summarize_runs.py refuses a seed run twice.

Output:
    results/runs/<YYYYMMDD-HHMM>_clip_vg_seed<k>/ with config.yaml, metrics.json
    (metrics, split hash, feature code hash) and env.json (commit, dirty flag, versions).
    Then build the table with scripts/summarize_runs.py.
"""

import argparse
import json
from dataclasses import replace

from mmxai.eval.clip_baseline import load_clip_eval_config, run_clip_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="YAML file in configs/")
    parser.add_argument("--seed", type=int, required=True, help="run label; CLIP itself is fixed")
    parser.add_argument("--split", choices=["val", "test"], help="override eval_split")
    args = parser.parse_args()

    config = load_clip_eval_config(args.config)
    if args.split is not None:
        config = replace(config, eval_split=args.split)
    run_dir = run_clip_evaluation(config, args.seed)
    with open(run_dir / "metrics.json", encoding="utf-8") as file:
        print(json.dumps(json.load(file), indent=2))
    print(f"saved {run_dir}")


if __name__ == "__main__":
    main()
