"""Train GraphSAGE on the PCST subgraph; early-stop and choose alpha on VAL (ADR-0051).

Logic: mmxai.eval.pcst_sage_rerank.train_and_select. Never touches test.

Before running:
    scripts/prepare_pcst_cache.py finished for the same config (and seed, if shuffled).

Run (from the repo root), once per seed:
    for s in 0 1 2; do
      uv run python scripts/train_pcst_sage.py --config configs/clip_pcst_sage_vg.yaml --seed $s
    done
    (the same for clip_pcst_sage_vg_no_tree.yaml and clip_pcst_sage_vg_shuffled.yaml)

Output:
    artifacts/models/visual_genome/<experiment>/seed<k>/model.pt and selection.json
    results/runs/<YYYYMMDD-HHMM>_<experiment>_seed<k>/ (split: val)
"""

import argparse

from mmxai.eval.pcst_sage_rerank import load_pcst_sage_config, train_and_select


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="YAML file in configs/")
    parser.add_argument("--seed", type=int, required=True, help="e.g. 0")
    args = parser.parse_args()

    config = load_pcst_sage_config(args.config)
    run_dir = train_and_select(config, args.seed)
    print(f"saved {run_dir}")


if __name__ == "__main__":
    main()
