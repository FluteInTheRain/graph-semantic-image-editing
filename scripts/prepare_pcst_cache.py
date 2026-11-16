r"""Cache the PCST subgraph S_q of every train and val text (ADR-0051, stage 2).

Logic: mmxai.eval.pcst_sage_rerank.prepare_caches. Never touches test (its cache is made
by scripts/evaluate_pcst_sage.py). Train caches only the texts that training keeps.

Before running:
    uv sync --group ml --group graph
    Visual Genome features, the graph and the PMI concept edges (see the config).

Run (from the repo root). First measure the speed; this writes nothing:
    uv run python scripts/prepare_pcst_cache.py --config configs/clip_pcst_sage_vg.yaml \
        --seed 0 --estimate 1000
Then the real run (about an hour per cache). The main and no_tree configs need ONE cache
for all seeds, so run them with seed 0 only; the shuffled config needs seeds 0 1 2:
    uv run python scripts/prepare_pcst_cache.py --config configs/clip_pcst_sage_vg.yaml --seed 0
    uv run python scripts/prepare_pcst_cache.py \
        --config configs/clip_pcst_sage_vg_no_tree.yaml --seed 0
    for s in 0 1 2; do
      uv run python scripts/prepare_pcst_cache.py \
          --config configs/clip_pcst_sage_vg_shuffled.yaml --seed $s
    done

Output:
    artifacts/pcst_cache/<experiment>/shared/ (or seed<k>/ when shuffled): train and val
    <split>.npz + <split>.meta.json. An existing cache is skipped, never overwritten.
"""

import argparse

from mmxai.eval.pcst_sage_rerank import (
    estimate_pcst_time,
    load_pcst_sage_config,
    prepare_caches,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="YAML file in configs/")
    parser.add_argument("--seed", type=int, required=True, help="e.g. 0")
    parser.add_argument(
        "--estimate", type=int, metavar="N", help="only time PCST on N train texts; write nothing"
    )
    args = parser.parse_args()

    config = load_pcst_sage_config(args.config)
    if args.estimate is not None:
        estimate_pcst_time(config, args.seed, args.estimate)
    else:
        prepare_caches(config, args.seed)


if __name__ == "__main__":
    main()
