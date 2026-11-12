"""Score CLIP + Graph on TEST, once per seed (ADR-0009, ADR-0047). Logic: mmxai.eval.graph_rerank.

What it does:
    Loads the weights and the alpha that scripts/train_graph_rerank.py chose on val,
    re-ranks CLIP's top 100 for every test text, and reports Recall@1/5/10 and MRR@100.
    It chooses nothing: no alpha sweep, no epoch choice.

Rules (ADR-0009):
    - Run only after the config is locked and the code is committed (a dirty tree
      is left out of results/tables/ by scripts/summarize_runs.py).
    - Once per experiment and seed. A second test run of the same seed is refused.

Run (from the repo root, about 1 min per seed):
    for s in 0 1 2; do
        uv run python scripts/evaluate_graph_rerank.py --config configs/clip_graph_vg.yaml --seed $s
    done

Output:
    results/runs/<YYYYMMDD-HHMM>_<experiment>_seed<k>/ (split: test). Then build the
    table: uv run python scripts/summarize_runs.py
"""

import argparse

from mmxai.eval.graph_rerank import evaluate_test, load_graph_rerank_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="YAML file in configs/")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    config = load_graph_rerank_config(args.config)
    run_dir = evaluate_test(config, args.seed)
    print(f"saved {run_dir}")


if __name__ == "__main__":
    main()
