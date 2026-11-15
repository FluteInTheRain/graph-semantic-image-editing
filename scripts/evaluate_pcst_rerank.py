"""Score TEST once with the PCST reranker setting chosen on val (ADR-0051, ADR-0009).

Logic: mmxai.eval.pcst_rerank.evaluate_test. Chooses nothing: it reads the selection.json
written by scripts/select_pcst_rerank.py and refuses to run the same experiment and seed twice.

Before running:
    scripts/select_pcst_rerank.py finished for the same config and seed, and the code is
    committed (scripts/summarize_runs.py drops a run made with uncommitted code).

Run (from the repo root), once per seed, only after the config is final:
    for s in 0 1 2; do
      uv run python scripts/evaluate_pcst_rerank.py --config configs/clip_pcst_vg.yaml --seed $s
    done

Output:
    results/runs/<YYYYMMDD-HHMM>_<experiment>_seed<k>/ (split: test)
    Then: uv run python scripts/summarize_runs.py
"""

import argparse

from mmxai.eval.pcst_rerank import evaluate_test, load_pcst_rerank_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="YAML file in configs/")
    parser.add_argument("--seed", type=int, required=True, help="e.g. 0")
    args = parser.parse_args()

    config = load_pcst_rerank_config(args.config)
    run_dir = evaluate_test(config, args.seed)
    print(f"saved {run_dir}")


if __name__ == "__main__":
    main()
