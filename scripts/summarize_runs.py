"""Summarise run folders into the committed results table. Logic lives in mmxai.eval.summary.

What it does:
    Reads every results/runs/*/ folder, keeps the runs of one split that were made
    on committed code, and writes one row per experiment with mean and sample std
    over seeds. Refuses a seed run twice; leaves out experiments with < 3 seeds.

Before running:
    Run folders made by scripts/evaluate_clip.py (or later CLIP + Graph scripts).

Run (from the repo root, under a second):
    uv run python scripts/summarize_runs.py
    Options: --runs-dir results/runs  --split test  --output results/tables/main.csv

Output:
    results/tables/main.csv (committed). Report numbers come only from this file (ADR-0010).
"""

import argparse

from mmxai.eval.summary import load_run_records, summarize_runs, write_table


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs-dir", default="results/runs", help="folder of run folders")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--output", default="results/tables/main.csv", help="CSV to write")
    args = parser.parse_args()

    rows, skipped = summarize_runs(load_run_records(args.runs_dir), args.split)
    for experiment in skipped:
        print(f"skipped {experiment}: fewer than 3 seeds with committed code")
    write_table(rows, args.output)
    print(f"saved {args.output}: {len(rows)} experiments")


if __name__ == "__main__":
    main()
