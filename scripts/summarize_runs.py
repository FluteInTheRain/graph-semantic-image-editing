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
    Extra tables (only from experiments ending in _eval2):
        --kind groups --experiment-suffix _eval2 --output results/tables/groups.csv
        --kind image_to_text --experiment-suffix _eval2 --output results/tables/image_to_text.csv

Output:
    results/tables/main.csv (committed). Report numbers come only from this file (ADR-0010).
"""

import argparse

from mmxai.eval.summary import (
    MIN_SEEDS,
    load_run_records,
    summarize_groups,
    summarize_image_to_text,
    summarize_runs,
    write_table,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs-dir", default="results/runs", help="folder of run folders")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--output", default="results/tables/main.csv", help="CSV to write")
    parser.add_argument(
        "--kind",
        default="main",
        choices=["main", "groups", "image_to_text"],
        help="which table to build",
    )
    parser.add_argument(
        "--experiment-suffix", default=None, help="keep only experiments ending with this"
    )
    args = parser.parse_args()

    records = load_run_records(args.runs_dir)
    if args.kind == "main":
        rows, skipped = summarize_runs(records, args.split, args.experiment_suffix)
    elif args.kind == "groups":
        rows, skipped = summarize_groups(records, args.split, args.experiment_suffix)
    else:
        rows, skipped = summarize_image_to_text(records, args.split, args.experiment_suffix)
    for experiment in skipped:
        print(f"skipped {experiment}: fewer than {MIN_SEEDS} seeds with committed code")
    write_table(rows, args.output)
    print(f"saved {args.output}: {len(rows)} rows")


if __name__ == "__main__":
    main()
