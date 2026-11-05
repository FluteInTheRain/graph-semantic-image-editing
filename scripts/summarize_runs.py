"""Summarise run folders into a committed table. All logic lives in mmxai.eval.summary."""

import argparse

from mmxai.eval.summary import load_run_records, summarize_runs, write_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
