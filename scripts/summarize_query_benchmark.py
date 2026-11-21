"""Summarize the KG query benchmark runs into one table (ADR-0057).

Logic: mmxai.eval.query_benchmark_models.summarize_benchmark_runs.

What it does:
    Reads every run folder in the runs directory, groups the runs by experiment and
    writes one CSV row per experiment: Recall@1/5/10 and MRR@100 as mean and sample std
    (ddof=1) over seeds, for all queries, direct, indirect and every activity / object
    group. It covers both the plain CLIP runs (scripts/evaluate_query_benchmark.py,
    written to the same folder by configs/eval_query_benchmark_vg_kg.yaml) and the
    trained models (scripts/evaluate_query_benchmark_models.py).

    It refuses a repeated seed within an experiment and runs made on different benchmark
    files (benchmark_sha256), because all models must be judged on the same queries.

Before running:
    Every experiment evaluated for seeds 0, 1 and 2 (see docs/query_benchmark.md).

Run (from the repo root):
    uv run python scripts/summarize_query_benchmark.py

Output:
    results/tables/kg_query_benchmark.csv
"""

import argparse
import csv
from pathlib import Path

from mmxai.eval.query_benchmark_models import summarize_benchmark_runs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs-dir", default="results/query_benchmark/runs")
    parser.add_argument("--output", default="results/tables/kg_query_benchmark.csv")
    args = parser.parse_args()

    rows = summarize_benchmark_runs(args.runs_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # All rows share the same keys in the same order, so the first row fixes the columns.
    with open(output, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            # An empty std (single seed) or a missing scope is written as an empty cell.
            writer.writerow({key: "" if value is None else value for key, value in row.items()})
    print(f"saved {output} ({len(rows)} experiments)")


if __name__ == "__main__":
    main()
