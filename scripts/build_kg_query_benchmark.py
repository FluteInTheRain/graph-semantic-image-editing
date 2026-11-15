r"""Build the official KG benchmark from the human-reviewed manifest (ADR-0050).

Logic: mmxai.queries.validator.run_build_benchmark.

What it does:
    Keeps ONLY pairs whose manifest row is "verified" (reviewer and a different
    checker filled in, human-confirmed answer ids from the test split), replaces
    the candidate answers by the confirmed ones, checks the result and writes it.
    Candidate, reviewed and rejected rows never enter the benchmark.

Before running:
    The candidates JSONL and the manifest written by scripts/generate_kg_queries.py,
    with at least min_verified_pairs rows reviewed (see docs/query_benchmark.md).

Run (from the repo root):
    uv run python scripts/build_kg_query_benchmark.py \
        --config configs/query_benchmark_vg_kg.yaml

Output:
    <benchmark_path> (queries/visual_genome_kg_test.jsonl). Refuses to overwrite it:
    once committed it is frozen (ADR-0009, ADR-0023).
"""

import argparse

from mmxai.queries.kg_generator import load_kg_query_config
from mmxai.queries.validator import run_build_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="YAML file in configs/")
    args = parser.parse_args()

    config = load_kg_query_config(args.config)
    if config.split != "test":
        raise SystemExit("the benchmark is built from test candidates only (ADR-0009)")
    num_pairs = run_build_benchmark(
        config.candidates_path,
        config.manifest_path,
        config.split_file,
        config.benchmark_path,
        config.min_verified_pairs,
    )
    print(f"wrote {num_pairs} verified pairs ({2 * num_pairs} queries) to {config.benchmark_path}")


if __name__ == "__main__":
    main()
