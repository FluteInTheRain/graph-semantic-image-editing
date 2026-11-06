"""Generate candidate direct/indirect query pairs from Visual Genome scene graphs (ADR-0044).

Logic: mmxai.queries.generator (templates, filters) and mmxai.queries.validator (answers).

What it does:
    Reads objects.json + relationships.json for the images of ONE split, finds real
    relation chains (>= 2 relations sharing an object id), writes one direct and one
    indirect query per chain, searches every image of that split for the same chain
    (candidate answers), and writes a candidates JSONL plus a review manifest CSV.
    Nothing it writes is a benchmark yet: every row starts as "candidate".

Before running:
    uv sync
    data/visual_genome/objects.json and data/visual_genome/relationships.json
    (download: see docs/query_benchmark.md, section "Chuẩn bị dữ liệu").

Run (from the repo root):
    Pilot, 5 pairs on val (output in results/runs/, gitignored), to check the rules:
        uv run python scripts/generate_queries.py --config configs/query_benchmark_vg_pilot.yaml
    Candidates on test, after the rules are fixed (writes into queries/):
        uv run python scripts/generate_queries.py --config configs/query_benchmark_vg.yaml
    Both refuse to overwrite existing outputs.

Output:
    <candidates_path> (JSONL) and <manifest_path> (CSV), paths set in the config.
    Next step: two members review the manifest, then scripts/build_query_benchmark.py.
"""

import argparse
import json

from mmxai.queries.generator import load_query_benchmark_config, run_generation


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="YAML file in configs/")
    args = parser.parse_args()

    config = load_query_benchmark_config(args.config)
    summary = run_generation(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
