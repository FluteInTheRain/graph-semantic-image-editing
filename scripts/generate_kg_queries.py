r"""Generate candidate KG direct/indirect query pairs for Visual Genome (ADR-0057).

Logic: mmxai.queries.kg_generator.

Before running:
    uv sync
    data/visual_genome/objects.json, data/visual_genome/relationships.json
    artifacts/conceptnet/en_edges.tsv (scripts/prepare_conceptnet.py)

Needs:
    Several GB of RAM: both annotation files are read once and train graphs for
    ~98k images are built.

Run (from the repo root):
    Pilot on val (output in results/runs/, gitignored):
        uv run python scripts/generate_kg_queries.py \
            --config configs/query_benchmark_vg_kg_pilot.yaml
    Candidates on test, only after the team fixed the config (writes into queries/):
        uv run python scripts/generate_kg_queries.py \
            --config configs/query_benchmark_vg_kg.yaml
    Both refuse to overwrite existing outputs.

Output:
    <candidates_path> (JSONL) and <manifest_path> (CSV). Next: two members review the
    manifest, then scripts/build_kg_query_benchmark.py.
"""

import argparse
import json

from mmxai.queries.kg_generator import load_kg_query_config, run_kg_generation


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="YAML file in configs/")
    args = parser.parse_args()

    summary = run_kg_generation(load_kg_query_config(args.config))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
