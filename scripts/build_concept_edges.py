"""Build concept -> concept co-occurrence edges from TRAIN annotations (ADR-0048).

Logic: mmxai.graph.cooccurrence.

What it does:
    Reads the objects of the TRAIN images only, maps them to the concepts of an
    existing contains graph, counts how many train images contain each pair of
    concepts, and keeps for each concept its top_m partners by PMI (pairs seen in at
    least min_pair_images train images, PMI > 0). Writes concept_edges.npz + meta.json.

Before running:
    data/visual_genome/objects.json (docs/guides/xay-do-thi.md, section 2)
    the contains graph in graph_dir (scripts/build_graph.py)

Run (from the repo root):
    uv run python scripts/build_concept_edges.py --config configs/concept_edges_vg_cooccurrence.yaml

Output:
    <output_dir>/concept_edges.npz and <output_dir>/meta.json (gitignored under
    artifacts/). Refuses to overwrite.
"""

import argparse
import json

from mmxai.graph.cooccurrence import load_concept_edge_config, run_build_concept_edges


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="YAML file in configs/")
    args = parser.parse_args()

    config = load_concept_edge_config(args.config)
    meta = run_build_concept_edges(config)
    summary = {key: value for key, value in meta.items() if key != "config"}
    print(json.dumps(summary, indent=2))
    print(f"saved {config.output_dir}")


if __name__ == "__main__":
    main()
