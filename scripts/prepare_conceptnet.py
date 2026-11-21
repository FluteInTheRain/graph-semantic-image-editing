"""Filter ConceptNet 5.7 into the small English TSV used by the KG queries (ADR-0057).

Logic: mmxai.data.conceptnet.prepare_conceptnet.

Before running:
    Download conceptnet-assertions-5.7.0.csv.gz into data/conceptnet/
    (https://github.com/commonsense/conceptnet5/wiki/Downloads; see data_cards/conceptnet.md).

Run (from the repo root):
    uv run python scripts/prepare_conceptnet.py --config configs/conceptnet_en.yaml

Output:
    <output_path> (TSV, in artifacts/, gitignored); the printed sha256 goes into the data card.
"""

import argparse
import json

from mmxai.data.conceptnet import load_conceptnet_prepare_config, prepare_conceptnet


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="YAML file in configs/")
    args = parser.parse_args()

    config = load_conceptnet_prepare_config(args.config)
    summary = prepare_conceptnet(
        config.source_path, config.output_path, config.relations, config.min_weight
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
