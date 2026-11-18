"""Write the MMKG evaluation queries (ADR-0053). Logic: mmxai.data.mmkg_queries.

What it does:
    For every MMKG item with images, writes 4 queries (direct/indirect x English/
    Vietnamese) with their correct images to queries_file. Refuses to overwrite
    an existing file: committed query files are frozen (queries/README.md).

Run (from the repo root):
    uv run python scripts/make_mmkg_queries.py --config configs/mmkg_vi_transfer.yaml

Output:
    The query file, and counts per (type, lang) plus how many queries have more
    than one correct item.
"""

import argparse
from collections import Counter
from pathlib import Path

import yaml

from mmxai.data.mmkg_queries import build_queries, load_mmkg, query_record, write_query_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="YAML file in configs/")
    args = parser.parse_args(argv)
    with open(args.config, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    output = Path(config["queries_file"])
    if output.exists():
        print(f"{output} already exists; query files are frozen, delete it by hand to rebuild")
        return 1

    entities, triples, images_by_item = load_mmkg(Path(config["mmkg_dir"]))
    # Sorted, so the file does not depend on the CSV row order.
    item_ids = sorted(images_by_item)
    queries = build_queries(entities, triples, item_ids)
    write_query_file([query_record(query, images_by_item) for query in queries], output)

    print(f"wrote {len(queries)} queries to {output}")
    for (kind, lang), count in sorted(Counter((q.kind, q.lang) for q in queries).items()):
        print(f"  {kind:8s} {lang}: {count}")
    shared = sum(len(query.relevant_items) > 1 for query in queries)
    print(f"  queries with more than one correct item: {shared}")
    with_indirect = {query.item_id for query in queries if query.kind == "indirect"}
    for item_id in item_ids:
        if item_id not in with_indirect:
            name = entities[item_id].name_en
            print(f"  no indirect query (item has no triple): {item_id} {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
