"""Look at the query groups on val before trusting their numbers (spec 2026-11-13, section 5.3).

What it does:
    Labels every val text -> image query by specificity (specific / generic) and by
    relation type (action / spatial / none), exactly as the evaluation does, then prints
    the size of every group and 30 random sentences per relation group. Read the
    sentences: a group whose sentences do not match its label means the predicate
    lists need a tweak (chosen here on val, never on test).
    Spec: docs/superpowers/specs/2026-11-13-cap-do-1-mo-rong-design.md

Run (from the repo root; needs the CLIP features and data/visual_genome):
    uv run python scripts/inspect_query_groups.py --config configs/clip_vg_eval2.yaml

It refuses every split except val (--split only accepts val), and it writes nothing.
"""

import argparse

import numpy as np

from mmxai.data.splits import load_split
from mmxai.data.visual_genome import load_region_texts
from mmxai.eval.clip_baseline import load_clip_eval_config
from mmxai.eval.extended import query_group_inputs, query_groupings
from mmxai.eval.query_groups import group_samples, group_sizes
from mmxai.features.cache import load_image_features, load_text_features
from mmxai.rerank.fusion import retrieve_candidates


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="a ClipEvalConfig _eval2 file in configs/")
    parser.add_argument("--split", choices=["val"], default="val", help="only val is allowed")
    parser.add_argument("--per-group", type=int, default=30, help="sentences shown per group")
    args = parser.parse_args()

    config = load_clip_eval_config(args.config)
    if config.vg_data_dir is None or config.predicates_file is None:
        raise ValueError("the config needs vg_data_dir and predicates_file")

    # Same wiring as run_clip_evaluation, so the labels are the ones the evaluation uses.
    image_embeddings, image_ids = load_image_features(config.features_dir)
    texts = load_text_features(config.features_dir)
    split_image_ids = load_split(config.split_file)[args.split]
    query_text_ids, predicates = query_group_inputs(
        texts,
        split_image_ids,
        config.predicates_file,
        config.extra_action_predicates,
        config.extra_spatial_predicates,
        config.excluded_predicates,
    )
    candidates = retrieve_candidates(
        image_embeddings,
        image_ids,
        texts["embeddings"],
        texts["image_ids"],
        split_image_ids,
        config.top_k,
        config.same_text_threshold,
    )
    groupings = query_groupings(candidates, query_text_ids, config.vg_data_dir, predicates)

    for grouping_name, labels in groupings.items():
        print(f"grouping {grouping_name}:")
        for label, (count, fraction) in group_sizes(labels).items():
            print(f"  {label}: {count} queries ({fraction:.1%})")

    region_texts = {}
    for record in load_region_texts(config.vg_data_dir):
        region_texts[record.text_id] = record.text
    sentences = [region_texts[text_id] for text_id in np.asarray(query_text_ids).tolist()]
    samples = group_samples(groupings["relation"], sentences, args.per_group, seed=0)
    for label, chosen in samples.items():
        print(f"\nrelation group {label}: {len(chosen)} random sentences")
        for sentence in chosen:
            print(f"  [{label}] {sentence}")


if __name__ == "__main__":
    main()
