"""Run the full pipeline on a few queries and explain each top result (xAI levels 1-2).

Logic: mmxai.explain.demo.

What it does:
    query -> CLIP text vector -> FAISS top 100 among the split's images
    clip:        rank by cosine only; the graph is not used, and the explanation says so.
    clip_graph:  re-rank with alpha * cosine + (1 - alpha) * graph score (trained GNN, alpha
                 chosen on val), then explain the top image by Occlusion on the model:
                 which concept -> image edges pushed it up or pulled it down.

Before running (once):
    uv run --group ml --group graph python scripts/setup_and_train.py

Run (from the repo root; a few seconds to a minute, mostly loading the features):
    5 random region descriptions of the test images, both modes:
        uv run --group ml --group graph python scripts/run_pipeline.py --mode clip
        uv run --group ml --group graph python scripts/run_pipeline.py --mode clip_graph
    More queries, another sample:
        uv run --group ml --group graph python scripts/run_pipeline.py --mode clip_graph \
            --num-queries 10 --sample-seed 1
    Your own sentences (no known answer, so no "correct" rank is shown):
        uv run --group ml --group graph python scripts/run_pipeline.py --mode clip_graph \
            --text "a dog on a sofa"

    Test images are only LOOKED AT here: change no setting after seeing them (ADR-0009).
    Use --split val while you are still tuning anything.

Output:
    Printed to the terminal: per query, the top images, the rank of the first correct
    image, and the explanation of the top image.
"""

import argparse

from mmxai.explain.demo import MODES, run_demo
from mmxai.explain.presentation import format_explanation
from mmxai.models.training import load_gnn_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--config", default="configs/gnn_vg.yaml", help="YAML file in configs/")
    parser.add_argument("--seed", type=int, default=0, help="seed of the trained model")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--num-queries", type=int, default=5, help="queries sampled from the split")
    parser.add_argument("--sample-seed", type=int, default=0, help="which queries are sampled")
    parser.add_argument("--text", action="append", default=[], help="your own query; repeatable")
    parser.add_argument("--show-top", type=int, default=5, help="images listed per query")
    parser.add_argument("--top-m", type=int, default=3, help="relations shown per side")
    parser.add_argument("--min-contribution", type=float, default=0.0)
    parser.add_argument(
        "--data-dir", default="data/visual_genome", help="where region_descriptions.json is"
    )
    args = parser.parse_args()

    results = run_demo(
        load_gnn_config(args.config),
        args.mode,
        args.seed,
        args.split,
        args.num_queries,
        args.text,
        args.sample_seed,
        args.show_top,
        args.top_m,
        args.min_contribution,
        args.data_dir,
    )
    for number, result in enumerate(results, start=1):
        print(f"\n=== Query {number} ({args.mode}): {result.query}")
        for rank, (image_id, score) in enumerate(
            zip(result.top_image_ids, result.top_scores, strict=True), start=1
        ):
            mark = ""
            if result.relevant_image_ids is not None and image_id in result.relevant_image_ids:
                mark = "  <- correct"
            print(f"  {rank}. image {image_id}  score {score:.4f}{mark}")
        if result.relevant_image_ids is not None:
            if result.first_relevant_rank is None:
                print("  First correct image: not in the top 100 (FAILED)")
            else:
                print(f"  First correct image at rank {result.first_relevant_rank}")
        print(f"  Why image {result.top_image_ids[0]} is first:")
        for line in format_explanation(result.explanation).splitlines():
            print(f"    {line}")


if __name__ == "__main__":
    main()
