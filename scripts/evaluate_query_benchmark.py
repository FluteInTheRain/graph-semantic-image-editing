"""Evaluate plain CLIP on the verified query benchmark, direct and indirect apart (ADR-0044).

Logic: mmxai.eval.query_benchmark.

What it does:
    Encodes the benchmark queries with the same frozen CLIP as the cached image
    embeddings, searches the 5000 test images (FAISS, exact, top 100) and reports
    Recall@1/5/10/100 and MRR@100 for all queries, direct only and indirect only,
    plus the rank of every query (for a paired comparison with CLIP + Graph later).

    The benchmark is evaluation-only (PDF section 6): never change a model, a
    template, a filter or alpha after looking at these numbers.

Before running:
    uv sync --group ml
    Visual Genome features in artifacts/features/visual_genome/ViT-B-32__laion2b_s34b_b79k/
    queries/visual_genome_test.jsonl built by scripts/build_query_benchmark.py
    (the script stops with a clear message if either is missing).

Run (from the repo root), once per seed on committed code:
    for s in 0 1 2; do
        uv run python scripts/evaluate_query_benchmark.py \
            --config configs/eval_query_benchmark_vg.yaml --seed $s
    done

Output:
    results/runs/<YYYYMMDD-HHMM>_clip_vg_query_benchmark_seed<k>/
    with config.yaml, metrics.json (all / direct / indirect / per_query) and env.json.
"""

import argparse
import json

from mmxai.eval.query_benchmark import (
    load_query_benchmark_eval_config,
    run_clip_benchmark_evaluation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="YAML file in configs/")
    parser.add_argument("--seed", type=int, required=True, help="run label; CLIP itself is fixed")
    args = parser.parse_args()

    config = load_query_benchmark_eval_config(args.config)
    run_dir = run_clip_benchmark_evaluation(config, args.seed)
    with open(run_dir / "metrics.json", encoding="utf-8") as file:
        metrics = json.load(file)
    metrics.pop("per_query")
    print(json.dumps(metrics, indent=2))
    print(f"saved {run_dir}")


if __name__ == "__main__":
    main()
