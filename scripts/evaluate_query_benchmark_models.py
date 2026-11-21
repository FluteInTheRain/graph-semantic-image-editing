"""Evaluate a trained model on the verified KG query benchmark (ADR-0057).

Logic: mmxai.eval.query_benchmark_models.

What it does:
    Loads the model saved by training (model.pt and selection.json, whose alpha was
    chosen on val), encodes the benchmark queries with the same frozen CLIP as the image
    embeddings, takes CLIP's exact top 100 of the 5000 test images and re-ranks them with
    the model. It reports Recall@1/5/10/100 and MRR@100 for all queries, direct only and
    indirect only, plus the rank of every query.

    The benchmark is evaluation-only (PDF section 6): never change a model, alpha or any
    setting after looking at these numbers. A second run of the same experiment and seed
    is refused.

Before running:
    uv sync --group ml --group graph
    Visual Genome features and graph in artifacts/ (see the model config)
    The trained model in artifacts/models/visual_genome/<experiment>/seed<k>/
    queries/visual_genome_kg_test.jsonl built by scripts/build_query_benchmark.py

Run (from the repo root), once per seed on committed code:
    for s in 0 1 2; do
        uv run python scripts/evaluate_query_benchmark_models.py \
            --config configs/eval_query_benchmark_vg_kg_clip_graph_vg_lr1e4.yaml --seed $s
    done

Output:
    results/query_benchmark/runs/<YYYYMMDD-HHMM>_<experiment>_seed<k>/
    with config.yaml, metrics.json (all / direct / indirect / per_query) and env.json.
"""

import argparse
import json

from mmxai.eval.query_benchmark_models import (
    load_model_benchmark_eval_config,
    run_model_benchmark_evaluation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="YAML file in configs/")
    parser.add_argument("--seed", type=int, required=True, help="seed of the trained model")
    args = parser.parse_args()

    config = load_model_benchmark_eval_config(args.config)
    run_dir = run_model_benchmark_evaluation(config, args.seed)
    with open(run_dir / "metrics.json", encoding="utf-8") as file:
        metrics = json.load(file)
    metrics.pop("per_query")
    print(json.dumps(metrics, indent=2))
    print(f"saved {run_dir}")


if __name__ == "__main__":
    main()
