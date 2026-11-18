"""Choose prize_k, edge_cost and alpha of the PCST reranker on VAL (ADR-0051, stage 1).

Logic: mmxai.eval.pcst_rerank.select_on_val. Never touches test.

What it does (no training):
    1. Per (prize_k, edge_cost): builds the PCST subgraph S_q of every val query on the
       train-counted PMI concept graph and scores the 100 CLIP candidates by
       |N(i) & S_q| / |S_q|.
    2. Tries every alpha, keeps the setting with the best val MRR@100, saves it, and writes
       a val run folder. The ablations use configs/clip_pcst_vg_no_tree.yaml (no tree) and
       configs/clip_pcst_vg_shuffled.yaml (shuffled PMI pairs, shuffle seed = --seed).

Before running:
    uv sync --group ml --group graph          (pins numpy<2, which pcst-fast needs)
    Visual Genome features in artifacts/features/visual_genome/ViT-B-32__laion2b_s34b_b79k/
        (extracted on Kaggle, ADR-0041)
    uv run python scripts/build_graph.py --config configs/graph_vg_clip_topk.yaml
    uv run python scripts/build_concept_edges.py --config configs/concept_edges_vg_cooccurrence.yaml

Run (from the repo root), once per seed:
    for s in 0 1 2; do
      uv run python scripts/select_pcst_rerank.py --config configs/clip_pcst_vg.yaml --seed $s
    done

Output:
    artifacts/models/visual_genome/<experiment>/seed<k>/selection.json
    results/runs/<YYYYMMDD-HHMM>_<experiment>_seed<k>/ (split: val)
    Then, once, on committed code: scripts/evaluate_pcst_rerank.py for test.
"""

import argparse

from mmxai.eval.pcst_rerank import load_pcst_rerank_config, select_on_val


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="YAML file in configs/")
    parser.add_argument("--seed", type=int, required=True, help="e.g. 0")
    args = parser.parse_args()

    config = load_pcst_rerank_config(args.config)
    run_dir = select_on_val(config, args.seed)
    print(f"saved {run_dir}")


if __name__ == "__main__":
    main()
