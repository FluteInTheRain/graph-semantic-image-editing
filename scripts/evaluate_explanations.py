"""Level-3 explanation evaluation on the chosen CLIP + Graph model (PDF section 5, ADR-0016).

Logic: mmxai.explain.evaluation (see its docstring for the flow).

What it does, for every seed in the explain config:
    1. Loads the weights and val-chosen alpha that scripts/train_graph_rerank.py saved for
       the model named by ranking_config (configs/clip_graph_vg_lr1e4.yaml), and ranks
       the split exactly as the pipeline does (on val, checks it reproduces training).
    2. Explains the image ranked first for num_queries fixed random queries with every
       explainer (Occlusion, Integrated Gradients, GNNExplainer) and every m.
    3. Measures fidelity+/-, rank drop, robust fidelity, the random control, sparsity and
       validity; summarizes mean ± std over seeds and a Wilcoxon test vs random.

Order (ADR-0009):
    1. eval_split: val. Read summary.md, write the proposed main_explainer and top_m into
       the explain config (a commit), decided by the team.
    2. eval_split: test, once. The script refuses a second test run.

Before running:
    uv sync --group ml --group graph
    The trained models: scripts/train_graph_rerank.py --config configs/clip_graph_vg_lr1e4.yaml
    for every seed in the explain config.

Run (from the repo root):
    uv run python scripts/evaluate_explanations.py --config configs/explain_vg.yaml

Output:
    results/runs/<YYYYMMDD-HHMM>_<experiment>_<split>/ with evaluation.json (every record
    and the summary), summary.md, config.yaml and env.json. No metrics.json, so
    scripts/summarize_runs.py does not count it as a model run.
"""

import argparse
from datetime import datetime

import torch

from mmxai.explain.config import load_explain_config
from mmxai.explain.evaluation import evaluate_split, make_explain_run_dir, write_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="YAML file in configs/ (explain_*.yaml)")
    parser.add_argument("--threads", type=int, help="CPU threads for torch (default: torch's)")
    args = parser.parse_args()

    if args.threads is not None:
        torch.set_num_threads(args.threads)
    config = load_explain_config(args.config)
    # Create the folder first: on test this refuses a second run before any work is done.
    run_dir = make_explain_run_dir(
        config.output_dir, config.experiment, config.eval_split, datetime.now()
    )
    evaluation = evaluate_split(config)
    write_evaluation(run_dir, evaluation, config)
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    main()
