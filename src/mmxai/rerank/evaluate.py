"""Re-rank CLIP's top 100 with the trained GNN and score it (ADR-0046 section 4, ADR-0043).

val:  try every alpha of alpha_grid and pick the best (by MRR); report all of them.
test: use the alpha written in the config, which must be the one chosen on val
      (ADR-0009: nothing is chosen on test).
Each run also saves scores.npz (candidates, cosine, graph score per query), from which
the success / failure cases for the explanations are picked.
"""

from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from mmxai.models.training import (
    GnnConfig,
    load_bipartite,
    load_trained_model,
    resolve_device,
    score_candidates,
    split_queries,
)
from mmxai.rerank.combine import choose_alpha, rerank_metrics
from mmxai.utils.runs import collect_env, make_run_dir, write_run_files

EVAL_SPLITS = ("val", "test")


def run_rerank_evaluation(config: GnnConfig, seed: int, split_name: str) -> Path:
    """Evaluate on one split, write the run folder (ADR-0010) and return its path."""
    if split_name not in EVAL_SPLITS:
        raise ValueError(f"split must be one of {EVAL_SPLITS}, got {split_name!r}")
    if split_name == "test" and config.alpha is None:
        raise ValueError("set `alpha` in the config to the value chosen on val before test")

    device = resolve_device(config.device)
    graph = load_bipartite(config, train_only=False, device=device)
    model = load_trained_model(config, seed, graph.image_x.shape[1], device)
    gallery_rows, query_x, relevant, text_ids = split_queries(config, graph, split_name)
    candidates, cosine, graph_score = score_candidates(
        model, graph, gallery_rows, query_x, config.top_k
    )

    metrics: dict = {"split": split_name, "seed": seed, "num_images": len(gallery_rows)}
    if split_name == "val":
        alpha, by_alpha = choose_alpha(
            candidates, cosine, graph_score, relevant, config.alpha_grid, config.recall_ks
        )
        metrics["chosen_alpha"] = alpha
        metrics["by_alpha"] = {str(value): result for value, result in by_alpha.items()}
    else:
        alpha = config.alpha
        result, _ = rerank_metrics(
            candidates, cosine, graph_score, alpha, relevant, config.recall_ks
        )
        metrics["alpha"] = alpha
        metrics["metrics"] = result
    metrics["split_sha256"] = graph.split_sha256

    run_dir = make_run_dir(config.output_dir, config.experiment, seed, datetime.now())
    env = collect_env({"numpy": np.__version__, "torch": torch.__version__})
    write_run_files(run_dir, asdict(config), metrics, env)
    np.savez(
        run_dir / "scores.npz",
        text_ids=text_ids,  # region ids, to look the sentences up for the report
        candidate_image_ids=graph.image_ids[candidates],
        cosine=cosine,
        graph=graph_score,
        alpha=np.float32(alpha),
    )
    return run_dir
