"""Diagnostic (val only): each trained seed's val MRR@100 with and without its graph edges.

Same trained weights and the alpha saved on val; "no edges" removes every concept -> image
edge of the val graph, so each image keeps only its own vector (W_self x_i + b). Nothing is
chosen from this; it was run once on 2026-11-13 to answer a review question (PR #8).

Run (from the repo root, about a minute per config):
    uv run python results/records/20261113_cap-do-1/diagnostics/global_noedges.py \
        configs/clip_graph_vg_lr1e4.yaml
Output of 2026-11-13: global_noedges_output.txt next to this file.
"""

import dataclasses
import json
import sys

import torch

from mmxai.eval.graph_rerank import (
    MODEL_FILE,
    SELECTION_FILE,
    build_split_graph,
    image_vectors,
    load_graph_rerank_config,
    load_inputs,
    make_model,
    seed_model_dir,
    split_candidates,
)
from mmxai.eval.metrics import compute_metrics
from mmxai.rerank.fusion import candidate_graph_scores, fuse_scores, rerank_ranks


def main() -> None:
    config = load_graph_rerank_config(sys.argv[1])
    inputs = load_inputs(config)
    graph = build_split_graph(inputs, "val")
    graph_without_edges = dataclasses.replace(graph, edge_index=graph.edge_index[:, :0])
    candidates = split_candidates(config, inputs, "val")

    for seed in (0, 1, 2):
        model_dir = seed_model_dir(config, seed)
        with open(model_dir / SELECTION_FILE, encoding="utf-8") as file:
            alpha = float(json.load(file)["alpha"])
        model = make_model(config, graph.x.shape[1])
        model.load_state_dict(torch.load(model_dir / MODEL_FILE, weights_only=True))

        for name, split_graph in (("with edges", graph), ("no edges", graph_without_edges)):
            h = image_vectors(model, split_graph)
            graph_scores = candidate_graph_scores(candidates, h, split_graph.image_ids)
            final_scores = fuse_scores(candidates.clip_scores, graph_scores, alpha)
            ranks = rerank_ranks(candidates, final_scores)
            mrr = compute_metrics(ranks, config.recall_ks, config.top_k)["MRR@100"]
            print(f"{config.experiment} seed {seed} alpha {alpha} {name} MRR@100 {mrr:.5f}")


if __name__ == "__main__":
    main()
