"""The five KG-benchmark evaluation configs load and point at real model configs (ADR-0057)."""

from pathlib import Path

import pytest

from mmxai.eval.query_benchmark_models import load_model_benchmark_eval_config

EXPERIMENTS = {
    "clip_nograph_vg": "graph_rerank",
    "clip_graph_vg_lr1e4": "graph_rerank",
    "clip_cooc_graph_vg_lr1e4": "graph_rerank",
    "clip_nograph_vg_lr1e4": "graph_rerank",
    "clip_pcst_sage_vg": "pcst_sage",
}


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_eval_config_loads_and_points_to_existing_model_config(experiment):
    config = load_model_benchmark_eval_config(
        f"configs/eval_query_benchmark_vg_kg_{experiment}.yaml"
    )

    assert config.experiment == f"{experiment}_kg_query_benchmark"
    assert config.model_kind == EXPERIMENTS[experiment]
    assert config.model_config == f"configs/{experiment}.yaml"
    assert Path(config.model_config).exists()
    assert config.benchmark_path == "queries/visual_genome_kg_test.jsonl"
    assert config.recall_ks == [1, 5, 10, 100]
