"""Level-3 explanation evaluation on the Vietnamese MMKG (ADR-0054). Logic: mmxai.explain.mmkg.

What it does:
    Builds the MMKG graph and CLIP candidates of the transfer evaluation (ADR-0053),
    loads the one-layer VG model and VG alpha of every seed, checks the ranking gives
    the MRR of the committed transfer run, then explains the rank-1 image of all 318
    queries with Integrated Gradients (m = 10, chosen on VG val) and measures fidelity,
    the random control, sparsity and validity. Runs once: a second run is refused.

Before running:
    uv sync --group ml --group graph
    scripts/evaluate_mmkg_transfer.py features and evaluate already run (the features
    cache and the clip_graph_vg_lr1e4_mmkg_vi run folders are read).

Run (from the repo root):
    uv run python scripts/evaluate_mmkg_explanations.py --config configs/explain_mmkg_vi.yaml \
        --transfer-config configs/mmkg_vi_transfer.yaml

Output:
    results/runs/<time>_explain_mmkg_vi_test/ (evaluation.json, summary.md, config.yaml,
    env.json) and results/tables/xai_mmkg.csv (overall and per query group).
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import yaml

from mmxai.data.mmkg_queries import load_mmkg
from mmxai.eval.graph_rerank import load_graph_rerank_config
from mmxai.eval.mmkg_transfer import load_features, load_vg_model
from mmxai.eval.summary import write_table
from mmxai.explain.config import load_explain_config
from mmxai.explain.evaluation import check_settings, make_explain_run_dir, write_evaluation
from mmxai.explain.mmkg import explain_seed, mmkg_graph_and_candidates, summarize_mmkg, table_rows
from mmxai.utils.runs import file_sha256

TABLE = "results/tables/xai_mmkg.csv"
# The transfer runs of the explained model (ADR-0053), one per seed.
TRANSFER_EXPERIMENT = "clip_graph_vg_lr1e4_mmkg_vi"


def transfer_mrr(runs_dir: Path, seed: int) -> float:
    """MRR@100 the transfer run recorded for this seed; exactly one run must exist."""
    found = sorted(runs_dir.glob(f"*_{TRANSFER_EXPERIMENT}_seed{seed}/metrics.json"))
    if len(found) != 1:
        raise FileNotFoundError(f"expected one {TRANSFER_EXPERIMENT} seed {seed} run, got {found}")
    with open(found[0], encoding="utf-8") as file:
        return json.load(file)["MRR@100"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="configs/explain_mmkg_vi.yaml")
    parser.add_argument("--transfer-config", required=True, help="configs/mmkg_vi_transfer.yaml")
    args = parser.parse_args(argv)

    config = load_explain_config(args.config)
    ranking = load_graph_rerank_config(config.ranking_config)
    check_settings(config, ranking)
    with open(args.transfer_config, encoding="utf-8") as file:
        transfer = yaml.safe_load(file)

    arrays, meta = load_features(Path(transfer["features_dir"]))
    if meta["queries_sha256"] != file_sha256(transfer["queries_file"]):
        raise ValueError("features were built from another queries file")
    with open(transfer["queries_file"], encoding="utf-8") as file:
        queries = [json.loads(line) for line in file if line.strip()]
    if len(queries) != config.num_queries:
        raise ValueError(f"config says {config.num_queries} queries, the file has {len(queries)}")
    _, triples, _ = load_mmkg(Path(transfer["mmkg_dir"]))
    graph, candidates = mmkg_graph_and_candidates(transfer, arrays, queries, triples)

    # Created before any work, so a second run stops here (ADR-0054 rule 7).
    run_dir = make_explain_run_dir(config.output_dir, config.experiment, "test", datetime.now())
    records = []
    alphas = {}
    for seed in config.seeds:
        seed_dir = Path(ranking.model_dir) / f"seed{seed}"
        with open(seed_dir / "selection.json", encoding="utf-8") as file:
            alpha = float(json.load(file)["alpha"])
        alphas[seed] = alpha
        model = load_vg_model(seed_dir / "model.pt", False, arrays["image_features"].shape[1])
        expected = transfer_mrr(Path(transfer["output_dir"]), seed)
        print(f"seed {seed}: alpha {alpha}, reproducing transfer MRR {expected:.6f}")
        records += explain_seed(
            model,
            alpha,
            seed,
            graph,
            candidates,
            queries,
            arrays["concept_keys"].tolist(),
            config,
            expected,
        )

    summary = summarize_mmkg(records, config)
    evaluation = {
        "settings": {
            "experiment": config.experiment,
            "ranking_experiment": f"{ranking.experiment} on MMKG",
            "split": "test",
            "seeds": config.seeds,
            "alpha_per_seed": alphas,
            "query_rows": list(range(len(queries))),
            "explained_image": "rank 1 under CLIP + Graph",
            "explainers": [config.main_explainer],
            "top_m_options": [config.top_m],
            "queries_sha256": meta["queries_sha256"],
        },
        "summary": summary["all"],
        "groups": summary,
        "records": records,
    }
    write_evaluation(run_dir, evaluation, config)
    write_table(table_rows(summary, run_dir.name), TABLE)
    print((run_dir / "summary.md").read_text(encoding="utf-8"))
    print(f"saved {run_dir} and {TABLE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
