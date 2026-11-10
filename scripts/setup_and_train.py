"""One-time setup + training for the CLIP + Graph pipeline (ADR-0046). Slow; run once.

What it does (each step is skipped when its output already exists, so re-running is safe):
    1. check the CLIP features and objects.json are there (both are downloaded by hand)
    2. build the CLIP top-k graph
    3. train the GNN for every seed (train images only, early stopping on val)
    4. re-rank val for every seed and choose alpha, stored next to each checkpoint
The environment itself is installed by `uv run --group ml --group graph` below.

Run (from the repo root):
    uv run --group ml --group graph python scripts/setup_and_train.py
    uv run --group ml --group graph python scripts/setup_and_train.py --seeds 0   # one seed
Behind a proxy with its own certificate, prefix the command with UV_NATIVE_TLS=1.

Then:
    uv run --group ml --group graph python scripts/run_pipeline.py --mode clip_graph
"""

import argparse
import json
from pathlib import Path

from mmxai.graph.pipeline import load_graph_config, run_build_graph
from mmxai.models.training import (
    ALPHA_FILE,
    CHECKPOINT_FILE,
    checkpoint_folder,
    load_gnn_config,
    train_gnn,
)
from mmxai.rerank.evaluate import run_rerank_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/gnn_vg.yaml", help="GNN config")
    parser.add_argument(
        "--graph-config", default="configs/graph_vg_clip_topk.yaml", help="graph config"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    config = load_gnn_config(args.config)
    graph_config = load_graph_config(args.graph_config)
    graph_ready = (Path(config.graph_dir) / "graph.npz").exists()

    print("== 1. Check the downloaded data")
    if not (Path(config.features_dir) / "meta.json").exists():
        raise SystemExit(
            f"Missing CLIP features in {config.features_dir}. "
            "Download them: docs/guides/chuan-bi-embedding-clip.md, section 2."
        )
    objects = Path(graph_config.data_dir) / graph_config.objects_file
    if not graph_ready and not objects.exists():
        raise SystemExit(
            f"Missing {objects} (needed to build the graph). "
            "Download it: docs/guides/xay-do-thi.md, section 2."
        )

    print("== 2. Build the graph")
    if graph_ready:
        print(f"already built: {config.graph_dir}")
    else:
        if graph_config.output_dir != config.graph_dir:
            raise SystemExit(
                f"{args.graph_config} writes {graph_config.output_dir} "
                f"but {args.config} reads {config.graph_dir}"
            )
        run_build_graph(graph_config)

    for seed in args.seeds:
        folder = checkpoint_folder(config, seed)
        print(f"== 3. Train seed {seed}")
        if (folder / CHECKPOINT_FILE).exists():
            print(f"already trained: {folder}")
        else:
            train_gnn(config, seed)

        print(f"== 4. Choose alpha on val, seed {seed}")
        if (folder / ALPHA_FILE).exists():
            print(f"already chosen: {json.loads((folder / ALPHA_FILE).read_text())}")
        else:
            run_dir = run_rerank_evaluation(config, seed, "val")
            print(f"val results: {run_dir}")

    print("Done. Try:")
    print("  uv run --group ml --group graph python scripts/run_pipeline.py --mode clip_graph")


if __name__ == "__main__":
    main()
