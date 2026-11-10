"""One-time setup for the CLIP + Graph pipeline (ADR-0045, ADR-0046): data + graph.

What it does (each step is skipped when its output already exists, so re-running is safe):
    1. check the CLIP features are there (they come from the Kaggle notebook and are
       NOT downloaded here: docs/guides/chuan-bi-embedding-clip.md, section 2)
    2. download the Visual Genome annotation files that are missing (configs/data_vg.yaml):
       objects.json (required, sha256 checked) and region_descriptions.json (optional,
       only to print query sentences)
    3. build the CLIP top-k graph
The environment itself is installed by `uv run --group ml --group graph` below.

Run (from the repo root; a few minutes, CPU is enough):
    uv run --group ml --group graph python scripts/setup.py
Behind a proxy with its own certificate, prefix the command with UV_NATIVE_TLS=1.

Then:
    uv run --group ml --group graph python scripts/train.py
"""

import argparse
from pathlib import Path

from mmxai.data.download import download_file, load_data_config
from mmxai.graph.pipeline import load_graph_config, run_build_graph
from mmxai.models.training import load_gnn_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/gnn_vg.yaml", help="GNN config")
    parser.add_argument(
        "--graph-config", default="configs/graph_vg_clip_topk.yaml", help="graph config"
    )
    parser.add_argument("--data-config", default="configs/data_vg.yaml", help="files to download")
    args = parser.parse_args()

    config = load_gnn_config(args.config)
    graph_config = load_graph_config(args.graph_config)
    graph_ready = (Path(config.graph_dir) / "graph.npz").exists()

    print("== 1. Check the CLIP features")
    if not (Path(config.features_dir) / "meta.json").exists():
        raise SystemExit(
            f"Missing CLIP features in {config.features_dir}. "
            "Download them: docs/guides/chuan-bi-embedding-clip.md, section 2."
        )
    print(f"found: {config.features_dir}")

    print("== 2. Download the Visual Genome annotations")
    data_dir, remote_files = load_data_config(args.data_config)
    if data_dir != Path(graph_config.data_dir):
        raise SystemExit(
            f"{args.data_config} downloads into {data_dir} "
            f"but {args.graph_config} reads {graph_config.data_dir}"
        )
    for remote in remote_files:
        if graph_ready and remote.name == graph_config.objects_file:
            continue  # only the graph build reads it
        try:
            download_file(remote, data_dir)
        except Exception as error:
            if remote.required:
                raise
            # An optional file only adds readable sentences; training does not need it.
            print(f"skipped optional {remote.name}: {error}")

    print("== 3. Build the graph")
    if graph_ready:
        print(f"already built: {config.graph_dir}")
    else:
        if graph_config.output_dir != config.graph_dir:
            raise SystemExit(
                f"{args.graph_config} writes {graph_config.output_dir} "
                f"but {args.config} reads {config.graph_dir}"
            )
        run_build_graph(graph_config)

    print("Done. Next:")
    print("  uv run --group ml --group graph python scripts/train.py")


if __name__ == "__main__":
    main()
