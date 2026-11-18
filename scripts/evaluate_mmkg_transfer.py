"""Run the Visual Genome models on the Vietnamese MMKG, no retraining (ADR-0053).
Logic: mmxai.eval.mmkg_transfer.

What it does:
    features   Encode the 380 MMKG images, the 119 entities ("a photo of a {name_en}")
               and the queries with frozen CLIP; cache them in features_dir.
    evaluate   For every experiment in the config (plain CLIP, one-layer GNN, two-layer
               GNN) and every seed: build the MMKG graph, load the VG weights and the
               VG alpha of that seed, score the queries once, write a run folder.
               A seed already run is refused: MMKG is evaluated once, like VG test.

Before running:
    uv sync --group ml --group graph
    queries_file from scripts/make_mmkg_queries.py, committed (runs on uncommitted
    code are left out of the summary, ADR-0010).
    The VG weights in artifacts/models/visual_genome/<config>/seed<k>/.

Run (from the repo root):
    uv run python scripts/evaluate_mmkg_transfer.py features --config configs/mmkg_vi_transfer.yaml
    uv run python scripts/evaluate_mmkg_transfer.py evaluate --config configs/mmkg_vi_transfer.yaml
    Then the tables:
    uv run python scripts/summarize_runs.py --experiment-suffix _mmkg_vi \
        --output results/tables/mmkg_transfer.csv
    uv run python scripts/summarize_runs.py --kind groups --experiment-suffix _mmkg_vi \
        --output results/tables/mmkg_transfer_groups.csv

Output:
    features_dir/features.npz + meta.json; results/runs/<time>_<experiment>_seed<k>/.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import yaml

from mmxai.data.mmkg_queries import load_mmkg
from mmxai.eval.graph_rerank import library_versions
from mmxai.eval.mmkg_transfer import encode_mmkg, load_features, run_transfer, save_features
from mmxai.utils.runs import (
    collect_env,
    file_sha256,
    git_state,
    make_run_dir,
    test_run_exists,
    write_run_files,
)


def read_queries(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def features(config: dict) -> None:
    entities, _, images_by_item = load_mmkg(Path(config["mmkg_dir"]))
    queries = read_queries(Path(config["queries_file"]))
    image_keys = sorted(key for keys in images_by_item.values() for key in keys)
    arrays = encode_mmkg(config, entities, queries, image_keys)
    meta = {
        "model": config["model"],
        "pretrained": config["pretrained"],
        "concept_prompt": config["concept_prompt"],
        "queries_file": config["queries_file"],
        "queries_sha256": file_sha256(config["queries_file"]),
        "num_images": len(image_keys),
        "num_concepts": len(entities),
        "num_queries": len(queries),
        "git": git_state(),
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    save_features(arrays, meta, Path(config["features_dir"]))
    counts = f"{meta['num_images']} images, {meta['num_concepts']} entities, {meta['num_queries']}"
    print(f"saved features of {counts} queries to {config['features_dir']}")


def evaluate(config: dict) -> None:
    arrays, meta = load_features(Path(config["features_dir"]))
    queries_sha256 = file_sha256(config["queries_file"])
    if meta["queries_sha256"] != queries_sha256:
        raise ValueError("features were built from another queries file; run the features step")
    _, triples, _ = load_mmkg(Path(config["mmkg_dir"]))
    queries = read_queries(Path(config["queries_file"]))

    for experiment in config["experiments"]:
        for seed in config["seeds"]:
            if test_run_exists(config["output_dir"], experiment["name"], seed):
                print(f"skip {experiment['name']} seed {seed}: already run (evaluated once)")
                continue
            metrics = run_transfer(config, experiment, seed, arrays, queries, triples)
            # The query file plays the role of the split file (ADR-0009, ADR-0010):
            # every image of MMKG is an evaluation image.
            metrics["split"] = "test"
            metrics["split_file"] = config["queries_file"]
            metrics["split_sha256"] = queries_sha256
            run_config = {**config, "experiment": experiment["name"], "run": experiment}
            run_dir = make_run_dir(config["output_dir"], experiment["name"], seed, datetime.now())
            write_run_files(run_dir, run_config, metrics, collect_env(library_versions()))
            print(
                f"{experiment['name']} seed {seed}: alpha {metrics['alpha']}, "
                f"R@1 {metrics['R@1']:.4f}, MRR {metrics['MRR@100']:.4f} -> {run_dir}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("step", choices=["features", "evaluate"])
    parser.add_argument("--config", required=True, help="YAML file in configs/")
    args = parser.parse_args(argv)
    with open(args.config, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if args.step == "features":
        features(config)
    else:
        evaluate(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
