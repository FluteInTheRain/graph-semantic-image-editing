"""Evaluate plain CLIP retrieval. All logic lives in mmxai.eval.clip_baseline."""

import argparse
import json
from dataclasses import replace

from mmxai.eval.clip_baseline import load_clip_eval_config, run_clip_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML file in configs/")
    parser.add_argument("--seed", type=int, required=True, help="run label; CLIP itself is fixed")
    parser.add_argument("--split", choices=["val", "test"], help="override eval_split")
    args = parser.parse_args()

    config = load_clip_eval_config(args.config)
    if args.split is not None:
        config = replace(config, eval_split=args.split)
    run_dir = run_clip_evaluation(config, args.seed)
    with open(run_dir / "metrics.json", encoding="utf-8") as file:
        print(json.dumps(json.load(file), indent=2))
    print(f"saved {run_dir}")


if __name__ == "__main__":
    main()
