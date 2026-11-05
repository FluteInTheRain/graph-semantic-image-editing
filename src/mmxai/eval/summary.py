"""Turn run folders into the committed summary table (ADR-0010, ADR-0011).

Report numbers come only from this table, so it refuses runs that cannot be
traced back to a commit and test runs that were repeated for the same seed.
"""

import csv
import json
import statistics
from pathlib import Path

import yaml

# Decision (PDF section 6, ADR-0011): an experiment enters the table only with >= 3 seeds.
MIN_SEEDS = 3


def load_run_records(runs_dir: str | Path) -> list[dict]:
    """Read config.yaml, metrics.json and env.json of every run folder."""
    records = []
    for run_dir in sorted(Path(runs_dir).iterdir()):
        if not (run_dir / "metrics.json").exists():
            continue
        with open(run_dir / "config.yaml", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        with open(run_dir / "metrics.json", encoding="utf-8") as file:
            metrics = json.load(file)
        with open(run_dir / "env.json", encoding="utf-8") as file:
            env = json.load(file)
        records.append(
            {
                "run": run_dir.name,
                "experiment": config["experiment"],
                "metrics": metrics,
                "git_commit": env["git_commit"],
                "git_dirty": env["git_dirty"],
            }
        )
    return records


def metric_names(metrics: dict) -> list[str]:
    """The reported metrics, e.g. R@1, R@5, R@10, MRR@100."""
    return [name for name in metrics if name.startswith("R@") or name.startswith("MRR@")]


def summarize_runs(records: list[dict], split: str) -> tuple[list[dict], list[str]]:
    """Return (one row per experiment, experiments skipped for having too few seeds)."""
    by_experiment: dict[str, list[dict]] = {}
    for record in records:
        if record["metrics"]["split"] != split:
            continue
        # Decision (ADR-0010): skip runs made with uncommitted code, because they
        # cannot be reproduced from their commit.
        if record["git_dirty"]:
            continue
        by_experiment.setdefault(record["experiment"], []).append(record)

    rows = []
    skipped = []
    for experiment in sorted(by_experiment):
        runs = by_experiment[experiment]
        seeds = [run["metrics"]["seed"] for run in runs]
        # Decision (ADR-0009): test runs once per seed; a repeat means someone re-ran
        # after seeing the numbers, so refuse instead of silently picking one.
        if len(set(seeds)) != len(seeds):
            raise ValueError(
                f"{experiment}: a seed was run more than once on {split}; "
                "keep only the first run of each seed"
            )
        if len(runs) < MIN_SEEDS:
            skipped.append(experiment)
            continue

        split_hashes = {run["metrics"]["split_sha256"] for run in runs}
        if len(split_hashes) != 1:
            raise ValueError(f"{experiment}: runs used different split files")

        row = {
            "experiment": experiment,
            "split": split,
            "num_seeds": len(runs),
            "seeds": " ".join(str(seed) for seed in sorted(seeds)),
        }
        for name in metric_names(runs[0]["metrics"]):
            values = [run["metrics"][name] for run in runs]
            row[f"{name}_mean"] = statistics.mean(values)
            # Decision (ADR-0011): sample standard deviation (ddof=1), because the seeds
            # are a small sample of possible runs.
            row[f"{name}_std"] = statistics.stdev(values)
        row["split_sha256"] = split_hashes.pop()
        row["git_commits"] = " ".join(sorted({run["git_commit"] for run in runs}))
        rows.append(row)
    return rows, skipped


def write_table(rows: list[dict], path: str | Path) -> None:
    """Write the rows as CSV, numbers rounded to 4 decimals."""
    if not rows:
        raise ValueError("no experiment has enough traceable runs to summarise")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            rounded = {}
            for key, value in row.items():
                if isinstance(value, float):
                    rounded[key] = round(value, 4)
                else:
                    rounded[key] = value
            writer.writerow(rounded)
