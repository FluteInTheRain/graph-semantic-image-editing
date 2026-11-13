"""Turn run folders into the committed summary table (ADR-0010, ADR-0011).

Report numbers come only from this table, so it refuses runs that cannot be
traced back to a commit and test runs that were repeated for the same seed.
"""

import csv
import json
import statistics
from pathlib import Path

import yaml

# Decision (ADR-0011): an experiment enters the table only with >= 3 seeds.
#   Why: PDF section 6 requires at least 3 seeds and mean +- std.
#   Decided by: FluteInTheRain, ADR-0011.
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
    """The reported metrics, e.g. R@1, R@5, R@10, MRR@100, nDCG@10."""
    prefixes = ("R@", "MRR@", "nDCG@")
    return [name for name in metrics if name.startswith(prefixes)]


def _eligible_experiments(
    records: list[dict], split: str, experiment_suffix: str | None
) -> tuple[dict[str, list[dict]], list[str]]:
    """Group usable runs by experiment; return (experiments with enough seeds, skipped ones).

    Shared by every table so they all follow the same rules (ADR-0009, -0010, -0011).
    """
    by_experiment: dict[str, list[dict]] = {}
    for record in records:
        if record["metrics"]["split"] != split:
            continue
        if experiment_suffix is not None and not record["experiment"].endswith(experiment_suffix):
            continue
        # Decision (ADR-0010): skip runs made with uncommitted code.
        #   Why: they cannot be reproduced from their commit.
        #   Decided by: FluteInTheRain, 2026-11-05 (proposed by Claude Code, applying ADR-0010).
        if record["git_dirty"]:
            continue
        by_experiment.setdefault(record["experiment"], []).append(record)

    enough = {}
    skipped = []
    for experiment in sorted(by_experiment):
        runs = by_experiment[experiment]
        seeds = [run["metrics"]["seed"] for run in runs]
        # Decision (ADR-0009): refuse a seed that was run more than once on the same split.
        #   Why: test runs once per seed; a repeat means someone re-ran after seeing the
        #   numbers, so silently picking one would hide it.
        #   Decided by: FluteInTheRain, 2026-11-05 (proposed by Claude Code, applying ADR-0009).
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
        enough[experiment] = runs
    return enough, skipped


def _mean_and_std(row: dict, name: str, values: list[float]) -> None:
    """Add <name>_mean and <name>_std to the row."""
    row[f"{name}_mean"] = statistics.mean(values)
    # Decision (ADR-0011): sample standard deviation (ddof=1).
    #   Why: the seeds are a small sample of possible runs.
    #   Decided by: FluteInTheRain, ADR-0011.
    row[f"{name}_std"] = statistics.stdev(values)


def summarize_runs(
    records: list[dict], split: str, experiment_suffix: str | None = None
) -> tuple[list[dict], list[str]]:
    """Return (one row per experiment, experiments skipped for having too few seeds).

    With experiment_suffix, only experiments whose name ends with it are kept.
    """
    experiments, skipped = _eligible_experiments(records, split, experiment_suffix)

    rows = []
    for experiment, runs in experiments.items():
        seeds = [run["metrics"]["seed"] for run in runs]
        row = {
            "experiment": experiment,
            "split": split,
            "num_seeds": len(runs),
            "seeds": " ".join(str(seed) for seed in sorted(seeds)),
        }
        metric_keys = [set(metric_names(run["metrics"])) for run in runs]
        _require_same_keys(experiment, runs, metric_keys, "metrics")
        for name in metric_names(runs[0]["metrics"]):
            _mean_and_std(row, name, [run["metrics"][name] for run in runs])
        row["split_sha256"] = runs[0]["metrics"]["split_sha256"]
        row["git_commits"] = " ".join(sorted({run["git_commit"] for run in runs}))
        rows.append(row)
    return rows, skipped


def _seed_of(run: dict) -> int:
    return run["metrics"]["seed"]


def _require_same_keys(experiment: str, runs: list[dict], key_sets: list[set], what: str) -> None:
    """Raise if some seed has different keys than the first run, in either direction.

    Why: averaging over fewer seeds for one key would silently weaken the table.
    """
    for run, keys in zip(runs, key_sets, strict=True):
        if keys != key_sets[0]:
            missing = sorted(map(str, key_sets[0] - keys))
            extra = sorted(map(str, keys - key_sets[0]))
            raise ValueError(
                f"{experiment}: {what} differ between seeds; seed {_seed_of(run)} "
                f"(run {run['run']}) is missing {missing} and has extra {extra} "
                f"compared with seed {_seed_of(runs[0])}"
            )


def summarize_groups(
    records: list[dict], split: str, experiment_suffix: str | None = None
) -> tuple[list[dict], list[str]]:
    """Return (one row per experiment, grouping and group, skipped experiments).

    Each row has the mean and std over seeds of every metric of that group.
    """
    experiments, skipped = _eligible_experiments(records, split, experiment_suffix)

    rows = []
    for experiment, runs in experiments.items():
        for run in runs:
            if "groups" not in run["metrics"]:
                raise ValueError(
                    f"{experiment}: seed {_seed_of(run)} (run {run['run']}) has no 'groups' "
                    "in metrics.json; use --experiment-suffix to keep only _eval2 experiments"
                )
        group_keys = []
        for run in runs:
            keys = set()
            for grouping, groups in run["metrics"]["groups"].items():
                for group in groups:
                    keys.add((grouping, group))
            group_keys.append(keys)
        _require_same_keys(experiment, runs, group_keys, "groups")

        first_groups = runs[0]["metrics"]["groups"]
        for grouping in first_groups:
            for group in first_groups[grouping]:
                per_seed = [run["metrics"]["groups"][grouping][group] for run in runs]
                metric_keys = [
                    set(metric_names(m)) | {"num_queries", "fraction_queries"} for m in per_seed
                ]
                _require_same_keys(experiment, runs, metric_keys, f"metrics of {grouping}/{group}")

                row = {"experiment": experiment, "grouping": grouping, "group": group}
                row["num_queries_mean"] = statistics.mean(m["num_queries"] for m in per_seed)
                row["fraction_queries_mean"] = statistics.mean(
                    m["fraction_queries"] for m in per_seed
                )
                for name in metric_names(per_seed[0]):
                    _mean_and_std(row, name, [m[name] for m in per_seed])
                rows.append(row)
    return rows, skipped


def summarize_image_to_text(
    records: list[dict], split: str, experiment_suffix: str | None = None
) -> tuple[list[dict], list[str]]:
    """Return (one row per experiment with image-to-text metrics and alphas, skipped).

    The row has mean and std of each metric and the alpha of each seed.
    """
    experiments, skipped = _eligible_experiments(records, split, experiment_suffix)

    rows = []
    for experiment, runs in experiments.items():
        # Order by seed so the alphas line up with the "seeds" column.
        runs = sorted(runs, key=_seed_of)
        blocks = []
        for run in runs:
            if "image_to_text" not in run["metrics"]:
                raise ValueError(
                    f"{experiment}: seed {_seed_of(run)} (run {run['run']}) has no "
                    "'image_to_text' in metrics.json"
                )
            block = run["metrics"]["image_to_text"]
            if "alpha" not in block:
                raise ValueError(
                    f"{experiment}: seed {_seed_of(run)} (run {run['run']}) has no "
                    "'alpha' in image_to_text"
                )
            blocks.append(block)
        metric_keys = [set(metric_names(block)) for block in blocks]
        _require_same_keys(experiment, runs, metric_keys, "image_to_text metrics")

        row = {
            "experiment": experiment,
            "split": split,
            "num_seeds": len(runs),
            "seeds": " ".join(str(_seed_of(run)) for run in runs),
            "alphas": " ".join(str(block["alpha"]) for block in blocks),
        }
        for name in metric_names(blocks[0]):
            _mean_and_std(row, name, [block[name] for block in blocks])
        rows.append(row)
    return rows, skipped


def write_table(rows: list[dict], path: str | Path) -> None:
    """Write the rows as CSV, numbers rounded to 4 decimals."""
    if not rows:
        raise ValueError("no experiment has enough traceable runs to summarise")
    expected_keys = list(rows[0])
    for row in rows:
        if list(row) != expected_keys:
            raise ValueError("all table rows must have the same columns")
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
