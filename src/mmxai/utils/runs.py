"""Write one run folder under results/runs/ (ADR-0010) so every number traces back to its run."""

import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path

import yaml


def make_run_dir(output_dir: str | Path, experiment: str, seed: int, now: datetime) -> Path:
    """Create results/runs/<YYYYMMDD-HHMM>_<experiment>_seed<k>/. Never reuse a folder."""
    name = f"{now:%Y%m%d-%H%M}_{experiment}_seed{seed}"
    run_dir = Path(output_dir) / name
    # exist_ok=False: a second run in the same minute must not overwrite the first one's files.
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def git_state() -> dict:
    """Return the current commit and whether tracked files have uncommitted changes."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    changes = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    # A dirty tree means the result cannot be traced back to the commit alone.
    return {"git_commit": commit, "git_dirty": changes != ""}


def collect_env(library_versions: dict[str, str]) -> dict:
    """Describe where the run happened, for env.json."""
    env = git_state()
    env["python"] = platform.python_version()
    env["platform"] = platform.platform()
    env["libraries"] = library_versions
    return env


def write_run_files(run_dir: Path, config: dict, metrics: dict, env: dict) -> None:
    """Write config.yaml, metrics.json and env.json into the run folder."""
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    with open(run_dir / "env.json", "w", encoding="utf-8") as file:
        json.dump(env, file, indent=2)
