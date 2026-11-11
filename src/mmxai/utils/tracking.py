"""Optional Weights & Biases logging of training (ADR-0047). Never stops a run.

Logging is off when the config has no wandb_project, when WANDB_MODE=disabled, when
wandb is not installed (`--group tracking`) or when it cannot reach the server: the
run then trains exactly the same and only prints a warning.
"""

import os
from pathlib import Path


class NoRun:
    """Stands in for a W&B run when logging is off, so training code never checks."""

    def __init__(self) -> None:
        self.summary: dict = {}

    def log(self, *args, **kwargs) -> None:
        pass

    def finish(self, *args, **kwargs) -> None:
        pass


def start_run(project: str | None, name: str, config: dict, job_type: str | None = None):
    """Return a W&B run, or a NoRun when logging is off or fails.

    job_type tells runs apart in one project, e.g. "train" and "xai_level1".
    """
    if project is None or os.environ.get("WANDB_MODE") == "disabled":
        return NoRun()
    try:
        import wandb
    except ImportError:
        print("W&B logging off: wandb is not installed (add --group tracking)")
        return NoRun()
    try:
        return wandb.init(
            project=project,
            entity=os.environ.get("WANDB_ENTITY"),
            name=name,
            job_type=job_type,
            config=config,
            settings=wandb.Settings(init_timeout=120),
        )
    except Exception as error:
        # Decision (ADR-0047): a logging problem must never cost a training run.
        print(f"W&B logging off: {error}")
        return NoRun()


def log_table(run, key: str, rows: list[dict]) -> None:
    """Log rows (one dict per row, same keys) as a W&B table; a no-op without a real run."""
    if isinstance(run, NoRun) or not rows:
        return
    import wandb

    columns = list(rows[0])
    table = wandb.Table(columns=columns)
    for row in rows:
        table.add_data(*[row[column] for column in columns])
    run.log({key: table})


def log_files(run, name: str, kind: str, paths: list[str | Path]) -> None:
    """Upload the files and folders that exist as one W&B artifact; a no-op without a run."""
    if isinstance(run, NoRun):
        return
    import wandb

    artifact = wandb.Artifact(name, type=kind)
    for path in map(Path, paths):
        if path.is_dir():
            artifact.add_dir(str(path), name=path.name)
        elif path.exists():
            artifact.add_file(str(path))
    run.log_artifact(artifact)
