"""Optional Weights & Biases logging of training (ADR-0047). Never stops a run.

Logging is off when the config has no wandb_project, when WANDB_MODE=disabled, when
wandb is not installed (`--group tracking`) or when it cannot reach the server: the
run then trains exactly the same and only prints a warning.
"""

import os


class NoRun:
    """Stands in for a W&B run when logging is off, so training code never checks."""

    def __init__(self) -> None:
        self.summary: dict = {}

    def log(self, *args, **kwargs) -> None:
        pass

    def finish(self, *args, **kwargs) -> None:
        pass


def start_run(project: str | None, name: str, config: dict):
    """Return a W&B run, or a NoRun when logging is off or fails."""
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
            config=config,
            settings=wandb.Settings(init_timeout=120),
        )
    except Exception as error:
        # Decision (ADR-0047): a logging problem must never cost a training run.
        print(f"W&B logging off: {error}")
        return NoRun()
