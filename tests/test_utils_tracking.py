"""Optional W&B logging must never stop or change a run (ADR-0047)."""

from mmxai.utils.tracking import NoRun, start_run


def test_no_project_means_no_logging():
    assert isinstance(start_run(None, "run", {}), NoRun)


def test_wandb_mode_disabled_turns_logging_off(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    assert isinstance(start_run("mmxai-gnn", "run", {}), NoRun)


def test_no_run_accepts_every_call():
    run = NoRun()
    run.log({"train/loss": 1.0})
    run.summary["best_epoch"] = 3
    run.finish()
