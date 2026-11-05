"""Summary table: mean +- sample std over seeds, from traceable runs only (ADR-0010, -0011)."""

import csv
import json
from pathlib import Path

import pytest
import yaml

from mmxai.eval.summary import load_run_records, summarize_runs, write_table


def write_run(
    runs_dir: Path,
    name: str,
    seed: int,
    r1: float,
    split: str = "test",
    dirty: bool = False,
    experiment: str = "clip_vg",
) -> None:
    run_dir = runs_dir / name
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump({"experiment": experiment}))
    metrics = {"R@1": r1, "MRR@100": r1 / 2, "split": split, "seed": seed, "split_sha256": "s"}
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    (run_dir / "env.json").write_text(json.dumps({"git_commit": "c", "git_dirty": dirty}))


def test_mean_and_sample_std_by_hand(tmp_path):
    for seed, r1 in [(0, 0.1), (1, 0.2), (2, 0.3)]:
        write_run(tmp_path, f"run{seed}", seed, r1)

    rows, skipped = summarize_runs(load_run_records(tmp_path), split="test")

    assert skipped == []
    assert rows[0]["R@1_mean"] == pytest.approx(0.2)
    # Sample std (ddof=1) of 0.1, 0.2, 0.3 is 0.1.
    assert rows[0]["R@1_std"] == pytest.approx(0.1)
    assert rows[0]["num_seeds"] == 3


def test_runs_of_another_split_are_ignored(tmp_path):
    for seed in range(3):
        write_run(tmp_path, f"test{seed}", seed, 0.2)
    write_run(tmp_path, "val0", 0, 0.9, split="val")

    rows, _ = summarize_runs(load_run_records(tmp_path), split="test")

    assert rows[0]["R@1_mean"] == pytest.approx(0.2)


def test_runs_with_uncommitted_code_are_ignored(tmp_path):
    for seed in range(3):
        write_run(tmp_path, f"clean{seed}", seed, 0.2)
    write_run(tmp_path, "dirty3", 3, 0.9, dirty=True)

    rows, _ = summarize_runs(load_run_records(tmp_path), split="test")

    assert rows[0]["num_seeds"] == 3


def test_experiment_with_fewer_than_three_seeds_is_skipped(tmp_path):
    write_run(tmp_path, "a0", 0, 0.2)
    write_run(tmp_path, "a1", 1, 0.2)

    rows, skipped = summarize_runs(load_run_records(tmp_path), split="test")

    assert rows == []
    assert skipped == ["clip_vg"]


def test_same_seed_run_twice_on_test_is_an_error(tmp_path):
    # Test runs once per seed (ADR-0009); a second run means someone re-ran after looking.
    for seed in range(3):
        write_run(tmp_path, f"first{seed}", seed, 0.2)
    write_run(tmp_path, "again0", 0, 0.3)

    with pytest.raises(ValueError, match="more than once"):
        summarize_runs(load_run_records(tmp_path), split="test")


def test_written_table_has_one_row_per_experiment(tmp_path):
    for seed in range(3):
        write_run(tmp_path / "runs", f"clip{seed}", seed, 0.2)
        write_run(tmp_path / "runs", f"graph{seed}", seed, 0.3, experiment="clip_graph_vg")
    rows, _ = summarize_runs(load_run_records(tmp_path / "runs"), split="test")

    write_table(rows, tmp_path / "main.csv")

    with open(tmp_path / "main.csv", encoding="utf-8") as file:
        table = list(csv.DictReader(file))
    assert [row["experiment"] for row in table] == ["clip_graph_vg", "clip_vg"]
