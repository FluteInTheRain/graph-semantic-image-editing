"""Summary table: mean +- sample std over seeds, from traceable runs only (ADR-0010, -0011)."""

import csv
import json
from pathlib import Path

import pytest
import yaml

from mmxai.eval.summary import (
    load_run_records,
    summarize_groups,
    summarize_image_to_text,
    summarize_runs,
    write_table,
)


def write_run(
    runs_dir: Path,
    name: str,
    seed: int,
    r1: float,
    split: str = "test",
    dirty: bool = False,
    experiment: str = "clip_vg",
    extra_metrics: dict | None = None,
) -> None:
    run_dir = runs_dir / name
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump({"experiment": experiment}))
    metrics = {"R@1": r1, "MRR@100": r1 / 2, "split": split, "seed": seed, "split_sha256": "s"}
    metrics.update(extra_metrics or {})
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


def group_metrics(value: float, fraction: float) -> dict:
    return {"R@1": value, "nDCG@10": value, "num_queries": 10, "fraction_queries": fraction}


def write_eval2_runs(tmp_path: Path, drop_generic_in_seed: int | None = None) -> None:
    for seed, value in [(0, 0.1), (1, 0.2), (2, 0.3)]:
        groups = {
            "specificity": {
                "specific": group_metrics(value, 0.25),
                "generic": group_metrics(value * 2, 0.75),
            }
        }
        if seed == drop_generic_in_seed:
            del groups["specificity"]["generic"]
        extra = {
            "nDCG@10": value,
            "groups": groups,
            "image_to_text": {"R@1": value, "nDCG@10": value, "alpha": 0.5 + seed},
        }
        write_run(tmp_path, f"e{seed}", seed, value, experiment="graph_eval2", extra_metrics=extra)


def test_default_summary_is_unchanged(tmp_path):
    for seed in range(3):
        write_run(tmp_path, f"run{seed}", seed, 0.2, extra_metrics={"groups": {}})

    rows, _ = summarize_runs(load_run_records(tmp_path), split="test")

    assert list(rows[0]) == [
        "experiment",
        "split",
        "num_seeds",
        "seeds",
        "R@1_mean",
        "R@1_std",
        "MRR@100_mean",
        "MRR@100_std",
        "split_sha256",
        "git_commits",
    ]


def test_ndcg_is_summarised_like_the_other_metrics(tmp_path):
    write_eval2_runs(tmp_path)

    rows, _ = summarize_runs(load_run_records(tmp_path), split="test")

    assert rows[0]["nDCG@10_mean"] == pytest.approx(0.2)
    assert rows[0]["nDCG@10_std"] == pytest.approx(0.1)


def test_suffix_keeps_only_eval2_experiments(tmp_path):
    write_eval2_runs(tmp_path)
    for seed in range(3):
        write_run(tmp_path, f"old{seed}", seed, 0.2, experiment="graph")

    rows, _ = summarize_runs(load_run_records(tmp_path), "test", experiment_suffix="_eval2")

    assert [row["experiment"] for row in rows] == ["graph_eval2"]


def test_groups_table_has_one_row_per_group_with_mean_and_std(tmp_path):
    write_eval2_runs(tmp_path)

    rows, _ = summarize_groups(load_run_records(tmp_path), "test", "_eval2")

    assert [(row["grouping"], row["group"]) for row in rows] == [
        ("specificity", "specific"),
        ("specificity", "generic"),
    ]
    generic = rows[1]
    assert generic["experiment"] == "graph_eval2"
    # Values over seeds are 0.2, 0.4, 0.6: mean 0.4, sample std 0.2.
    assert generic["R@1_mean"] == pytest.approx(0.4)
    assert generic["R@1_std"] == pytest.approx(0.2)
    assert generic["num_queries_mean"] == pytest.approx(10)
    assert generic["fraction_queries_mean"] == pytest.approx(0.75)


def test_group_missing_in_one_seed_is_an_error(tmp_path):
    write_eval2_runs(tmp_path, drop_generic_in_seed=1)

    with pytest.raises(ValueError, match="seed 1.*generic"):
        summarize_groups(load_run_records(tmp_path), "test", "_eval2")


def test_image_to_text_table_lists_alpha_of_each_seed(tmp_path):
    write_eval2_runs(tmp_path)

    rows, _ = summarize_image_to_text(load_run_records(tmp_path), "test", "_eval2")

    assert rows[0]["alphas"] == "0.5 1.5 2.5"
    assert rows[0]["seeds"] == "0 1 2"
    assert rows[0]["R@1_mean"] == pytest.approx(0.2)
    assert rows[0]["R@1_std"] == pytest.approx(0.1)
    assert "alpha_mean" not in rows[0]


def rewrite_metrics(tmp_path: Path, run_name: str, change) -> None:
    path = tmp_path / run_name / "metrics.json"
    metrics = json.loads(path.read_text())
    change(metrics)
    path.write_text(json.dumps(metrics))


def test_missing_groups_block_is_a_clear_error(tmp_path):
    write_eval2_runs(tmp_path)
    rewrite_metrics(tmp_path, "e1", lambda metrics: metrics.pop("groups"))

    with pytest.raises(ValueError, match=r"graph_eval2: seed 1 .* no 'groups'"):
        summarize_groups(load_run_records(tmp_path), "test", "_eval2")


def test_group_only_in_a_later_seed_is_an_error(tmp_path):
    write_eval2_runs(tmp_path)

    def add_group(metrics):
        metrics["groups"]["specificity"]["extra"] = group_metrics(0.1, 0.1)

    rewrite_metrics(tmp_path, "e2", add_group)

    with pytest.raises(ValueError, match="seed 2.*extra"):
        summarize_groups(load_run_records(tmp_path), "test", "_eval2")


def test_metric_missing_in_a_later_seed_is_an_error(tmp_path):
    write_eval2_runs(tmp_path)

    def drop_metric(metrics):
        del metrics["groups"]["specificity"]["generic"]["nDCG@10"]

    rewrite_metrics(tmp_path, "e2", drop_metric)

    with pytest.raises(ValueError, match="seed 2.*nDCG@10"):
        summarize_groups(load_run_records(tmp_path), "test", "_eval2")


def test_image_to_text_without_alpha_is_an_error(tmp_path):
    write_eval2_runs(tmp_path)
    rewrite_metrics(tmp_path, "e0", lambda metrics: metrics["image_to_text"].pop("alpha"))

    with pytest.raises(ValueError, match=r"seed 0 .* no 'alpha'"):
        summarize_image_to_text(load_run_records(tmp_path), "test", "_eval2")


def test_under_seeded_eval2_experiment_is_reported_as_skipped(tmp_path):
    write_eval2_runs(tmp_path)
    for seed in range(2):
        write_run(
            tmp_path,
            f"short{seed}",
            seed,
            0.2,
            experiment="short_eval2",
            extra_metrics={"groups": {}, "image_to_text": {"alpha": 1.0}},
        )

    group_rows, group_skipped = summarize_groups(load_run_records(tmp_path), "test", "_eval2")
    _, text_skipped = summarize_image_to_text(load_run_records(tmp_path), "test", "_eval2")

    assert group_skipped == ["short_eval2"]
    assert text_skipped == ["short_eval2"]
    assert {row["experiment"] for row in group_rows} == {"graph_eval2"}


def test_missing_image_to_text_block_is_a_clear_error(tmp_path):
    write_eval2_runs(tmp_path)
    rewrite_metrics(tmp_path, "e1", lambda metrics: metrics.pop("image_to_text"))

    with pytest.raises(ValueError, match=r"graph_eval2: seed 1 .* no 'image_to_text'"):
        summarize_image_to_text(load_run_records(tmp_path), "test", "_eval2")


def test_image_to_text_metric_missing_in_later_seed_is_an_error(tmp_path):
    write_eval2_runs(tmp_path)
    rewrite_metrics(tmp_path, "e2", lambda metrics: metrics["image_to_text"].pop("nDCG@10"))

    with pytest.raises(ValueError, match="seed 2.*nDCG@10"):
        summarize_image_to_text(load_run_records(tmp_path), "test", "_eval2")


def test_dirty_runs_are_skipped_by_groups_and_image_to_text_tables(tmp_path):
    write_eval2_runs(tmp_path)
    for seed in range(3):
        write_run(
            tmp_path,
            f"dirty{seed}",
            seed,
            0.2,
            experiment="dirty_eval2",
            dirty=True,
            extra_metrics={"groups": {}, "image_to_text": {"alpha": 1.0}},
        )

    group_rows, group_skipped = summarize_groups(load_run_records(tmp_path), "test", "_eval2")
    text_rows, text_skipped = summarize_image_to_text(
        load_run_records(tmp_path), "test", "_eval2"
    )

    assert group_skipped == []
    assert text_skipped == []
    assert {row["experiment"] for row in group_rows} == {"graph_eval2"}
    assert {row["experiment"] for row in text_rows} == {"graph_eval2"}
