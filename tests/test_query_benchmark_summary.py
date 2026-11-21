"""summarize_benchmark_runs with fake run folders (no network, no models)."""

import json

import pytest
import yaml

from mmxai.eval.query_benchmark_models import summarize_benchmark_runs


def block(r1: float) -> dict:
    return {"R@1": r1, "R@5": r1, "R@10": r1, "MRR@100": r1, "num_queries": 4}


def write_run(
    runs_dir, experiment: str, seed: int, r1: float, sha: str = "abc", name: str | None = None
) -> None:
    run_dir = runs_dir / (name or f"20261121-0000_{experiment}_seed{seed}")
    run_dir.mkdir()
    metrics = {
        "all": block(r1),
        "direct": block(r1),
        "indirect": None,
        "by_query_group": {"activity": {"direct": block(r1), "indirect": None}},
        "seed": seed,
        "benchmark_sha256": sha,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump({"experiment": experiment}, file)


def test_mean_and_sample_std(tmp_path):
    for seed, value in enumerate([0.2, 0.4, 0.6]):
        write_run(tmp_path, "model_a", seed, value)
    write_run(tmp_path, "model_b", 0, 0.5)

    rows = {row["experiment"]: row for row in summarize_benchmark_runs(tmp_path)}

    a = rows["model_a"]
    assert a["num_seeds"] == 3
    assert a["seeds"] == "0 1 2"
    assert a["num_queries"] == 4
    assert a["all_R@1_mean"] == pytest.approx(0.4)
    assert a["all_R@1_std"] == pytest.approx(0.2)  # ddof=1
    assert a["activity_direct_MRR@100_mean"] == pytest.approx(0.4)
    # A scope that is None in the metrics stays empty.
    assert a["indirect_R@1_mean"] is None
    assert a["activity_indirect_R@1_mean"] is None
    # One seed: mean is kept, std is empty.
    assert rows["model_b"]["all_R@1_mean"] == pytest.approx(0.5)
    assert rows["model_b"]["all_R@1_std"] is None


def test_duplicated_seed_is_refused(tmp_path):
    write_run(tmp_path, "model_a", 0, 0.2)
    write_run(tmp_path, "model_a", 0, 0.3, name="20261121-0001_model_a_seed0")
    with pytest.raises(ValueError, match="duplicated seed"):
        summarize_benchmark_runs(tmp_path)


def test_mixed_benchmark_hashes_are_refused(tmp_path):
    write_run(tmp_path, "model_a", 0, 0.2, sha="one")
    write_run(tmp_path, "model_a", 1, 0.3, sha="two")
    with pytest.raises(ValueError, match="different benchmark"):
        summarize_benchmark_runs(tmp_path)


def test_hashes_must_match_across_experiments(tmp_path):
    write_run(tmp_path, "model_a", 0, 0.2, sha="one")
    write_run(tmp_path, "model_b", 0, 0.3, sha="two")
    with pytest.raises(ValueError, match="different benchmark"):
        summarize_benchmark_runs(tmp_path)
