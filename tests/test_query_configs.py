"""The val pilot must test exactly the rules later used on test (ADR-0044)."""

from pathlib import Path

from mmxai.queries.generator import load_query_benchmark_config

REPO_ROOT = Path(__file__).parent.parent

# Only these may differ: which split, how many pairs, where files go.
ALLOWED_TO_DIFFER = {
    "split",
    "limit_pairs",
    "candidates_path",
    "manifest_path",
    "benchmark_path",
}


def test_pilot_and_test_configs_share_every_rule():
    test_config = load_query_benchmark_config(REPO_ROOT / "configs/query_benchmark_vg.yaml")
    pilot_config = load_query_benchmark_config(REPO_ROOT / "configs/query_benchmark_vg_pilot.yaml")

    for field in test_config.__dataclass_fields__:
        if field in ALLOWED_TO_DIFFER:
            continue
        assert getattr(test_config, field) == getattr(pilot_config, field), field


def test_kg_pilot_and_test_configs_share_every_rule():
    from mmxai.queries.kg_generator import load_kg_query_config

    test_config = load_kg_query_config(REPO_ROOT / "configs/query_benchmark_vg_kg.yaml")
    pilot_config = load_kg_query_config(REPO_ROOT / "configs/query_benchmark_vg_kg_pilot.yaml")

    allowed = {"split", "candidates_path", "manifest_path", "benchmark_path"}
    for field in test_config.__dataclass_fields__:
        if field in allowed:
            continue
        assert getattr(test_config, field) == getattr(pilot_config, field), field


def test_kg_config_rejects_play_predicates_outside_activity_predicates(tmp_path):
    import pytest

    from mmxai.queries.kg_generator import load_kg_query_config

    text = (REPO_ROOT / "configs/query_benchmark_vg_kg.yaml").read_text(encoding="utf-8")
    bad = text.replace("play_predicates: [playing]", "play_predicates: [playing, cooking]")
    path = tmp_path / "bad.yaml"
    path.write_text(bad, encoding="utf-8")

    with pytest.raises(ValueError, match="cooking"):
        load_kg_query_config(path)
