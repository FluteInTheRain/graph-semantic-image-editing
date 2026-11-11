import dataclasses

import pytest
import yaml

from mmxai.explain.config import load_explain_config

COMMITTED = "configs/explain_vg.yaml"


def write_config(tmp_path, **changes):
    with open(COMMITTED, encoding="utf-8") as file:
        values = yaml.safe_load(file)
    values.update(changes)
    path = tmp_path / "explain.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    return path


def test_committed_config_loads():
    config = load_explain_config(COMMITTED)

    assert config.eval_split == "val"
    assert len(config.seeds) >= 3


def test_rejects_an_unknown_explainer(tmp_path):
    with pytest.raises(ValueError, match="unknown explainers"):
        load_explain_config(write_config(tmp_path, explainers=["shortest_path"]))


def test_rejects_fewer_than_three_seeds(tmp_path):
    with pytest.raises(ValueError, match="3 seeds"):
        load_explain_config(write_config(tmp_path, seeds=[0, 1]))


def test_rejects_a_probability_above_one(tmp_path):
    with pytest.raises(ValueError, match="robust_remove_prob"):
        load_explain_config(write_config(tmp_path, robust_remove_prob=1.5))


def test_rejects_a_split_other_than_val_or_test(tmp_path):
    with pytest.raises(ValueError, match="val or test"):
        load_explain_config(write_config(tmp_path, eval_split="train"))


def test_config_is_frozen():
    config = load_explain_config(COMMITTED)

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.top_m = 3


def test_test_split_needs_the_choices_made_on_val(tmp_path):
    with pytest.raises(ValueError, match="on val before running test"):
        load_explain_config(write_config(tmp_path, eval_split="test"))


def test_test_split_runs_with_choices_made_on_val(tmp_path):
    config = load_explain_config(
        write_config(tmp_path, eval_split="test", main_explainer="occlusion", top_m=5)
    )

    assert config.main_explainer == "occlusion"


def test_rejects_a_top_m_not_among_the_options(tmp_path):
    with pytest.raises(ValueError, match="top_m_options"):
        load_explain_config(write_config(tmp_path, top_m=7))


def test_rejects_a_main_explainer_not_compared(tmp_path):
    with pytest.raises(ValueError, match="not among explainers"):
        load_explain_config(
            write_config(tmp_path, explainers=["occlusion"], main_explainer="gnn_explainer")
        )
