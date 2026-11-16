"""The Flickr30k level-1 configs change only the dataset, never a setting (ADR-0050)."""

from pathlib import Path

import pytest
import yaml

CONFIGS = Path(__file__).resolve().parent.parent / "configs"

# Keys that name the dataset or where its files live; every other key must match VG.
DATASET_KEYS = {
    "experiment",
    "dataset",
    "split_file",
    "features_dir",
    "graph_dir",
    "model_dir",
    "output_dir",
}

PAIRS = [
    ("clip_vg", "clip_flickr30k"),
    ("clip_nograph_vg", "clip_nograph_flickr30k"),
    ("clip_graph_vg_lr1e4", "clip_graph_flickr30k_lr1e4"),
    ("clip_graph_vg_lr1e4_shuffled", "clip_graph_flickr30k_lr1e4_shuffled"),
    ("graph_vg_clip_topk_shuffled", "graph_flickr30k_clip_topk_shuffled"),
    ("splits_visual_genome", "splits_flickr30k"),
]


def read(name: str) -> dict:
    with open(CONFIGS / f"{name}.yaml", encoding="utf-8") as file:
        return yaml.safe_load(file)


@pytest.mark.parametrize(("vg_name", "flickr_name"), PAIRS)
def test_flickr30k_config_differs_from_vg_only_in_the_dataset(vg_name, flickr_name):
    vg = read(vg_name)
    flickr = read(flickr_name)

    assert set(vg) == set(flickr)
    for key in set(vg) - DATASET_KEYS:
        assert flickr[key] == vg[key], key


@pytest.mark.parametrize(("vg_name", "flickr_name"), PAIRS)
def test_flickr30k_paths_point_to_flickr30k(vg_name, flickr_name):
    flickr = read(flickr_name)

    for key in ("split_file", "features_dir", "graph_dir", "model_dir"):
        if key in flickr:
            assert "flickr30k" in flickr[key], key


def test_graph_k_matches_the_vg_main_graph():
    assert (
        read("graph_flickr30k_clip_topk")["clip_topk_k"]
        == read("graph_vg_clip_topk")["clip_topk_k"]
    )
