"""The Flickr30k G-Retriever configs change only the dataset (ADR-0056)."""

from pathlib import Path

import pytest
import yaml

from mmxai.eval.graph_rerank import load_graph_rerank_config
from mmxai.eval.pcst_sage_rerank import load_pcst_sage_config

CONFIGS = Path(__file__).resolve().parent.parent / "configs"

# Keys that name the dataset or where its files live; every other key must match VG.
DATASET_KEYS = {
    "experiment",
    "dataset",
    "split_file",
    "features_dir",
    "graph_dir",
    "cache_dir",
    "model_dir",
    "output_dir",
}

PCST = ("clip_pcst_sage_vg", "clip_pcst_sage_flickr30k")
NOGRAPH = ("clip_nograph_vg_lr1e4", "clip_nograph_flickr30k_lr1e4")
PAIRS = [PCST, NOGRAPH]


def read(name: str) -> dict:
    with open(CONFIGS / f"{name}.yaml", encoding="utf-8") as file:
        return yaml.safe_load(file)


@pytest.mark.parametrize(("vg_name", "flickr_name"), PAIRS)
def test_flickr30k_config_differs_from_vg_only_in_the_dataset(vg_name, flickr_name):
    vg = read(vg_name)
    flickr = read(flickr_name)

    # Only the pcst config borrows VG's edges, so only it gains one key.
    extra = {"concept_edges_split_file"} if flickr_name == PCST[1] else set()
    assert set(flickr) - set(vg) == extra
    assert set(vg) - set(flickr) == set()
    for key in set(vg) - DATASET_KEYS:
        assert flickr[key] == vg[key], key


@pytest.mark.parametrize(("vg_name", "flickr_name"), PAIRS)
def test_flickr30k_paths_point_to_flickr30k(vg_name, flickr_name):
    flickr = read(flickr_name)

    for key in ("split_file", "features_dir", "graph_dir", "cache_dir", "model_dir"):
        if key in flickr:
            assert "flickr30k" in flickr[key], key


def test_pcst_config_borrows_the_vg_concept_edges():
    vg = read(PCST[0])
    flickr = read(PCST[1])

    assert flickr["concept_edges_dir"] == vg["concept_edges_dir"]
    assert flickr["concept_edges_split_file"] == "splits/visual_genome/split_seed0.json"


def test_both_configs_load():
    assert load_pcst_sage_config(CONFIGS / "clip_pcst_sage_flickr30k.yaml").dataset == "flickr30k"
    assert (
        load_graph_rerank_config(CONFIGS / "clip_nograph_flickr30k_lr1e4.yaml").dataset
        == "flickr30k"
    )
