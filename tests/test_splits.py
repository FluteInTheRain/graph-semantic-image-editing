"""Tests for the fixed train/val/test split (ADR-0009, ADR-0042). Numpy only."""

import json
from pathlib import Path

import numpy as np
import pytest

from mmxai.data.splits import (
    SPLIT_NAMES,
    assign_groups,
    check_split,
    group_duplicate_images,
    load_split,
    load_split_config,
    make_split,
    save_split,
    split_path,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def make_embeddings(num_images: int) -> np.ndarray:
    """Distinct unit vectors, one per image."""
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(num_images, 8)).astype(np.float32)
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)


def make_small_split() -> dict:
    embeddings = make_embeddings(20)
    image_ids = np.arange(100, 120)
    return make_split(embeddings, image_ids, "visual_genome", num_val=4, num_test=5, seed=0)


def all_ids(split: dict) -> list[str]:
    return split["train"] + split["val"] + split["test"]


def test_duplicate_embeddings_form_one_group():
    embeddings = make_embeddings(4)
    embeddings[3] = embeddings[1]
    image_ids = np.array([40, 10, 30, 20])

    groups = group_duplicate_images(embeddings, image_ids)

    assert groups == [[10, 20], [30], [40]]


def test_distinct_embeddings_give_single_image_groups():
    groups = group_duplicate_images(make_embeddings(3), np.array([3, 1, 2]))

    assert groups == [[1], [2], [3]]


def test_assign_groups_fills_test_and_val_to_exact_size():
    groups = [[image_id] for image_id in range(20)]

    assigned = assign_groups(groups, num_val=4, num_test=5, seed=0)

    assert len(assigned["test"]) == 5
    assert len(assigned["val"]) == 4
    assert len(assigned["train"]) == 11


def test_assign_groups_keeps_a_group_together():
    groups = [[0, 1, 2]] + [[image_id] for image_id in range(3, 20)]

    for seed in range(10):
        assigned = assign_groups(groups, num_val=4, num_test=5, seed=seed)
        holders = [name for name in SPLIT_NAMES if 0 in assigned[name]]
        assert len(holders) == 1
        assert {1, 2} <= set(assigned[holders[0]])


def test_assign_groups_rejects_sizes_larger_than_the_data():
    groups = [[image_id] for image_id in range(5)]

    with pytest.raises(ValueError, match="only 5 images"):
        assign_groups(groups, num_val=3, num_test=3, seed=0)


def test_split_covers_every_image_exactly_once():
    split = make_small_split()

    assert sorted(all_ids(split), key=int) == [str(image_id) for image_id in range(100, 120)]


def test_split_sets_do_not_overlap():
    split = make_small_split()

    assert len(set(all_ids(split))) == len(all_ids(split))


def test_split_ids_are_sorted_strings():
    split = make_small_split()

    for name in SPLIT_NAMES:
        assert all(isinstance(image_id, str) for image_id in split[name])
        assert split[name] == sorted(split[name], key=int)


def test_same_seed_gives_same_split():
    assert make_small_split() == make_small_split()


def test_split_does_not_depend_on_row_order():
    embeddings = make_embeddings(20)
    embeddings[7] = embeddings[2]
    image_ids = np.arange(100, 120)
    shuffled_rows = np.random.default_rng(5).permutation(20)

    original = make_split(embeddings, image_ids, "visual_genome", num_val=4, num_test=5, seed=0)
    shuffled = make_split(
        embeddings[shuffled_rows],
        image_ids[shuffled_rows],
        "visual_genome",
        num_val=4,
        num_test=5,
        seed=0,
    )

    assert original == shuffled


def test_make_split_rejects_repeated_image_id():
    embeddings = make_embeddings(20)
    image_ids = np.arange(100, 120)
    image_ids[3] = image_ids[4]

    with pytest.raises(ValueError):
        make_split(embeddings, image_ids, "visual_genome", num_val=4, num_test=5, seed=0)


def test_different_seed_gives_different_test_set():
    embeddings = make_embeddings(20)
    image_ids = np.arange(100, 120)

    first = make_split(embeddings, image_ids, "visual_genome", num_val=4, num_test=5, seed=0)
    second = make_split(embeddings, image_ids, "visual_genome", num_val=4, num_test=5, seed=1)

    assert first["test"] != second["test"]


def test_duplicate_images_land_in_the_same_set():
    embeddings = make_embeddings(20)
    embeddings[7] = embeddings[2]
    image_ids = np.arange(100, 120)

    for seed in range(10):
        split = make_split(embeddings, image_ids, "visual_genome", num_val=4, num_test=5, seed=seed)
        holders = [name for name in SPLIT_NAMES if "102" in split[name]]
        assert "107" in split[holders[0]]


def test_check_split_rejects_overlapping_sets():
    split = {
        "dataset": "visual_genome",
        "seed": 0,
        "train": ["1", "2"],
        "val": ["2"],
        "test": ["3"],
    }

    with pytest.raises(ValueError, match="overlap"):
        check_split(split)


def test_check_split_rejects_repeated_id_inside_one_set():
    split = {
        "dataset": "visual_genome",
        "seed": 0,
        "train": ["1", "1"],
        "val": ["2"],
        "test": ["3"],
    }

    with pytest.raises(ValueError, match="repeated"):
        check_split(split)


def test_check_split_rejects_missing_set():
    split = {"dataset": "visual_genome", "seed": 0, "train": ["1"], "val": ["2"]}

    with pytest.raises(ValueError, match="test"):
        check_split(split)


def test_save_and_load_round_trip(tmp_path):
    split = make_small_split()
    path = tmp_path / "visual_genome" / "split_seed0.json"

    save_split(split, path)
    loaded = load_split(path)

    assert loaded["test"].dtype == np.int64
    assert loaded["test"].tolist() == [int(image_id) for image_id in split["test"]]


def test_save_refuses_to_overwrite(tmp_path):
    split = make_small_split()
    path = tmp_path / "split_seed0.json"
    save_split(split, path)

    with pytest.raises(FileExistsError):
        save_split(split, path)


def test_split_path_follows_readme_format():
    assert split_path("splits/visual_genome", 0) == Path("splits/visual_genome/split_seed0.json")


def test_load_split_config_rejects_unknown_key(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "dataset: visual_genome\nfeatures_dir: f\noutput_dir: o\n"
        "split_seed: 0\nnum_val: 1\nnum_test: 1\nnum_tset: 1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="num_tset"):
        load_split_config(path)


def test_committed_config_loads():
    config = load_split_config(REPO_ROOT / "configs" / "splits_visual_genome.yaml")

    assert config.num_test == 5000
    assert config.num_val == 5000


def test_every_committed_split_file_is_valid():
    # Guards the files in splits/ themselves, not only the code that wrote them (ADR-0009).
    for path in sorted((REPO_ROOT / "splits").glob("*/split_seed*.json")):
        with open(path, encoding="utf-8") as file:
            check_split(json.load(file))
