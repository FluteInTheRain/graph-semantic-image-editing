"""Make, save and load the fixed train/val/test split of images (ADR-0009, ADR-0042)."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mmxai.utils.yaml_config import load_yaml_dataclass

SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class SplitConfig:
    """Settings for making one split file."""

    dataset: str
    features_dir: str
    output_dir: str
    split_seed: int
    num_val: int
    num_test: int
    duplicate_threshold: float


def load_split_config(path: str | Path) -> SplitConfig:
    """Load the split config from YAML (ADR-0007)."""
    return load_yaml_dataclass(path, SplitConfig)


def group_near_duplicate_images(
    embeddings: np.ndarray, image_ids: np.ndarray, threshold: float, chunk_size: int = 512
) -> list[list[int]]:
    """Group images whose embeddings have cosine >= threshold, following chains.

    A near-identical embedding almost surely means the same picture stored twice,
    possibly resized or re-compressed. Keeping such copies in one set stops a test
    image from also being seen in train. If A is close to B and B to C, all three
    form one group. Grouping too much is harmless: it only keeps images together.
    Groups are sorted by their smallest id, so the result never depends on row order.
    """
    vectors = np.asarray(embeddings, dtype=np.float32)
    num_images = len(vectors)

    # Union-find: parent[row] points towards the row that represents its group.
    parent = list(range(num_images))

    def find_root(row: int) -> int:
        while parent[row] != row:
            row = parent[row]
        return row

    # Chunks keep memory at chunk_size x num_images instead of num_images x num_images.
    for start in range(0, num_images, chunk_size):
        similarities = vectors[start : start + chunk_size] @ vectors.T
        rows, columns = np.nonzero(similarities >= threshold)
        for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
            root_a = find_root(start + row)
            root_b = find_root(column)
            if root_a != root_b:
                parent[root_b] = root_a

    ids_by_root: dict[int, list[int]] = {}
    for row in range(num_images):
        ids_by_root.setdefault(find_root(row), []).append(int(image_ids[row]))

    groups = [sorted(ids) for ids in ids_by_root.values()]
    groups.sort(key=lambda ids: ids[0])
    return groups


def assign_groups(
    groups: list[list[int]], num_val: int, num_test: int, seed: int
) -> dict[str, list[int]]:
    """Shuffle the groups, then fill test, then val, and put the rest in train.

    A group goes into test or val only if the whole group still fits, so a group
    is never split and both sets reach their exact size whenever possible.
    """
    num_images = sum(len(group) for group in groups)
    if num_val + num_test > num_images:
        raise ValueError(
            f"asked for {num_val} val + {num_test} test images, "
            f"but there are only {num_images} images"
        )

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))

    assigned: dict[str, list[int]] = {name: [] for name in SPLIT_NAMES}
    for index in order.tolist():
        group = groups[index]
        if len(assigned["test"]) + len(group) <= num_test:
            assigned["test"].extend(group)
        elif len(assigned["val"]) + len(group) <= num_val:
            assigned["val"].extend(group)
        else:
            assigned["train"].extend(group)
    return assigned


def make_split(
    groups: list[list[int]], dataset: str, num_val: int, num_test: int, seed: int
) -> dict:
    """Return the split in the file format of splits/README.md: sorted string ids.

    Takes the groups from group_near_duplicate_images, which is slow on the full
    dataset, so the caller computes them once.
    """
    num_images = sum(len(group) for group in groups)
    assigned = assign_groups(groups, num_val, num_test, seed)

    # Later code filters by these lists, so a missing image or a short test set
    # must stop here instead of silently changing what gets evaluated.
    if len(assigned["test"]) != num_test or len(assigned["val"]) != num_val:
        raise ValueError(
            f"got {len(assigned['test'])} test and {len(assigned['val'])} val images, "
            f"expected {num_test} and {num_val}"
        )
    if sum(len(assigned[name]) for name in SPLIT_NAMES) != num_images:
        raise ValueError("the split does not cover every image exactly once")

    split = {"dataset": dataset, "seed": seed}
    for name in SPLIT_NAMES:
        # Strings, because the README fixes ids as strings; numeric order keeps git diffs readable.
        split[name] = [str(image_id) for image_id in sorted(assigned[name])]
    check_split(split)
    return split


def check_split(split: dict) -> None:
    """Raise ValueError if a set is missing, repeats an id, or shares ids with another set."""
    for name in SPLIT_NAMES:
        if name not in split:
            raise ValueError(f"split has no '{name}' set")
        if len(set(split[name])) != len(split[name]):
            raise ValueError(f"'{name}' has a repeated id")

    for first_index, first in enumerate(SPLIT_NAMES):
        for second in SPLIT_NAMES[first_index + 1 :]:
            shared = set(split[first]) & set(split[second])
            if shared:
                example = sorted(shared)[0]
                raise ValueError(f"'{first}' and '{second}' overlap, for example id {example}")


def split_path(output_dir: str | Path, seed: int) -> Path:
    """Return splits/<dataset>/split_seed<k>.json as fixed by splits/README.md."""
    return Path(output_dir) / f"split_seed{seed}.json"


def save_split(split: dict, path: str | Path) -> None:
    """Write a split file. A committed split is frozen, so never overwrite one."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists; a new split needs a new seed and an ADR")
    check_split(split)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(split, file, indent=1)
        file.write("\n")


def load_split(path: str | Path) -> dict[str, np.ndarray]:
    """Return the image ids of each set as int64 arrays, ready to compare with .npz image_ids."""
    with open(path, encoding="utf-8") as file:
        split = json.load(file)
    check_split(split)

    ids_by_set = {}
    for name in SPLIT_NAMES:
        ids = [int(image_id) for image_id in split[name]]
        ids_by_set[name] = np.array(ids, dtype=np.int64)
    return ids_by_set
