"""End-to-end graph build on tiny synthetic files, with a fake text encoder (ADR-0045).

Made-up data: images 1-2 train, 3 val, 4 test. "zebra" is only in the test image.
The val image's "grass" has no synset and must get grass.n.01 from the train dictionary.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from mmxai.data.splits import save_split
from mmxai.graph.contains import load_contains_graph
from mmxai.graph.pipeline import GraphConfig, load_graph_config, run_build_graph

REPO_ROOT = Path(__file__).resolve().parent.parent


def obj(object_id: int, name: str, synset: str) -> dict:
    synsets = [synset] if synset else []
    return {"object_id": object_id, "names": [name], "synsets": synsets}


def make_config(tmp_path: Path) -> GraphConfig:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    objects = [
        {"image_id": 1, "objects": [obj(1, "dog", "dog.n.01"), obj(2, "grass", "grass.n.01")]},
        {"image_id": 2, "objects": [obj(3, "dog", "dog.n.01")]},
        {"image_id": 3, "objects": [obj(4, "dog", "dog.n.01"), obj(7, "grass", "")]},
        {"image_id": 4, "objects": [obj(5, "dog", "dog.n.01"), obj(6, "zebra", "zebra.n.01")]},
    ]
    (data_dir / "objects.json").write_text(json.dumps(objects), encoding="utf-8")
    split_file = tmp_path / "split.json"
    save_split({"train": ["1", "2"], "val": ["3"], "test": ["4"]}, split_file)
    return GraphConfig(
        dataset="visual_genome",
        data_dir=str(data_dir),
        objects_file="objects.json",
        split_file=str(split_file),
        min_train_images=1,
        concept_prompt="a photo of a {}",
        model="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        device="cpu",
        text_batch_size=8,
        output_dir=str(tmp_path / "graph"),
    )


def fake_encoder(texts: list[str]) -> np.ndarray:
    """One unit vector per text; records what it was asked to encode."""
    fake_encoder.seen = list(texts)
    return np.eye(len(texts), 4, dtype=np.float32)


def test_build_writes_a_graph_of_every_split_image(tmp_path):
    config = make_config(tmp_path)

    run_build_graph(config, fake_encoder)
    graph, names, features, _ = load_contains_graph(config.output_dir)

    assert graph.image_ids.tolist() == [1, 2, 3, 4]
    assert names.tolist() == ["dog", "grass"]
    assert features.shape == (2, 4)


def test_build_never_adds_a_concept_seen_only_in_test(tmp_path):
    config = make_config(tmp_path)

    run_build_graph(config, fake_encoder)
    graph, _, _, _ = load_contains_graph(config.output_dir)

    assert "zebra.n.01" not in graph.concept_keys.tolist()


def test_concepts_are_encoded_with_the_prompt(tmp_path):
    config = make_config(tmp_path)

    run_build_graph(config, fake_encoder)

    assert fake_encoder.seen == ["a photo of a dog", "a photo of a grass"]


def test_meta_records_inputs_and_counts(tmp_path):
    config = make_config(tmp_path)

    run_build_graph(config, fake_encoder)
    _, _, _, meta = load_contains_graph(config.output_dir)

    assert meta["num_images"] == 4
    assert meta["num_concepts"] == 2
    assert meta["num_edges"] == 6
    assert meta["images_without_edges"] == {"train": 0, "val": 0, "test": 0}
    assert len(meta["split_sha256"]) == 64
    assert len(meta["objects_sha256"]) == 64


def test_object_without_synset_gets_an_edge_through_the_train_dictionary(tmp_path):
    config = make_config(tmp_path)

    meta = run_build_graph(config, fake_encoder)
    graph, _, _, _ = load_contains_graph(config.output_dir)

    image_row = graph.image_ids.tolist().index(3)
    concepts = set()
    for edge in range(len(graph.edge_image)):
        if graph.edge_image[edge] == image_row:
            concepts.add(str(graph.concept_keys[graph.edge_concept[edge]]))
    assert concepts == {"dog.n.01", "grass.n.01"}
    assert meta["synset_sources"] == {"annotated": 6, "from_name": 1, "unmapped": 0}


def test_missing_objects_file_explains_where_to_get_it(tmp_path):
    config = make_config(tmp_path)
    (Path(config.data_dir) / "objects.json").unlink()

    with pytest.raises(FileNotFoundError, match="objects.json"):
        run_build_graph(config, fake_encoder)


def test_committed_config_loads():
    config = load_graph_config(REPO_ROOT / "configs" / "graph_vg.yaml")

    assert config.dataset == "visual_genome"
    assert "{}" in config.concept_prompt
