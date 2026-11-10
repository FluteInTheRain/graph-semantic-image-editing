"""Training and re-ranking with the one-layer GNN (ADR-0046) on a tiny made-up dataset."""

import json
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("faiss")

from mmxai.features.cache import save_features  # noqa: E402
from mmxai.graph.contains import ContainsGraph, save_contains_graph  # noqa: E402
from mmxai.models.one_layer_sage import OneLayerSage  # noqa: E402
from mmxai.models.training import (  # noqa: E402
    GnnConfig,
    embed_images,
    load_bipartite,
    sample_hard_negatives,
    texts_of_image_matrix,
    train_gnn,
)
from mmxai.rerank.evaluate import run_rerank_evaluation  # noqa: E402

DIM = 8
CPU = torch.device("cpu")


def make_dataset(tmp_path):
    """6 train, 3 val, 3 test images; 2 texts each; 4 concepts, 2 edges per image."""
    rng = np.random.default_rng(0)
    image_ids = np.arange(100, 112)
    split = np.array(["train"] * 6 + ["val"] * 3 + ["test"] * 3)
    image_x = rng.normal(size=(12, DIM)).astype(np.float32)
    image_x /= np.linalg.norm(image_x, axis=1, keepdims=True)
    concept_x = rng.normal(size=(4, DIM)).astype(np.float32)
    concept_x /= np.linalg.norm(concept_x, axis=1, keepdims=True)
    # Each text is its image's vector plus a little noise, so CLIP mostly finds it.
    text_image_ids = np.repeat(image_ids, 2)
    text_x = image_x[np.repeat(np.arange(12), 2)] + 0.1 * rng.normal(size=(24, DIM))
    text_x = (text_x / np.linalg.norm(text_x, axis=1, keepdims=True)).astype(np.float32)

    features_dir = tmp_path / "features"
    meta = {
        "dataset": "visual_genome",
        "model": "ViT-B-32",
        "pretrained": "laion2b_s34b_b79k",
        "is_partial": False,
        "code_hash": "test",
    }
    save_features(
        features_dir, image_x, image_ids, text_x, text_image_ids, "text_ids", np.arange(24), meta
    )

    graph = ContainsGraph(
        image_ids=image_ids,
        image_split=split,
        concept_keys=np.array(["a.n.01", "b.n.01", "c.n.01", "d.n.01"]),
        edge_image=np.repeat(np.arange(12), 2),
        edge_concept=np.array([[i % 4, (i + 1) % 4] for i in range(12)]).ravel(),
    )
    graph_dir = tmp_path / "graph"
    graph_meta = {
        "config": {"model": "ViT-B-32", "pretrained": "laion2b_s34b_b79k"},
        "split_sha256": "test",
    }
    save_contains_graph(graph_dir, graph, np.array(["a", "b", "c", "d"]), concept_x, graph_meta)

    return GnnConfig(
        experiment="gnn_test",
        dataset="visual_genome",
        features_dir=str(features_dir),
        graph_dir=str(graph_dir),
        top_k=3,
        n_hard=2,
        temperature=0.05,
        learning_rate=0.01,
        batch_size=4,
        max_epochs=2,
        patience=1,
        val_max_queries=6,
        same_text_threshold=0.9999,
        recall_ks=[1, 3],
        alpha_grid=[0.0, 0.5, 1.0],
        alpha=None,
        device="cpu",
        checkpoint_dir=str(tmp_path / "checkpoints"),
        output_dir=str(tmp_path / "runs"),
    )


def test_training_graph_has_no_val_or_test_node(tmp_path):
    """PDF section 6, ADR-0046: the GNN must not see val/test images while training."""
    graph = load_bipartite(make_dataset(tmp_path), train_only=True, device=CPU)
    assert set(graph.image_split.tolist()) == {"train"}
    assert len(graph.image_x) == 6
    assert int(graph.edge_index[1].max()) < 6


def test_bipartite_edges_go_from_concept_to_image(tmp_path):
    graph = load_bipartite(make_dataset(tmp_path), train_only=False, device=CPU)
    assert graph.edge_index.shape == (2, 24)
    assert int(graph.edge_index[0].max()) < 4  # concept rows
    assert int(graph.edge_index[1].max()) < 12  # image rows


def test_texts_of_image_matrix_pads_with_minus_one():
    matrix = texts_of_image_matrix(np.array([1, 0, 1, 1]), 3)
    assert matrix.tolist() == [[1, -1, -1], [0, 2, 3], [-1, -1, -1]]


def test_embed_images_on_a_subset_equals_the_whole_graph(tmp_path):
    graph = load_bipartite(make_dataset(tmp_path), train_only=False, device=CPU)
    model = OneLayerSage(DIM)
    with torch.no_grad():
        model.conv.lin_l.weight.normal_()
    full = model(graph.concept_x, graph.image_x, graph.edge_index)
    rows = torch.tensor([7, 2])
    assert torch.allclose(embed_images(model, graph, rows), full[rows], atol=1e-6)


def test_hard_negatives_skip_the_answer_and_same_sentence_images():
    image_x = torch.eye(4)
    text_x = torch.eye(4)
    # Image 1 owns the same sentence as the query (text 1 = query), so it is relevant too.
    texts_of_image = torch.tensor([[0], [1], [2], [3]])
    query = torch.tensor([[1.0, 1.0, 0.5, 0.0]])
    query = query / query.norm()
    text_x[1] = query[0]
    generator = torch.Generator().manual_seed(0)
    rows, valid = sample_hard_negatives(
        query, torch.tensor([0]), image_x, text_x, texts_of_image, 4, 3, 0.9999, generator
    )
    chosen = set(rows[valid].tolist())
    assert chosen == {2, 3}
    assert int(valid.sum()) == 2


def test_train_then_rerank_val_and_test(tmp_path):
    config = make_dataset(tmp_path)
    checkpoint = train_gnn(config, seed=0)
    meta = json.loads((checkpoint / "meta.json").read_text())
    assert meta["history"][0]["epoch"] == 0

    val_run = run_rerank_evaluation(config, seed=0, split_name="val")
    val_metrics = json.loads((val_run / "metrics.json").read_text())
    assert val_metrics["chosen_alpha"] in config.alpha_grid
    scores = np.load(val_run / "scores.npz")
    assert scores["candidate_image_ids"].shape == (6, 3)
    assert set(scores["candidate_image_ids"].ravel().tolist()) <= {106, 107, 108}

    with pytest.raises(ValueError, match="alpha"):
        run_rerank_evaluation(config, seed=0, split_name="test")
    test_config = replace(config, alpha=val_metrics["chosen_alpha"])
    test_run = run_rerank_evaluation(test_config, seed=0, split_name="test")
    assert "metrics" in json.loads((test_run / "metrics.json").read_text())


def test_training_refuses_to_overwrite_a_checkpoint(tmp_path):
    config = make_dataset(tmp_path)
    train_gnn(config, seed=0)
    with pytest.raises(FileExistsError):
        train_gnn(config, seed=0)
