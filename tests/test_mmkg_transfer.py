import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from mmxai.eval.mmkg_transfer import (  # noqa: E402
    build_transfer_graph,
    clip_candidates,
    contains_edge_index,
    group_metrics,
    image_graph_vectors,
    transfer_ranks,
    triple_edge_index,
)
from mmxai.models.sage import ConceptToImageSage, CooccurrenceSage  # noqa: E402

DIM = 4
CONCEPTS = ["drum", "bronze", "pagoda"]
TRIPLES = [{"head_id": "drum", "relation": "made_from", "tail_id": "bronze"}]


def unit(rows: list[list[float]]) -> np.ndarray:
    array = np.array(rows, dtype=np.float32)
    return array / np.linalg.norm(array, axis=1, keepdims=True)


IMAGE_IDS = np.array([0, 1, 2], dtype=np.int64)
IMAGES = unit([[1, 0, 0, 0], [0.9, 0.1, 0, 0], [0, 0, 1, 0]])
CONCEPT_VECTORS = unit([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.1]])


def test_contains_edges_link_each_image_to_its_closest_concepts():
    edges = contains_edge_index(IMAGE_IDS, IMAGES, CONCEPTS, CONCEPT_VECTORS, k=1)

    num_images = len(IMAGE_IDS)
    pairs = {(CONCEPTS[source - num_images], target) for source, target in edges.T.tolist()}
    assert pairs == {("drum", 0), ("drum", 1), ("pagoda", 2)}


def test_triples_become_concept_edges_in_both_directions():
    edges = triple_edge_index(TRIPLES, CONCEPTS)

    assert sorted(map(tuple, edges.T.tolist())) == [(0, 1), (1, 0)]


def test_candidates_are_sorted_by_clip_score_and_keep_the_given_answers():
    queries = unit([[1, 0, 0, 0]])

    candidates = clip_candidates(queries, IMAGES, IMAGE_IDS, [{1}], top_k=100)

    assert candidates.candidate_ids.tolist() == [[0, 1, 2]]
    assert candidates.relevant == [{1}]
    assert np.all(np.diff(candidates.clip_scores[0]) <= 0)


def test_alpha_one_gives_exactly_the_clip_ranks():
    queries = unit([[1, 0, 0, 0], [0, 0, 1, 0]])
    candidates = clip_candidates(queries, IMAGES, IMAGE_IDS, [{1}, {2}], top_k=3)
    graph = build_transfer_graph(IMAGE_IDS, IMAGES, CONCEPTS, CONCEPT_VECTORS, TRIPLES, k=1)
    torch.manual_seed(0)
    model = ConceptToImageSage(DIM)
    torch.nn.init.normal_(model.conv.lin_l.weight)  # any graph weights at all
    h = image_graph_vectors(model, graph)

    assert transfer_ranks(candidates, h, IMAGE_IDS, alpha=1.0) == transfer_ranks(
        candidates, None, IMAGE_IDS, alpha=1.0
    )


def test_untrained_model_scores_like_clip_even_with_alpha_zero():
    queries = unit([[1, 0, 0, 0], [0.5, 0.5, 0, 0], [0, 0, 1, 0]])
    candidates = clip_candidates(queries, IMAGES, IMAGE_IDS, [{1}, {1}, {2}], top_k=3)
    graph = build_transfer_graph(IMAGE_IDS, IMAGES, CONCEPTS, CONCEPT_VECTORS, TRIPLES, k=1)
    # Initialised as W_self = I, W_nb = 0, b = 0 (ADR-0046): h equals the CLIP vector.
    h = image_graph_vectors(ConceptToImageSage(DIM), graph)

    assert transfer_ranks(candidates, h, IMAGE_IDS, alpha=0.0) == transfer_ranks(
        candidates, None, IMAGE_IDS, alpha=0.0
    )


def test_triples_change_the_two_layer_scores():
    graph = build_transfer_graph(IMAGE_IDS, IMAGES, CONCEPTS, CONCEPT_VECTORS, TRIPLES, k=1)
    model = CooccurrenceSage(DIM)
    with torch.no_grad():
        model.concept_conv.lin_l.weight.copy_(torch.eye(DIM))  # concepts hear their triples
        model.conv.lin_l.weight.copy_(torch.eye(DIM))  # images hear their concepts
    without_triples = graph.__class__(
        graph.image_ids, graph.x, graph.edge_index, torch.zeros((2, 0), dtype=torch.long)
    )

    with_h = image_graph_vectors(model, graph)
    without_h = image_graph_vectors(model, without_triples)

    # Image 0 links to "drum", which has a triple to "bronze": its vector must move.
    assert not np.allclose(with_h[0], without_h[0])
    # Image 2 links to "pagoda", which has no triple: its vector must not move.
    assert np.allclose(with_h[2], without_h[2])


def test_group_metrics_are_computed_per_label():
    ranks = [1, 2, None, 1]
    groups = {"type": ["direct", "direct", "indirect", "indirect"]}

    result = group_metrics(ranks, groups, recall_ks=[1], top_k=10)

    assert result["type"]["direct"]["R@1"] == pytest.approx(0.5)
    assert result["type"]["direct"]["MRR@10"] == pytest.approx(0.75)
    assert result["type"]["indirect"]["R@1"] == pytest.approx(0.5)
    assert result["type"]["indirect"]["fraction_queries"] == pytest.approx(0.5)


def test_query_groups_split_indirect_queries_by_specificity():
    from mmxai.eval.mmkg_transfer import query_groups

    queries = [
        {"type": "direct", "lang": "en", "relevant_items": ["I1"]},
        {"type": "indirect", "lang": "vi", "relevant_items": ["I1"]},
        {"type": "indirect", "lang": "en", "relevant_items": ["I1", "I2"]},
    ]

    groups = query_groups(queries)

    assert groups["type_language"] == ["direct_en", "indirect_vi", "indirect_en"]
    assert groups["indirect"] == ["direct", "indirect_specific_vi", "indirect_generic_en"]


def write_vg_seed(model_dir, two_layer: bool, alpha: float) -> None:
    seed_dir = model_dir / "seed0"
    seed_dir.mkdir(parents=True)
    model = CooccurrenceSage(DIM) if two_layer else ConceptToImageSage(DIM)
    torch.save(model.state_dict(), seed_dir / "model.pt")
    (seed_dir / "selection.json").write_text(
        f'{{"alpha": {alpha}, "best_epoch": 3, "git": "abc"}}', encoding="utf-8"
    )


def test_run_transfer_uses_the_vg_alpha_and_the_given_answers(tmp_path):
    from mmxai.eval.mmkg_transfer import run_transfer

    write_vg_seed(tmp_path / "two", two_layer=True, alpha=0.1)
    arrays = {
        "image_keys": np.array(["A_1", "A_2", "B_1"]),
        "image_ids": IMAGE_IDS,
        "image_features": IMAGES,
        "concept_keys": np.array(CONCEPTS),
        "concept_features": CONCEPT_VECTORS,
        "query_qids": np.array(["q1", "q2"]),
        "query_features": unit([[1, 0, 0, 0], [0, 0, 1, 0]]),
    }
    queries = [
        {"qid": "q1", "type": "direct", "lang": "en", "relevant": ["A_2"], "relevant_items": ["A"]},
        {"qid": "q2", "type": "direct", "lang": "vi", "relevant": ["B_1"], "relevant_items": ["B"]},
    ]
    config = {"top_k": 100, "recall_ks": [1], "clip_topk_k": 1}
    experiment = {"name": "x_mmkg_vi", "model_dir": str(tmp_path / "two"), "two_layer": True}

    metrics = run_transfer(config, experiment, 0, arrays, queries, TRIPLES)

    assert metrics["alpha"] == 0.1
    # Untrained weights score like CLIP: q1 ranks A_1 first (wrong), A_2 second; q2 is right.
    assert metrics["R@1"] == pytest.approx(0.5)
    assert metrics["MRR@3"] == pytest.approx(0.75)
    assert metrics["num_concept_edges"] == 2


def test_run_transfer_refuses_features_of_another_query_file():
    from mmxai.eval.mmkg_transfer import run_transfer

    arrays = {"query_qids": np.array(["old"])}
    queries = [{"qid": "new"}]

    with pytest.raises(ValueError, match="different query file"):
        run_transfer({}, {"name": "x", "model_dir": None}, 0, arrays, queries, [])


def test_query_spread_counts_distinct_top1_images_per_group():
    from mmxai.eval.mmkg_transfer import query_spread

    arrays = {
        "query_qids": np.array(["a", "b", "c"]),
        "query_features": unit([[1, 0, 0, 0], [0, 0, 1, 0], [1, 0.01, 0, 0]]),
        "image_features": IMAGES,
    }
    queries = [
        {"qid": "a", "type": "direct", "lang": "en"},
        {"qid": "b", "type": "direct", "lang": "en"},
        {"qid": "c", "type": "direct", "lang": "vi"},
    ]

    rows = {row["group"]: row for row in query_spread(arrays, queries)}

    assert rows["direct_en"]["distinct_top1_images"] == 2
    assert rows["direct_en"]["largest_top1_share"] == pytest.approx(0.5)
    assert rows["direct_en"]["mean_query_cosine"] == pytest.approx(0.0, abs=1e-6)
    assert rows["direct_vi"]["num_queries"] == 1
