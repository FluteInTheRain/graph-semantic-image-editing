"""search_retriever ranks like G-Retriever stage 2 and explains with the concepts it scores."""

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pcst_fast")

from mmxai.eval.graph_rerank import SplitGraph  # noqa: E402
from mmxai.explain.retriever_path import (  # noqa: E402
    RetrieverEngine,
    query_subgraph_for,
    search_retriever,
)
from mmxai.graph.contains import ContainsGraph  # noqa: E402
from mmxai.models.subgraph_sage import SubgraphSage  # noqa: E402

DIM = 6
NUM_IMAGES = 5
IMAGE_IDS = np.arange(200, 200 + NUM_IMAGES, dtype=np.int64)
KEYS = np.array(["a.n.01", "b.n.01", "c.n.01", "d.n.01"])
# Image i contains concepts listed here (row numbers into KEYS).
IMAGE_CONCEPTS = [[0, 1], [1, 2], [2, 3], [3], [0, 3]]
# PMI concept graph: a chain a - b - c - d.
PAIRS = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)


def unit(rows: np.ndarray) -> np.ndarray:
    return (rows / np.linalg.norm(rows, axis=1, keepdims=True)).astype(np.float32)


RNG = np.random.default_rng(0)
IMAGES = unit(RNG.normal(size=(NUM_IMAGES, DIM)))
CONCEPTS = unit(RNG.normal(size=(len(KEYS), DIM)))
# Close to concepts a and c, so PCST must keep b as a bridge to connect them.
QUERY = unit((CONCEPTS[0] + CONCEPTS[2] + 0.1 * RNG.normal(size=DIM))[None, :])[0]


def trained_model() -> SubgraphSage:
    """Random weights, so the subgraph really changes the score."""
    torch.manual_seed(0)
    model = SubgraphSage(DIM)
    with torch.no_grad():
        model.neighbor_weight.weight.normal_()
        model.neighbor_weight.bias.normal_(std=0.1)
    return model


def make_engine(alpha: float) -> RetrieverEngine:
    edge_image = np.array([i for i, concepts in enumerate(IMAGE_CONCEPTS) for _ in concepts])
    edge_concept = np.array([c for concepts in IMAGE_CONCEPTS for c in concepts])
    graph = ContainsGraph(IMAGE_IDS, np.array(["val"] * NUM_IMAGES), KEYS, edge_image, edge_concept)
    x = torch.from_numpy(np.concatenate([IMAGES, CONCEPTS]))
    edge_index = torch.from_numpy(np.array([NUM_IMAGES + edge_concept, edge_image]))
    config = SimpleNamespace(
        prize_k=2, edge_cost=0.1, subgraph_mode="pcst", top_k=NUM_IMAGES, score_chunk=4
    )
    return RetrieverEngine(
        config=config,
        inputs=SimpleNamespace(graph=graph, concept_features=CONCEPTS),
        pairs=PAIRS,
        split_graph=SplitGraph(IMAGE_IDS, x, edge_index, None),
        model=trained_model(),
        alpha=alpha,
        image_embeddings=IMAGES,
        image_ids=IMAGE_IDS,
    )


def expected_graph_score(model: SubgraphSage, image: int, kept: set[int]) -> float:
    """cos(q, W_self x_i + W_nb mean(x_c for c in N(i) ∩ kept) + b), written out by hand."""
    concepts = [c for c in IMAGE_CONCEPTS[image] if c in kept]
    mean = CONCEPTS[concepts].mean(axis=0) if concepts else np.zeros(DIM, dtype=np.float32)
    with torch.no_grad():
        h = model(torch.from_numpy(IMAGES[image]), torch.from_numpy(mean)).numpy()
    return float(h @ QUERY / np.linalg.norm(h))


def subgraph_rows(engine: RetrieverEngine) -> set[int]:
    return set(query_subgraph_for(engine, QUERY)[0].concepts.tolist())


def test_results_are_ordered_by_the_fused_score():
    engine = make_engine(alpha=0.5)
    kept = subgraph_rows(engine)
    expected = [
        0.5 * float(IMAGES[i] @ QUERY) + 0.5 * expected_graph_score(engine.model, i, kept)
        for i in range(NUM_IMAGES)
    ]

    answer = search_retriever(engine, QUERY, k=NUM_IMAGES)

    assert [item.image_id for item in answer.results] == IMAGE_IDS[
        np.argsort(expected)[::-1]
    ].tolist()
    for item in answer.results:
        assert item.final_score == pytest.approx(expected[item.image_id - 200], abs=1e-5)


def test_alpha_one_keeps_the_clip_order():
    answer = search_retriever(make_engine(alpha=1.0), QUERY, k=NUM_IMAGES)
    assert [item.clip_rank for item in answer.results] == list(range(1, NUM_IMAGES + 1))


def test_pcst_keeps_a_bridge_between_the_prized_concepts():
    answer = search_retriever(make_engine(alpha=0.5), QUERY, k=1)
    assert sorted(answer.prized) == ["a.n.01", "c.n.01"]
    assert answer.bridges == ["b.n.01"]


def test_matched_concepts_are_the_image_concepts_inside_the_subgraph():
    engine = make_engine(alpha=0.5)
    kept = subgraph_rows(engine)
    for item in search_retriever(engine, QUERY, k=NUM_IMAGES).results:
        image = item.image_id - 200
        expected = sorted(KEYS[c] for c in IMAGE_CONCEPTS[image] if c in kept)
        assert sorted(concept.name for concept in item.matched) == expected


def test_drop_is_the_score_lost_when_the_concept_leaves_the_subgraph():
    engine = make_engine(alpha=0.5)
    kept = subgraph_rows(engine)
    row_of_key = {str(key): row for row, key in enumerate(KEYS)}
    for item in search_retriever(engine, QUERY, k=NUM_IMAGES).results:
        image = item.image_id - 200
        for concept in item.matched:
            without = expected_graph_score(engine.model, image, kept - {row_of_key[concept.name]})
            assert concept.drop == pytest.approx(item.graph_score - without, abs=1e-5)


@pytest.mark.parametrize("k", [0, NUM_IMAGES + 1])
def test_k_outside_the_candidates_is_rejected(k):
    with pytest.raises(ValueError):
        search_retriever(make_engine(alpha=0.5), QUERY, k=k)
