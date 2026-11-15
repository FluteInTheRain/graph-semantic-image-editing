"""The drawn graph highlights exactly the edges Occlusion picked, and only real edges."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("faiss")
pytest.importorskip("matplotlib")

from matplotlib import pyplot as plt  # noqa: E402

from mmxai.explain.path_graph import (  # noqa: E402
    CONTAINS,
    QUERY_NODE,
    QUERY_TERM,
    SELF,
    build_path_graph,
    concept_node,
    draw_path_graph,
    image_node,
)
from mmxai.explain.path_view import shown_edges  # noqa: E402
from mmxai.explain.query_path import search_and_explain  # noqa: E402
from test_explain_query_path import QUERY, make_engine  # noqa: E402
from test_explain_ranking_model import EDGE_INDEX  # noqa: E402


def results_and_graph(num_edges: int = 1):
    engine = make_engine(alpha=0.5)
    results = search_and_explain(engine, QUERY, k=3)
    return results, build_path_graph(engine, results, num_edges)


def test_highlighted_edges_are_the_ones_occlusion_picked():
    results, graph = results_and_graph(num_edges=1)

    for item in results:
        positive, negative = shown_edges(item, num_edges=1)
        expected = {concept_node(edge.concept_row) for edge in positive + negative}
        drawn = {
            edge.source
            for edge in graph.edges
            if edge.kind == CONTAINS and edge.target == image_node(item)
        }
        assert drawn == expected


def test_every_highlighted_edge_exists_in_the_graph():
    _, graph = results_and_graph(num_edges=2)
    real_edges = {
        (f"concept:{source}", f"image:{100 + target}")  # image row r has id 100 + r
        for source, target in EDGE_INDEX.T.tolist()
    }

    drawn = [(edge.source, edge.target) for edge in graph.edges if edge.kind == CONTAINS]
    assert drawn  # otherwise the check below would pass on an empty drawing
    for pair in drawn:
        assert pair in real_edges


def test_contains_weight_is_the_occlusion_drop():
    results, graph = results_and_graph()

    for item in results:
        for edge in graph.edges:
            if edge.kind == CONTAINS and edge.target == image_node(item):
                row = int(edge.source.split(":")[1])
                assert edge.weight == item.occlusion_drop[row]


def test_every_image_has_its_own_part_drawn_from_the_query():
    results, graph = results_and_graph()

    for item in results:
        selves = [
            edge for edge in graph.edges if edge.kind == SELF and edge.target == image_node(item)
        ]
        assert len(selves) == 1
        assert selves[0].source == QUERY_NODE
        assert selves[0].weight == item.path.self_share


def test_a_concept_shared_by_two_images_is_one_node():
    _, graph = results_and_graph(num_edges=2)
    nodes = [node for node, _ in graph.concepts]

    assert len(nodes) == len(set(nodes))
    query_terms = [edge.target for edge in graph.edges if edge.kind == QUERY_TERM]
    assert sorted(query_terms) == sorted(nodes)


def test_drawing_gives_a_figure():
    _, graph = results_and_graph()
    figure = draw_path_graph(graph, "a test query")

    assert len(figure.axes) == 1
    plt.close(figure)
