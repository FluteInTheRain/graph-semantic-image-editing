"""The demo table shows the Occlusion edges and every part of g, so its parts add up to g."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("faiss")

from mmxai.explain.path_view import format_image_path  # noqa: E402
from mmxai.explain.query_path import (  # noqa: E402
    ImagePath,
    concept_name,
    search_and_explain,
)
from mmxai.explain.score_path import ConceptShare, ScorePath  # noqa: E402
from test_explain_query_path import QUERY, make_engine  # noqa: E402


def part_column(table: str) -> list[float]:
    """The 'part of g' numbers of every row except the header and the total."""
    values = []
    for line in table.splitlines():
        cells = [cell.strip() for cell in line.split("|")]
        if len(cells) == 5 and cells[3] and not cells[1].startswith(("**", "term", "---")):
            values.append(float(cells[3]))
    return values


def fake_item(shares: list[float], drops: list[float]) -> ImagePath:
    """An image with concept rows 3, 4, 5, ... (only formatting is tested here)."""
    concept_shares = [ConceptShare(3 + k, share) for k, share in enumerate(shares)]
    concept_shares.sort(key=lambda edge: edge.share, reverse=True)
    graph_score = 0.2 + sum(shares)
    path = ScorePath(0, graph_score, 0.21, -0.01, concept_shares)
    occlusion_drop = {3 + k: drop for k, drop in enumerate(drops)}
    return ImagePath(1, 2, 100, 0.3, graph_score, graph_score, path, occlusion_drop)


def test_rows_add_up_to_the_graph_score_on_real_results():
    engine = make_engine(alpha=0.5)
    for item in search_and_explain(engine, QUERY, k=3):
        table = format_image_path(engine, item, num_edges=1)
        assert sum(part_column(table)) == pytest.approx(item.graph_score, abs=1e-3)


def test_the_image_itself_and_the_bias_are_always_shown():
    table = format_image_path(make_engine(alpha=0.5), fake_item([0.01], [0.01]), num_edges=3)

    assert "| image itself (W_self x_i) | | +0.2100 |" in table
    assert "| bias b | | -0.0100 |" in table


def test_edges_are_chosen_by_occlusion_not_by_their_part_of_g():
    engine = make_engine(alpha=0.5)
    # Row 3 has the largest part of g (+0.03) but removing it raises g (drop -0.04):
    # it is shown on the pulling-down side. Row 4 is the weakest by Occlusion: summed.
    item = fake_item(shares=[0.03, -0.02, 0.01], drops=[-0.04, -0.01, 0.02])
    table = format_image_path(engine, item, num_edges=1)

    assert f"| {concept_name(engine, 5)} → contains → image | +0.0200 | +0.0100 |" in table
    assert f"| {concept_name(engine, 3)} → contains → image | -0.0400 | +0.0300 |" in table
    assert concept_name(engine, 4) not in table
    assert "| 1 other edges | | -0.0200 |" in table


def test_no_other_edges_row_when_every_edge_is_shown():
    item = fake_item(shares=[0.03, -0.02], drops=[0.03, -0.02])
    table = format_image_path(make_engine(alpha=0.5), item, num_edges=3)

    assert "other edges" not in table


def test_alpha_one_says_the_graph_plays_no_part():
    table = format_image_path(make_engine(alpha=1.0), fake_item([0.01], [0.01]), num_edges=3)

    assert "the graph plays no part" in table


def test_num_edges_below_one_is_rejected():
    with pytest.raises(ValueError, match="num_edges"):
        format_image_path(make_engine(alpha=0.5), fake_item([0.01], [0.01]), num_edges=0)
