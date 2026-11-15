"""The demo table shows every part of g, so its rows add up to the graph score."""

import re

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
    """The 'part of g' numbers of every row except the total."""
    values = []
    for line in table.splitlines():
        if line.startswith("| query") or re.match(r"\| \d+ other edges", line):
            values.append(float(line.split("|")[2]))
    return values


def fake_item(shares: list[float]) -> ImagePath:
    """An image with concept rows 3, 4, 5, ... and the given shares (only formatting is tested)."""
    concept_shares = [ConceptShare(3 + k, share) for k, share in enumerate(shares)]
    concept_shares.sort(key=lambda edge: edge.share, reverse=True)
    graph_score = 0.2 + sum(shares)
    path = ScorePath(0, graph_score, 0.21, -0.01, concept_shares)
    drops = {edge.concept_row: edge.share for edge in concept_shares}
    return ImagePath(1, 2, 100, 0.3, graph_score, graph_score, path, drops)


def test_rows_add_up_to_the_graph_score_on_real_results():
    engine = make_engine(alpha=0.5)
    for item in search_and_explain(engine, QUERY, k=3):
        table = format_image_path(engine, item, num_edges=1)
        assert sum(part_column(table)) == pytest.approx(item.graph_score, abs=1e-3)


def test_the_image_itself_and_the_bias_are_always_shown():
    table = format_image_path(make_engine(alpha=0.5), fake_item([0.01]), num_edges=3)

    assert "| query → image itself | +0.2100 |" in table
    assert "| query → bias | -0.0100 |" in table


def test_strongest_edges_of_each_side_are_shown_and_the_rest_summed():
    engine = make_engine(alpha=0.5)
    # Rows 3, 4, 5 get +0.03, -0.02, +0.01: with one per side, row 5 is summed.
    table = format_image_path(engine, fake_item([0.03, -0.02, 0.01]), num_edges=1)

    assert f"| query → {concept_name(engine, 3)} → contains → image | +0.0300 |" in table
    assert f"| query → {concept_name(engine, 4)} → contains → image | -0.0200 |" in table
    assert concept_name(engine, 5) not in table
    assert "| 1 other edges | +0.0100 |" in table


def test_no_other_edges_row_when_every_edge_is_shown():
    table = format_image_path(make_engine(alpha=0.5), fake_item([0.03, -0.02]), num_edges=3)

    assert "other edges" not in table


def test_alpha_one_says_the_graph_plays_no_part():
    table = format_image_path(make_engine(alpha=1.0), fake_item([0.01]), num_edges=3)

    assert "the graph plays no part" in table


def test_num_edges_below_one_is_rejected():
    with pytest.raises(ValueError, match="num_edges"):
        format_image_path(make_engine(alpha=0.5), fake_item([0.01]), num_edges=0)
