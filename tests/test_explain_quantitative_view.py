"""The level-3 demo tables show the explanation edges and the measures of each image."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("faiss")

from mmxai.explain.quantitative_view import (  # noqa: E402
    format_quantified,
    format_quantified_summary,
)
from mmxai.explain.query_path import concept_name  # noqa: E402
from mmxai.explain.query_quantitative import explain_quantitatively  # noqa: E402
from test_explain_query_path import QUERY, make_engine  # noqa: E402
from test_explain_query_quantitative import config  # noqa: E402


def results(alpha: float = 0.5):
    engine = make_engine(alpha=alpha)
    return engine, explain_quantitatively(engine, QUERY, 3, config(), seed=0)


def test_every_explanation_edge_is_named_in_the_table():
    engine, items = results()
    for item in items:
        table = format_quantified(engine, item, "integrated_gradients")
        for concept_row, _ in item.explanation:
            assert f"| {concept_name(engine, concept_row)} → contains → image |" in table


def test_measured_image_shows_fidelity_next_to_the_random_control():
    engine, items = results()
    measured = [item for item in items if item.report is not None]
    assert measured
    for item in measured:
        table = format_quantified(engine, item, "integrated_gradients")
        expected = f"| {item.report.fidelity_plus:+.4f} | {item.report.random_plus_mean:+.4f} |"
        assert expected in table


def test_unmeasured_image_says_why():
    engine, items = results()
    unmeasured = [item for item in items if item.report is None]
    assert unmeasured  # image 102 has a single edge, which is the whole explanation
    table = format_quantified(engine, unmeasured[0], "integrated_gradients")
    assert "Not measured. The explanation keeps every edge" in table


def test_summary_averages_only_the_measured_images():
    _, items = results()
    measured = [item.report for item in items if item.report is not None]
    summary = format_quantified_summary(items)

    mean_plus = sum(report.fidelity_plus for report in measured) / len(measured)
    assert f"Mean over {len(measured)} of {len(items)} images" in summary
    assert f"| Fid+ (g lost) | {mean_plus:+.4f} |" in summary


def test_summary_with_alpha_one_says_nothing_was_measured():
    _, items = results(alpha=1.0)
    assert format_quantified_summary(items).startswith("No image could be measured")
