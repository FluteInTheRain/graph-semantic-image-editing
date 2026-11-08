import pytest

from mmxai.explain.occlusion import EdgeContribution
from mmxai.explain.presentation import build_explanation, format_explanation

NODE_NAMES = {42: "image_42", 1532: "ball", 877: "field", 903: "car"}
RELATION_NAMES = {(1532, 42): "in", (877, 1532): "under", (903, 42): "near"}

CONTRIBUTIONS = [
    EdgeContribution(source=1532, target=42, contribution=0.31),
    EdgeContribution(source=877, target=1532, contribution=0.12),
    EdgeContribution(source=903, target=42, contribution=-0.05),
]


def test_supporting_relations_are_named_and_sorted():
    explanation = build_explanation(CONTRIBUTIONS, 2, NODE_NAMES, RELATION_NAMES)

    names = [(r.source, r.relation, r.target) for r in explanation.supporting]
    assert names == [("ball", "in", "image_42"), ("field", "under", "ball")]


def test_negative_contributions_go_to_the_opposing_side():
    explanation = build_explanation(CONTRIBUTIONS, 2, NODE_NAMES, RELATION_NAMES)

    assert [r.source for r in explanation.opposing] == ["car"]


def test_top_m_limits_each_side():
    explanation = build_explanation(CONTRIBUTIONS, 1, NODE_NAMES, RELATION_NAMES)

    assert [r.source for r in explanation.supporting] == ["ball"]


def test_relations_that_did_not_change_the_score_are_left_out():
    contributions = [EdgeContribution(source=1532, target=42, contribution=0.0)]

    explanation = build_explanation(contributions, 3, NODE_NAMES, RELATION_NAMES)

    assert explanation.supporting == []
    assert explanation.opposing == []


def test_strongest_opposing_relation_comes_first():
    contributions = [
        EdgeContribution(source=903, target=42, contribution=-0.05),
        EdgeContribution(source=1532, target=42, contribution=-0.40),
    ]

    explanation = build_explanation(contributions, 2, NODE_NAMES, RELATION_NAMES)

    assert [r.contribution for r in explanation.opposing] == pytest.approx([-0.40, -0.05])


def test_missing_node_name_raises():
    contributions = [EdgeContribution(source=7, target=42, contribution=0.2)]

    with pytest.raises(ValueError, match="node 7"):
        build_explanation(contributions, 1, NODE_NAMES, RELATION_NAMES)


def test_missing_relation_name_raises():
    contributions = [EdgeContribution(source=42, target=1532, contribution=0.2)]

    with pytest.raises(ValueError, match="relation name"):
        build_explanation(contributions, 1, NODE_NAMES, RELATION_NAMES)


def test_rejects_top_m_below_one():
    with pytest.raises(ValueError, match="at least 1"):
        build_explanation(CONTRIBUTIONS, 0, NODE_NAMES, RELATION_NAMES)


def test_text_shows_both_sides():
    explanation = build_explanation(CONTRIBUTIONS, 2, NODE_NAMES, RELATION_NAMES)

    assert format_explanation(explanation) == (
        "Pushed the image up:\n"
        "  1. ball -[in]-> image_42 (+0.31)\n"
        "  2. field -[under]-> ball (+0.12)\n"
        "Pulled the image down:\n"
        "  1. car -[near]-> image_42 (-0.05)"
    )


def test_text_marks_an_empty_side():
    explanation = build_explanation(CONTRIBUTIONS[:1], 2, NODE_NAMES, RELATION_NAMES)

    assert format_explanation(explanation).endswith("Pulled the image down:\n  (none)")
