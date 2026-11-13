"""Shared evaluation reports (text->image, image->text, query groupings) on tiny data."""

import json

import numpy as np
import pytest

import mmxai.eval.extended
from mmxai.eval.extended import (
    check_extension_settings,
    image_to_text_report,
    query_group_inputs,
    query_groupings,
    ranked_image_ids,
    text_to_image_report,
)
from mmxai.eval.image_to_text import TextCandidates
from mmxai.eval.metrics import compute_metrics
from mmxai.eval.query_groups import load_predicates
from mmxai.rerank.fusion import Candidates, rerank_ranks, retrieve_candidates

PREDICATE_YAML = """
predicate_templates:
  holding: is holding
  "on": is on
predicate_aliases:
  holds: holding
action_predicates: [holding]
"""


def make_candidates():
    """Two queries, three candidates each. Query 0: image 7 is correct; query 1: images 5 and 6."""
    return Candidates(
        query_embeddings=np.zeros((2, 2), dtype=np.float32),
        candidate_ids=np.array([[5, 6, 7], [5, 6, 7]]),
        clip_scores=np.array([[0.9, 0.8, 0.7], [0.9, 0.8, 0.7]], dtype=np.float32),
        relevant=[{7}, {5, 6}],
    )


def test_ranked_image_ids_sorts_by_final_score_with_stable_ties():
    candidates = make_candidates()
    final = np.array([[0.1, 0.1, 0.9], [0.5, 0.5, 0.5]])

    assert ranked_image_ids(candidates, final) == [[7, 5, 6], [5, 6, 7]]


def test_text_to_image_report_matches_the_pipeline_ranks():
    candidates = make_candidates()
    final = candidates.clip_scores

    report = text_to_image_report(candidates, final, None, [1, 3], 3, 3)

    expected = compute_metrics(rerank_ranks(candidates, final), [1, 3], 3)
    for key, value in expected.items():
        assert report[key] == value
    assert "nDCG@3" in report
    assert "groups" not in report


def test_text_to_image_report_adds_groups_when_given_groupings():
    candidates = make_candidates()
    groupings = {"specificity": ["specific", "generic"]}

    report = text_to_image_report(candidates, candidates.clip_scores, groupings, [1], 3, 3)

    assert set(report["groups"]["specificity"]) == {"specific", "generic"}


def test_report_raises_when_ranks_disagree(monkeypatch):
    candidates = make_candidates()
    monkeypatch.setattr(mmxai.eval.extended, "rerank_ranks", lambda c, s: [None] * len(s))

    with pytest.raises(ValueError, match="disagree"):
        text_to_image_report(candidates, candidates.clip_scores, None, [1], 3, 3)


def test_image_to_text_report_scores_the_ranked_text_rows():
    # Image 0: text row 4 is correct and ranked first. Image 1: row 9 is correct, ranked second.
    candidates = TextCandidates(
        query_image_ids=np.array([10, 11]),
        text_embeddings=np.zeros((10, 2), dtype=np.float32),
        candidate_rows=np.array([[4, 2], [3, 9]]),
        clip_scores=np.array([[0.9, 0.5], [0.8, 0.7]], dtype=np.float32),
        relevant=[{4}, {9}],
    )

    report = image_to_text_report(candidates, candidates.clip_scores, [1, 2], 2, 2)

    assert report["R@1"] == 0.5
    assert report["R@2"] == 1.0
    assert "nDCG@2" in report


def test_query_groupings_label_each_query_by_specificity_and_relation(tmp_path):
    predicates_file = tmp_path / "predicates.yaml"
    predicates_file.write_text(PREDICATE_YAML, encoding="utf-8")
    predicates = load_predicates(predicates_file, None, None, None)
    regions = [
        {
            "image_id": 5,
            "regions": [{"region_id": 100, "image_id": 5, "phrase": "man holding cup"}],
        },
        {"image_id": 6, "regions": [{"region_id": 101, "image_id": 6, "phrase": "red car"}]},
    ]
    (tmp_path / "region_descriptions.json").write_text(json.dumps(regions), encoding="utf-8")
    candidates = make_candidates()

    groupings = query_groupings(candidates, np.array([100, 101]), str(tmp_path), predicates)

    assert groupings["specificity"] == ["specific", "generic"]
    assert groupings["relation"] == ["action", "none"]


def test_query_group_inputs_follow_the_query_rows_of_retrieve_candidates(tmp_path):
    pytest.importorskip("faiss")
    predicates_file = tmp_path / "predicates.yaml"
    predicates_file.write_text(PREDICATE_YAML, encoding="utf-8")
    # Texts of the split's images (1 and 3) are interleaved with a text of image 2.
    text_embeddings = np.eye(5, dtype=np.float32)
    texts = {
        "embeddings": text_embeddings,
        "image_ids": np.array([3, 1, 2, 3, 1]),
        "text_ids": np.array([10, 11, 12, 13, 14]),
    }
    split_image_ids = np.array([1, 3])
    candidates = retrieve_candidates(
        np.eye(5, dtype=np.float32)[:3],
        np.array([1, 2, 3]),
        texts["embeddings"],
        texts["image_ids"],
        split_image_ids,
        2,
        0.9999,
    )

    text_ids, predicates = query_group_inputs(
        texts, split_image_ids, str(predicates_file), None, None, None
    )

    assert text_ids.tolist() == [10, 11, 13, 14]
    # Query q of retrieve_candidates is the text whose id is text_ids[q].
    rows = [texts["text_ids"].tolist().index(text_id) for text_id in text_ids.tolist()]
    np.testing.assert_array_equal(candidates.query_embeddings, text_embeddings[rows])
    assert predicates == load_predicates(predicates_file, None, None, None)


def test_check_extension_settings_rejects_partial_group_inputs():
    with pytest.raises(ValueError, match="both vg_data_dir and predicates_file"):
        check_extension_settings(10, False, "data/vg", None)
    with pytest.raises(ValueError, match="both vg_data_dir and predicates_file"):
        check_extension_settings(10, False, None, "configs/preds.yaml")


def test_check_extension_settings_rejects_groups_or_image_to_text_without_ndcg():
    with pytest.raises(ValueError, match="ndcg_k"):
        check_extension_settings(None, False, "data/vg", "configs/preds.yaml")
    with pytest.raises(ValueError, match="ndcg_k"):
        check_extension_settings(None, True, None, None)


def test_check_extension_settings_accepts_disabled_and_complete_settings():
    check_extension_settings(None, False, None, None)
    check_extension_settings(10, True, "data/vg", "configs/preds.yaml")


def test_ranked_image_ids_rejects_mismatched_final_scores_shape():
    candidates = make_candidates()
    with pytest.raises(ValueError, match="candidate_ids"):
        ranked_image_ids(candidates, np.zeros((2, 2), dtype=np.float32))
