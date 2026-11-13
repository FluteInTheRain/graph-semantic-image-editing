import pytest

from mmxai.eval.query_groups import (
    group_samples,
    group_sizes,
    load_predicates,
    metrics_by_group,
    relation_group,
    relation_groups,
    specificity_groups,
)

PREDICATE_YAML = """
predicate_templates:
  holding: is holding
  "on": is on
  in: is in
  next to: is next to
predicate_aliases:
  holds: holding
action_predicates: [holding]
"""


@pytest.fixture
def predicates(tmp_path):
    path = tmp_path / "predicates.yaml"
    path.write_text(PREDICATE_YAML, encoding="utf-8")
    return load_predicates(path, None, None, None)


def test_specificity_splits_on_the_number_of_relevant_images():
    assert specificity_groups([{1}, {1, 2}, {3}]) == ["specific", "generic", "specific"]


def test_action_wins_over_spatial(predicates):
    assert relation_group("man holding umbrella on street", predicates) == "action"


def test_multiword_predicate_matches_only_as_consecutive_words(predicates):
    assert relation_group("cat next to dog", predicates) == "spatial"
    assert relation_group("next dog to", predicates) == "none"


def test_predicate_does_not_match_inside_a_word(predicates):
    assert relation_group("onion", predicates) == "none"


def test_alias_joins_its_canonical_group(predicates):
    assert relation_group("man holds cup", predicates) == "action"


def test_excluded_predicate_is_not_a_relation(tmp_path):
    path = tmp_path / "predicates.yaml"
    path.write_text(PREDICATE_YAML, encoding="utf-8")
    predicates = load_predicates(path, None, None, ["in"])
    assert relation_group("man in shirt", predicates) == "none"


def test_extra_predicates_are_added_to_their_group(tmp_path):
    path = tmp_path / "predicates.yaml"
    path.write_text(PREDICATE_YAML, encoding="utf-8")
    predicates = load_predicates(path, ["eating"], ["behind"], None)
    assert relation_group("cat eating fish", predicates) == "action"
    assert relation_group("cat behind tree", predicates) == "spatial"


def test_missing_region_text_raises_with_the_id(predicates):
    with pytest.raises(KeyError, match="42"):
        relation_groups([42], {1: "man holding cup"}, predicates)


def test_relation_groups_follow_query_row_order(predicates):
    texts = {3: "cat next to dog", 5: "man holding cup"}
    assert relation_groups([5, 3], texts, predicates) == ["action", "spatial"]


def test_metrics_by_group_cover_every_query_once():
    rankings = [[1, 2], [2, 1], [3, 1], [1, 3]]
    relevant = [{1}, {1}, {3}, {2}]
    labels = ["a", "a", "b", "b"]
    result = metrics_by_group(rankings, relevant, labels, [1, 2], 2, 2)
    assert result["a"]["num_queries"] + result["b"]["num_queries"] == 4
    assert result["a"]["fraction_queries"] + result["b"]["fraction_queries"] == pytest.approx(1.0)
    assert result["a"]["R@1"] == 0.5


def test_group_sizes_give_count_and_fraction_of_every_label():
    sizes = group_sizes(["action", "none", "none", "none"])

    assert sizes == {"action": (1, 0.25), "none": (3, 0.75)}


def test_group_samples_pick_a_random_subset_of_each_group_with_a_fixed_seed():
    labels = ["action"] * 5 + ["none"] * 2
    sentences = [f"sentence {row}" for row in range(7)]

    first = group_samples(labels, sentences, per_group=3, seed=0)
    again = group_samples(labels, sentences, per_group=3, seed=0)

    assert first == again
    assert len(first["action"]) == 3
    assert set(first["action"]) <= set(sentences[:5])
    # A group smaller than per_group is shown whole.
    assert sorted(first["none"]) == ["sentence 5", "sentence 6"]


def test_specificity_groups_rejects_query_without_relevant_images():
    with pytest.raises(ValueError, match="at least one relevant image"):
        specificity_groups([set()])


def test_metrics_by_group_and_group_sizes_reject_empty_labels():
    with pytest.raises(ValueError, match="at least one query"):
        metrics_by_group([], [], [], [1], 1, 1)
    with pytest.raises(ValueError, match="at least one query"):
        group_sizes([])


def test_group_samples_rejects_mismatched_labels_and_sentences():
    with pytest.raises(ValueError, match="one sentence per label"):
        group_samples(["action"], ["s1", "s2"], per_group=1, seed=0)
