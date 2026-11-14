import pytest

from mmxai.explain.statistics import (
    average_ranks,
    choose_on_validation,
    mean_and_std,
    spearman,
    wilcoxon_greater,
)


def test_average_ranks_share_ties():
    assert average_ranks([10, 20, 20, 5]).tolist() == [2.0, 3.5, 3.5, 1.0]


def test_spearman_is_one_for_the_same_order():
    assert spearman([0.1, 0.5, 0.9], [1, 2, 30]) == pytest.approx(1.0)


def test_spearman_is_minus_one_for_the_reverse_order():
    assert spearman([0.1, 0.5, 0.9], [3, 2, 1]) == pytest.approx(-1.0)


def test_spearman_rejects_constant_lists():
    with pytest.raises(ValueError, match="all equal"):
        spearman([1, 1, 1], [1, 2, 3])


def test_wilcoxon_all_positive_gives_the_smallest_p_value():
    # Only 1 of the 2^5 sign choices is all positive.
    assert wilcoxon_greater([0.1, 0.2, 0.3, 0.4, 0.5]) == pytest.approx(1 / 32)


def test_wilcoxon_on_a_hand_computed_example():
    # Ranks 1, 2, 3; positive rank sum = 1 + 3 = 4.
    # Subset sums of {1, 2, 3}: 0, 1, 2, 3, 3, 4, 5, 6, so P(sum >= 4) = 3/8.
    assert wilcoxon_greater([1.0, -2.0, 3.0]) == pytest.approx(3 / 8)


def test_wilcoxon_drops_zero_differences():
    assert wilcoxon_greater([0.0, 0.0, 1.0]) == pytest.approx(1 / 2)


def test_wilcoxon_with_no_differences_is_not_significant():
    assert wilcoxon_greater([0.0, 0.0]) == 1.0


def test_wilcoxon_handles_tied_magnitudes():
    # |1| and |-1| tie at rank 1.5; positive rank sum = 1.5 + 3 = 4.5.
    # Doubled ranks 3, 3, 6: subset sums 0, 3, 3, 6, 6, 9, 9, 12, so P(>= 9) = 3/8.
    assert wilcoxon_greater([1.0, -1.0, 2.0]) == pytest.approx(3 / 8)


def test_mean_and_std_use_the_sample_standard_deviation():
    mean, std = mean_and_std([1.0, 2.0, 3.0])

    assert mean == pytest.approx(2.0)
    assert std == pytest.approx(1.0)


def test_mean_and_std_need_two_values():
    with pytest.raises(ValueError, match="at least 2"):
        mean_and_std([1.0])


def test_choose_on_validation_picks_the_highest_mean():
    scores = {"occlusion": [0.6, 0.7], "integrated_gradients": [0.8, 0.9]}

    assert choose_on_validation(scores, split="val") == "integrated_gradients"


def test_choose_on_validation_breaks_ties_in_sorted_order():
    scores = {"occlusion": [0.5], "gnn_explainer": [0.5]}

    assert choose_on_validation(scores, split="val") == "gnn_explainer"


def test_choose_on_validation_refuses_test_scores():
    with pytest.raises(ValueError, match="val only"):
        choose_on_validation({"occlusion": [0.5]}, split="test")


def test_choose_on_validation_skips_missing_scores():
    scores = {("occlusion", 3): [0.2, None], ("occlusion", 5): [0.4, None]}

    assert choose_on_validation(scores, split="val") == ("occlusion", 5)


def test_choose_on_validation_rejects_an_option_without_scores():
    with pytest.raises(ValueError, match="no scores"):
        choose_on_validation({"occlusion": [None]}, split="val")


def test_wilcoxon_treats_float_noise_as_zero():
    assert wilcoxon_greater([1e-17, 1.0]) == pytest.approx(1 / 2)
