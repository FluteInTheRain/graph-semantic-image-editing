import numpy as np
import pytest

from mmxai.explain.quality import sparsity, validity

# Image 0 <- object 1 <- object 2, and image 0 <- object 3. Node 9 -> 8 is far away.
GRAPH = np.array([[1, 2, 3, 9], [0, 1, 0, 8]])


def test_sparsity_on_a_hand_computed_example():
    assert sparsity(num_explained=2, num_in_reach=8) == pytest.approx(0.75)


def test_sparsity_rejects_more_relations_than_exist():
    with pytest.raises(ValueError, match="cannot explain"):
        sparsity(num_explained=5, num_in_reach=4)


def test_sparsity_rejects_an_empty_neighbourhood():
    with pytest.raises(ValueError, match="no relations"):
        sparsity(num_explained=0, num_in_reach=0)


def test_a_path_into_the_image_is_fully_valid():
    result = validity(GRAPH, [(2, 1), (1, 0)], 0, num_layers=2, both_directions=False)

    assert result.all_in_reach
    assert result.connected_share == pytest.approx(1.0)


def test_a_relation_not_linked_to_the_image_lowers_the_connected_share():
    # (2, 1) alone does not touch the image.
    result = validity(GRAPH, [(2, 1), (3, 0)], 0, num_layers=2, both_directions=False)

    assert result.connected_share == pytest.approx(0.5)


def test_a_relation_far_from_the_image_is_not_in_reach():
    result = validity(GRAPH, [(9, 8)], 0, num_layers=2, both_directions=False)

    assert not result.all_in_reach


def test_a_made_up_relation_is_not_in_reach():
    result = validity(GRAPH, [(5, 0)], 0, num_layers=2, both_directions=False)

    assert not result.all_in_reach


def test_undirected_relation_matches_either_direction():
    undirected = np.array([[0, 1], [1, 0]])

    result = validity(undirected, [(0, 1)], 0, num_layers=1, both_directions=True)

    assert result.all_in_reach


def test_empty_explanation_has_nothing_invalid():
    result = validity(GRAPH, [], 0, num_layers=2, both_directions=False)

    assert result.all_in_reach
    assert result.connected_share == 1.0
