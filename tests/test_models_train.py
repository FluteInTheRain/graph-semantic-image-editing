"""Training the ADR-0046 model: hard negatives, loss and early stopping, on tiny data."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("faiss")

from mmxai.models.sage import ConceptToImageSage  # noqa: E402
from mmxai.models.train import (  # noqa: E402
    TrainingData,
    batch_loss,
    fit,
    hard_negative_pool,
    same_text_mask,
    sample_negatives,
    texts_with_positive_in_candidates,
)


def unit_rows(matrix: np.ndarray) -> np.ndarray:
    return (matrix / np.linalg.norm(matrix, axis=1, keepdims=True)).astype(np.float32)


def test_hard_negatives_never_include_the_texts_own_image():
    generator = np.random.default_rng(0)
    image_x = unit_rows(generator.normal(size=(6, 4)))
    text_x = image_x.copy()  # each text sits exactly on its image
    text_image_row = np.arange(6)

    candidate_rows, is_negative = hard_negative_pool(image_x, text_x, text_image_row, 3, 0.9999)

    assert (candidate_rows[:, 0] == text_image_row).all()  # CLIP finds the image first
    assert not is_negative[:, 0].any()
    assert is_negative[:, 1:].all()


def test_image_with_the_same_sentence_is_not_a_negative():
    # Texts 0 and 1 are the same sentence ("a man") on images 0 and 1 (ADR-0043).
    text_x = unit_rows(np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
    text_image_row = np.array([0, 1, 2])
    candidate_rows = np.array([[0, 1, 2], [1, 0, 2], [2, 0, 1]])

    is_same = same_text_mask(text_x, text_image_row, candidate_rows, 0.9999)

    assert is_same[0].tolist() == [True, True, False]
    assert is_same[2].tolist() == [True, False, False]


def test_sample_negatives_only_picks_allowed_columns():
    is_negative = torch.tensor([[False, True, True, False, True]] * 50)

    columns, valid = sample_negatives(is_negative, 3, torch.Generator().manual_seed(0))

    assert valid.all()
    assert set(columns.ravel().tolist()) <= {1, 2, 4}


def test_sample_negatives_marks_missing_slots_invalid():
    is_negative = torch.tensor([[False, True, False]])

    _, valid = sample_negatives(is_negative, 2, torch.Generator().manual_seed(0))

    assert valid.tolist() == [[True, False]]


def tiny_training_data() -> TrainingData:
    """4 images, 1 text each; the text equals its image's concept mean, not its image.

    Plain CLIP (h = x) is therefore not perfect, and the graph part can do better.
    """
    generator = np.random.default_rng(0)
    image_x = torch.from_numpy(unit_rows(generator.normal(size=(4, 8))))
    neighbor_mean = torch.from_numpy(unit_rows(generator.normal(size=(4, 8))))
    rows = torch.arange(4)
    candidate_rows = torch.tensor([[1, 2, 3, 0], [0, 2, 3, 1], [0, 1, 3, 2], [0, 1, 2, 3]])
    is_negative = candidate_rows != rows[:, None]
    return TrainingData(
        image_x=image_x,
        neighbor_mean=neighbor_mean,
        text_x=neighbor_mean.clone(),
        text_image_row=rows,
        candidate_rows=candidate_rows,
        is_negative=is_negative,
    )


def test_loss_is_finite_with_masked_slots():
    data = tiny_training_data()
    # Text 0 may only use image 1 as a negative, so 2 of its 3 slots are masked.
    data.is_negative[0] = torch.tensor([True, False, False, False])
    model = ConceptToImageSage(8)

    loss = batch_loss(model, data, torch.arange(4), 3, 0.05, torch.Generator().manual_seed(0))

    assert torch.isfinite(loss)


def test_asking_more_negatives_than_candidates_is_an_error():
    with pytest.raises(ValueError, match="5 negatives"):
        sample_negatives(torch.ones(2, 4, dtype=torch.bool), 5, torch.Generator())


def full_batch_loss(model: ConceptToImageSage, data: TrainingData) -> float:
    with torch.no_grad():
        generator = torch.Generator().manual_seed(1)
        return batch_loss(model, data, torch.arange(4), 3, 0.05, generator).item()


def test_training_lowers_the_loss_when_the_graph_is_useful():
    torch.manual_seed(0)
    data = tiny_training_data()
    model = ConceptToImageSage(8)
    before = full_batch_loss(model, data)

    def validate(current_model):
        return -full_batch_loss(current_model, data)

    result = fit(
        model, data, validate, 0.05, 4, 3, 0.05, 30, 100, torch.Generator().manual_seed(0), str
    )
    model.load_state_dict(result.best_state)

    assert full_batch_loss(model, data) < before


def test_early_stopping_keeps_the_best_epoch():
    data = tiny_training_data()
    model = ConceptToImageSage(8)
    val_scores = iter([0.1, 0.3, 0.2, 0.2, 0.2])  # best after epoch 1

    result = fit(
        model,
        data,
        lambda m: next(val_scores),
        0.01,
        4,
        3,
        0.05,
        10,
        2,
        torch.Generator().manual_seed(0),
        log=str,
    )

    assert result.best_epoch == 1
    assert len(result.history) == 4  # epoch 0, then 1-3; stops after 2 epochs without gain


def test_untrained_model_is_kept_if_training_never_helps():
    data = tiny_training_data()
    model = ConceptToImageSage(8)
    untrained = {name: value.clone() for name, value in model.state_dict().items()}
    val_scores = iter([0.5, 0.4, 0.4])

    result = fit(
        model,
        data,
        lambda m: next(val_scores),
        0.01,
        4,
        3,
        0.05,
        10,
        2,
        torch.Generator().manual_seed(0),
        log=str,
    )

    assert result.best_epoch == 0
    for name, value in untrained.items():
        assert torch.equal(result.best_state[name], value)


def test_same_seed_gives_the_same_trained_weights():
    def train_once():
        torch.manual_seed(0)
        model = ConceptToImageSage(8)
        result = fit(
            model,
            tiny_training_data(),
            lambda m: float(m.conv.lin_l.weight.detach().sum()),
            0.01,
            2,
            2,
            0.05,
            3,
            10,
            torch.Generator().manual_seed(0),
            log=str,
        )
        return result.best_state

    first = train_once()
    second = train_once()

    for name in first:
        assert torch.equal(first[name], second[name])


def test_only_texts_with_their_image_among_the_candidates_are_kept():
    # Text 1's image (row 1) is not in its CLIP candidates: it is left out, otherwise
    # "the candidate CLIP likes least is correct" becomes a shortcut (ADR-0047).
    text_image_row = np.array([0, 1, 2])
    candidate_rows = np.array([[0, 3], [2, 3], [3, 2]])

    keep = texts_with_positive_in_candidates(text_image_row, candidate_rows)

    assert keep.tolist() == [True, False, True]
