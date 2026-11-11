"""Fidelity: does removing or keeping only the explanation change the score as expected?"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from mmxai.explain.config import ExplainConfig
from mmxai.explain.intervention import mark_remaining_edges
from mmxai.explain.quality import sparsity, validity
from mmxai.explain.subgraph import relations_in_reach


def fidelity_plus(full_score: float, score_without_explanation: float) -> float:
    """How much the score drops when the explanation edges are removed.

    Larger is better: a faithful explanation is necessary for the score.
    """
    return full_score - score_without_explanation


def fidelity_minus(full_score: float, score_with_only_explanation: float) -> float:
    """How much the score drops when only the explanation edges are kept.

    Closer to 0 is better: a faithful explanation is enough to keep the score.
    """
    return full_score - score_with_only_explanation


def rank_drop(rank_before: int, rank_after: int) -> int:
    """How many places a candidate falls after an intervention (1 = top rank).

    Positive means it fell, negative means it rose.
    """
    if rank_before < 1 or rank_after < 1:
        raise ValueError(f"ranks start at 1, got {rank_before} and {rank_after}")
    return rank_after - rank_before


def rank_of(scores: np.ndarray, candidate: int) -> int:
    """Rank of one candidate among all candidates (1 = highest score).

    Ties share the better rank, so a tie never makes a candidate look like it fell.
    """
    return 1 + int((scores > scores[candidate]).sum())


@dataclass(frozen=True)
class FidelityResult:
    """Fidelity of one explanation for one candidate."""

    fidelity_plus: float
    fidelity_minus: float
    rank_drop: int


def evaluate_fidelity(
    edge_index: np.ndarray,
    explanation_edges: Iterable[tuple[int, int]],
    both_directions: bool,
    graph_scores: Callable[[np.ndarray], np.ndarray],
    cosine: np.ndarray,
    alpha: float,
    candidate: int,
) -> FidelityResult:
    """Score the graph three times (full, without the explanation, only the explanation)
    and report how the candidate's graph score and rank change.

    graph_scores takes the edges that remain and returns one graph score per candidate.
    The rank uses the final score alpha * cosine + (1 - alpha) * graph (ADR-0016).
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    all_edges = np.ones(edge_index.shape[1], dtype=bool)
    without_explanation = mark_remaining_edges(edge_index, explanation_edges, both_directions)
    # Keeping only the explanation is exactly the set of edges the removal took away.
    only_explanation = ~without_explanation

    full = graph_scores(all_edges)
    without = graph_scores(without_explanation)
    only = graph_scores(only_explanation)

    final_before = alpha * cosine + (1 - alpha) * full
    final_after = alpha * cosine + (1 - alpha) * without

    return FidelityResult(
        fidelity_plus=fidelity_plus(full[candidate], without[candidate]),
        fidelity_minus=fidelity_minus(full[candidate], only[candidate]),
        rank_drop=rank_drop(rank_of(final_before, candidate), rank_of(final_after, candidate)),
    )


def random_control(
    edge_index: np.ndarray,
    image_node: int,
    num_layers: int,
    both_directions: bool,
    num_removed: int,
    score_image: Callable[[np.ndarray], float],
    repeats: int,
    seed: int,
) -> np.ndarray:
    """Fidelity+ of removing num_removed random relations near the image, repeated.

    The relations are drawn from those that can change the image's score, so the control
    touches the same neighbourhood as the explanation. Returns one value per repeat.
    """
    relations = relations_in_reach(edge_index, image_node, num_layers, both_directions)
    if not 0 <= num_removed <= len(relations):
        raise ValueError(f"cannot remove {num_removed} of {len(relations)} relations")

    rng = np.random.default_rng(seed)
    full = score_image(np.ones(edge_index.shape[1], dtype=bool))
    values = np.empty(repeats)
    for repeat in range(repeats):
        picked = rng.choice(len(relations), size=num_removed, replace=False)
        removed = [relations[index] for index in picked]
        remaining = mark_remaining_edges(edge_index, removed, both_directions)
        values[repeat] = full - score_image(remaining)
    return values


def robust_fidelity(
    edge_index: np.ndarray,
    image_node: int,
    num_layers: int,
    explanation: list[tuple[int, int]],
    both_directions: bool,
    score_image: Callable[[np.ndarray], float],
    remove_prob: float,
    add_back_prob: float,
    repeats: int,
    seed: int,
) -> tuple[float, float]:
    """Robust fidelity+ and fidelity- (Zheng et al., ICLR 2024), averaged over repeats.

    Fidelity+ removes each explanation relation only with probability remove_prob;
    fidelity- keeps the explanation and adds back each other nearby relation with
    probability add_back_prob. Both keep the changed graph close to the original one,
    so the model is not judged on graphs unlike those it was trained on.
    """
    for name, value in (("remove_prob", remove_prob), ("add_back_prob", add_back_prob)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")

    explained = {_relation_key(edge, both_directions) for edge in explanation}
    others = [
        relation
        for relation in relations_in_reach(edge_index, image_node, num_layers, both_directions)
        if _relation_key(relation, both_directions) not in explained
    ]

    rng = np.random.default_rng(seed)
    full = score_image(np.ones(edge_index.shape[1], dtype=bool))
    plus_values = np.empty(repeats)
    minus_values = np.empty(repeats)
    for repeat in range(repeats):
        removed = [edge for edge in explanation if rng.random() < remove_prob]
        without = mark_remaining_edges(edge_index, removed, both_directions)
        plus_values[repeat] = full - score_image(without)

        kept = ~mark_remaining_edges(edge_index, explanation, both_directions)
        for relation in others:
            if rng.random() < add_back_prob:
                kept |= ~mark_remaining_edges(edge_index, [relation], both_directions)
        minus_values[repeat] = full - score_image(kept)

    return float(plus_values.mean()), float(minus_values.mean())


def _relation_key(edge: tuple[int, int], both_directions: bool):
    return frozenset(edge) if both_directions else edge


def normalized_fidelity(
    full: float, without: float, only: float, empty: float
) -> tuple[float, float] | None:
    """Fidelity+ and fidelity- as shares of everything the graph adds to the score, in [0, 1].

    empty is the score with no edges at all. Returns None when the edges do not change
    the score, because then no explanation can be measured.
    """
    graph_effect = full - empty
    if np.isclose(graph_effect, 0.0):
        return None
    plus = float(np.clip((full - without) / graph_effect, 0.0, 1.0))
    minus = float(np.clip((full - only) / graph_effect, 0.0, 1.0))
    return plus, minus


def characterization(fid_plus: float, fid_minus: float, plus_weight: float) -> float:
    """One score from normalized fidelity+ and fidelity- (GraphFramEx, Amara et al. 2022).

    Weighted harmonic mean of fidelity+ (higher is better) and 1 - fidelity- (higher is
    better), so it is high only when the explanation is both necessary and sufficient.
    """
    if not 0.0 <= plus_weight <= 1.0:
        raise ValueError(f"plus_weight must be in [0, 1], got {plus_weight}")
    sufficiency = 1.0 - fid_minus
    if fid_plus == 0.0 or sufficiency == 0.0:
        return 0.0
    minus_weight = 1.0 - plus_weight
    return 1.0 / (plus_weight / fid_plus + minus_weight / sufficiency)


@dataclass(frozen=True)
class ExplanationReport:
    """Every level-3 measure of one explanation for one image (PDF section 5)."""

    fidelity_plus: float
    fidelity_minus: float
    normalized_plus: float | None
    normalized_minus: float | None
    characterization: float | None
    robust_plus: float
    robust_minus: float
    random_plus_mean: float
    sparsity: float
    all_in_reach: bool
    connected_share: float


def evaluate_explanation(
    edge_index: np.ndarray,
    image_node: int,
    explanation: list[tuple[int, int]],
    score_image: Callable[[np.ndarray], float],
    config: ExplainConfig,
    seed: int,
) -> ExplanationReport:
    """Measure fidelity, robust fidelity, the random control, sparsity and validity."""
    layers = config.num_layers
    both = config.both_directions

    full = score_image(np.ones(edge_index.shape[1], dtype=bool))
    without_mask = mark_remaining_edges(edge_index, explanation, both)
    without = score_image(without_mask)
    only = score_image(~without_mask)
    empty = score_image(np.zeros(edge_index.shape[1], dtype=bool))

    normalized = normalized_fidelity(full, without, only, empty)
    if normalized is None:
        normalized_plus, normalized_minus, charact = None, None, None
    else:
        normalized_plus, normalized_minus = normalized
        charact = characterization(normalized_plus, normalized_minus, config.plus_weight)

    robust_plus, robust_minus = robust_fidelity(
        edge_index,
        image_node,
        layers,
        explanation,
        both,
        score_image,
        config.robust_remove_prob,
        config.robust_add_back_prob,
        config.robust_repeats,
        seed,
    )
    control = random_control(
        edge_index,
        image_node,
        layers,
        both,
        len(explanation),
        score_image,
        config.random_repeats,
        seed,
    )
    nearby = relations_in_reach(edge_index, image_node, layers, both)
    checked = validity(edge_index, explanation, image_node, layers, both)

    return ExplanationReport(
        fidelity_plus=fidelity_plus(full, without),
        fidelity_minus=fidelity_minus(full, only),
        normalized_plus=normalized_plus,
        normalized_minus=normalized_minus,
        characterization=charact,
        robust_plus=robust_plus,
        robust_minus=robust_minus,
        random_plus_mean=float(control.mean()),
        sparsity=sparsity(len(explanation), len(nearby)),
        all_in_reach=checked.all_in_reach,
        connected_share=checked.connected_share,
    )
