"""Fidelity: does removing or keeping only the explanation change the score as expected?"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from mmxai.explain.checks import check_score_function
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


def final_scores(cosine: np.ndarray, graph: np.ndarray, alpha: float) -> np.ndarray:
    """The score used for ranking: alpha * cosine + (1 - alpha) * graph (ADR-0016)."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    return alpha * cosine + (1 - alpha) * graph


def sample_relation_sets(
    relations: list[tuple[int, int]], size: int, repeats: int, seed: int
) -> list[list[tuple[int, int]]]:
    """Draw `repeats` random sets of `size` different relations, for the random control.

    A random set may include explanation relations; it is a plain same-size baseline.
    """
    if not 0 <= size <= len(relations):
        raise ValueError(f"cannot pick {size} of {len(relations)} relations")
    rng = np.random.default_rng(seed)
    sets = []
    for _ in range(repeats):
        picked = rng.choice(len(relations), size=size, replace=False)
        sets.append([relations[index] for index in picked])
    return sets


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


@dataclass(frozen=True)
class NormalizedFidelity:
    """Fidelity+ and fidelity- as shares of the graph's effect, in [0, 1]."""

    plus: float
    minus: float
    clipped: bool


def normalized_fidelity(
    full: float, without: float, only: float, empty: float
) -> NormalizedFidelity | None:
    """Fidelity+ and fidelity- as shares of everything the graph adds to the score.

    empty is the score with no edges at all. Values are clipped to [0, 1]; clipped says
    whether that happened (the graph's effects were not simply additive). Returns None
    when the edges do not change the score, because then nothing can be explained.
    This share is the team's adaptation: GraphFramEx uses class probabilities instead.
    """
    graph_effect = full - empty
    if np.isclose(graph_effect, 0.0):
        return None
    raw_plus = (full - without) / graph_effect
    raw_minus = (full - only) / graph_effect
    plus = float(np.clip(raw_plus, 0.0, 1.0))
    minus = float(np.clip(raw_minus, 0.0, 1.0))
    return NormalizedFidelity(plus, minus, clipped=(plus != raw_plus or minus != raw_minus))


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
    """Every level-3 measure of one explanation for one candidate (PDF section 5)."""

    fidelity_plus: float
    fidelity_minus: float
    rank_drop: int
    normalized_plus: float | None
    normalized_minus: float | None
    normalized_clipped: bool
    characterization: float | None
    robust_plus: float
    robust_minus: float
    random_plus_mean: float
    random_rank_drop_mean: float
    random_plus_mean_near: float | None
    sparsity: float
    sparsity_near: float | None
    all_in_reach: bool
    connected_share: float


def evaluate_explanation(
    edge_index: np.ndarray,
    image_node: int,
    candidate: int,
    explanation: list[tuple[int, int]],
    graph_scores: Callable[[np.ndarray], np.ndarray],
    ranked_graph_scores: np.ndarray,
    cosine: np.ndarray,
    alpha: float,
    config: ExplainConfig,
    seed: int,
) -> ExplanationReport | None:
    """Measure one explanation of one candidate: score and rank changes, the random
    control, robust fidelity, sparsity and validity.

    graph_scores takes the edges that remain and returns one graph score per candidate;
    ranked_graph_scores and cosine are the scores the ranking used. Returns None when
    alpha = 1, because then the graph plays no part in the ranking.
    Raises ValueError if graph_scores does not reproduce the ranking.
    """
    if alpha == 1.0:
        return None
    layers = config.num_layers
    both = config.both_directions

    def score_image(remaining: np.ndarray) -> float:
        return float(graph_scores(remaining)[candidate])

    all_edges = np.ones(edge_index.shape[1], dtype=bool)
    if not np.allclose(graph_scores(all_edges), ranked_graph_scores):
        raise ValueError("graph_scores does not reproduce the graph scores used for ranking")
    full = check_score_function(
        edge_index, image_node, layers, score_image, ranked_graph_scores[candidate]
    )
    rank_before = rank_of(final_scores(cosine, ranked_graph_scores, alpha), candidate)

    def drops(removed: list[tuple[int, int]]) -> tuple[float, int]:
        remaining = mark_remaining_edges(edge_index, removed, both)
        scores = graph_scores(remaining)
        rank_after = rank_of(final_scores(cosine, scores, alpha), candidate)
        return full - float(scores[candidate]), rank_drop(rank_before, rank_after)

    plus, explanation_rank_drop = drops(explanation)
    without_mask = mark_remaining_edges(edge_index, explanation, both)
    only = score_image(~without_mask)
    empty = score_image(np.zeros(edge_index.shape[1], dtype=bool))

    normalized = normalized_fidelity(full, full - plus, only, empty)
    charact = None
    if normalized is not None:
        charact = characterization(normalized.plus, normalized.minus, config.plus_weight)

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

    nearby = relations_in_reach(edge_index, image_node, layers, both)
    random_results = [
        drops(removed)
        for removed in sample_relation_sets(nearby, len(explanation), config.random_repeats, seed)
    ]

    # The neighbourhood without the extra hop added for degree-normalised models (GCN).
    near = relations_in_reach(edge_index, image_node, layers, both, extra_hop=False)
    random_plus_near = None
    sparsity_near = None
    if len(explanation) <= len(near) and near:
        near_sets = sample_relation_sets(near, len(explanation), config.random_repeats, seed)
        random_plus_near = float(np.mean([drops(removed)[0] for removed in near_sets]))
        sparsity_near = sparsity(len(explanation), len(near))

    checked = validity(edge_index, explanation, image_node, layers, both)
    return ExplanationReport(
        fidelity_plus=plus,
        fidelity_minus=fidelity_minus(full, only),
        rank_drop=explanation_rank_drop,
        normalized_plus=None if normalized is None else normalized.plus,
        normalized_minus=None if normalized is None else normalized.minus,
        normalized_clipped=False if normalized is None else normalized.clipped,
        characterization=charact,
        robust_plus=robust_plus,
        robust_minus=robust_minus,
        random_plus_mean=float(np.mean([result[0] for result in random_results])),
        random_rank_drop_mean=float(np.mean([result[1] for result in random_results])),
        random_plus_mean_near=random_plus_near,
        sparsity=sparsity(len(explanation), len(nearby)),
        sparsity_near=sparsity_near,
        all_in_reach=checked.all_in_reach,
        connected_share=checked.connected_share,
    )
