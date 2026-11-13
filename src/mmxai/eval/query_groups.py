"""Query groups for text-to-image evaluation: specificity and relation type.

Each query gets one label per grouping, and metrics are reported per label.

Decision (spec 2026-11-13): specificity = "specific" if the query has exactly one
relevant image, otherwise "generic".
  Why: a query with many relevant images is easier to hit, so the two kinds
  should not be averaged together unseen.
  Decided by: FluteInTheRain, 2026-11-13 (proposed by Claude Code in the level-1
  evaluation extensions spec).

Decision (spec 2026-11-13): relation type = "action" if the sentence contains an
action predicate, else "spatial" if it contains a spatial predicate, else "none".
Predicates come from configs/query_benchmark_vg.yaml, matched as whole consecutive words.
  Why: reuse the predicate lists the team already fixed for the query benchmark.
  Decided by: FluteInTheRain, 2026-11-13 (proposed by Claude Code in the level-1
  evaluation extensions spec).
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from mmxai.eval.metrics import ranked_metrics


def specificity_groups(relevant: Sequence[set[int]]) -> list[str]:
    """Label each query "specific" (one relevant image) or "generic" (more)."""
    labels = []
    for relevant_images in relevant:
        if len(relevant_images) == 1:
            labels.append("specific")
        else:
            labels.append("generic")
    return labels


@dataclass(frozen=True)
class Predicates:
    """Predicates as tuples of words, e.g. ("next", "to")."""

    action: list[tuple[str, ...]]
    spatial: list[tuple[str, ...]]


def load_predicates(
    path: str | Path,
    extra_action: list[str] | None,
    extra_spatial: list[str] | None,
    excluded: list[str] | None,
) -> Predicates:
    """Read the predicate lists of the query benchmark config.

    Action = action_predicates plus their aliases; spatial = the other template
    keys plus their aliases. Extras are added, excluded predicates are removed.
    """
    with open(path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    action_names = list(config["action_predicates"])
    spatial_names = []
    for name in config["predicate_templates"]:
        if name not in action_names:
            spatial_names.append(name)

    # An alias joins the group of the predicate it stands for.
    for alias, canonical in config["predicate_aliases"].items():
        if canonical in action_names:
            action_names.append(alias)
        else:
            spatial_names.append(alias)

    action_names += extra_action or []
    spatial_names += extra_spatial or []

    excluded = excluded or []
    action = [tuple(name.split()) for name in action_names if name not in excluded]
    spatial = [tuple(name.split()) for name in spatial_names if name not in excluded]
    return Predicates(action=action, spatial=spatial)


def _contains(words: list[str], predicate: tuple[str, ...]) -> bool:
    """True if the predicate's words appear next to each other, in order."""
    size = len(predicate)
    for start in range(len(words) - size + 1):
        if tuple(words[start : start + size]) == predicate:
            return True
    return False


def relation_group(text: str, predicates: Predicates) -> str:
    """Label a sentence "action", "spatial" or "none"."""
    words = re.sub("[^a-z]", " ", text.lower()).split()
    for predicate in predicates.action:
        if _contains(words, predicate):
            return "action"
    for predicate in predicates.spatial:
        if _contains(words, predicate):
            return "spatial"
    return "none"


def relation_groups(
    text_ids: Sequence[int], region_texts: dict[int, str], predicates: Predicates
) -> list[str]:
    """Relation label of each query, in the order of text_ids."""
    labels = []
    for text_id in text_ids:
        if text_id not in region_texts:
            raise KeyError(f"no text for region id {text_id}")
        labels.append(relation_group(region_texts[text_id], predicates))
    return labels


def metrics_by_group(
    rankings: Sequence[Sequence[int]],
    relevant: Sequence[set[int]],
    labels: Sequence[str],
    recall_ks: Sequence[int],
    top_k: int,
    ndcg_k: int,
) -> dict[str, dict]:
    """ranked_metrics of each group, plus the share of queries in that group."""
    if not (len(rankings) == len(relevant) == len(labels)):
        raise ValueError("need one label and one relevant set per ranked list")

    result = {}
    for group in sorted(set(labels)):
        rows = [i for i, label in enumerate(labels) if label == group]
        group_metrics = ranked_metrics(
            [rankings[i] for i in rows],
            [relevant[i] for i in rows],
            recall_ks,
            top_k,
            ndcg_k,
        )
        group_metrics["fraction_queries"] = len(rows) / len(labels)
        result[group] = group_metrics
    return result


def group_sizes(labels: Sequence[str]) -> dict[str, tuple[int, float]]:
    """Return (count, fraction of all queries) of every label that occurs."""
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return {label: (count, count / len(labels)) for label, count in counts.items()}


def group_samples(
    labels: Sequence[str], sentences: Sequence[str], per_group: int, seed: int
) -> dict[str, list[str]]:
    """Pick up to per_group random sentences of each label, to read them by eye.

    labels[i] belongs to sentences[i]. A fixed seed makes the check repeatable.
    """
    rng = np.random.default_rng(seed)
    samples = {}
    for label in sorted(set(labels)):
        rows = [row for row in range(len(labels)) if labels[row] == label]
        chosen = rng.choice(rows, size=min(per_group, len(rows)), replace=False)
        samples[label] = [sentences[row] for row in chosen]
    return samples
