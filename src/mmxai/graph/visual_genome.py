"""Visual Genome part of the graph build: which concepts each image contains (ADR-0045).

This is the only dataset-specific step. It returns the plain {image_id: concepts}
mapping that mmxai.graph.contains turns into a graph; Flickr30k will get its own
module returning the same mapping.
"""

import numpy as np

from mmxai.data.scene_graph import SceneGraph


def build_name_to_synset(graphs: dict[int, SceneGraph], train_ids: np.ndarray) -> dict[str, str]:
    """Learn name -> synset from TRAIN objects that have a synset; keep the most common one.

    Decision (ADR-0045): fill missing synsets by name, with a dictionary counted on train.
      Why: 13,320 VG images list objects ("couch", "table") with an empty synsets field,
      so without a lookup they get no edge at all. The dictionary is a statistic, so
      test images never vote on it; train_ids has no default on purpose.
      Decided by: qphulong, 2026-11-07 (proposed by Claude Code).
    """
    counts: dict[str, dict[str, int]] = {}
    for image_id in train_ids.tolist():
        graph = graphs.get(image_id)
        if graph is None:
            continue
        for scene_object in graph.objects.values():
            if not scene_object.synset:
                continue
            synset_counts = counts.setdefault(scene_object.name, {})
            synset_counts[scene_object.synset] = synset_counts.get(scene_object.synset, 0) + 1

    name_to_synset = {}
    for name, synset_counts in counts.items():
        # Highest count wins; on a tie the alphabetically first synset, so reruns agree.
        best = sorted(synset_counts.items(), key=lambda item: (-item[1], item[0]))[0]
        name_to_synset[name] = best[0]
    return name_to_synset


def image_concepts_from_scene_graphs(
    graphs: dict[int, SceneGraph], name_to_synset: dict[str, str]
) -> tuple[dict[int, set[str]], dict[str, int]]:
    """Return ({image_id: set of synsets}, how many objects got their synset each way).

    An object keeps the synset VG annotated; without one it is looked up by name in
    name_to_synset; an unknown name is skipped. Counts: "annotated", "from_name",
    "unmapped".

    Decision (ADR-0045): one concept per WordNet synset.
      Why: the synset joins "dog" and "puppy" into one node without a hand-made
      synonym list. Using raw names would give "man" and "men" two nodes.
      Proposed by Claude Code, 2026-11-07; not yet decided by the team.
    """
    image_concepts: dict[int, set[str]] = {}
    counts = {"annotated": 0, "from_name": 0, "unmapped": 0}
    for image_id, graph in graphs.items():
        concepts = set()
        for scene_object in graph.objects.values():
            if scene_object.synset:
                concepts.add(scene_object.synset)
                counts["annotated"] += 1
            elif scene_object.name in name_to_synset:
                concepts.add(name_to_synset[scene_object.name])
                counts["from_name"] += 1
            else:
                counts["unmapped"] += 1
        image_concepts[image_id] = concepts
    return image_concepts, counts


def concept_text(synset: str) -> str:
    """The words CLIP reads for a concept: "fire_hydrant.n.01" -> "fire hydrant"."""
    # A synset is "<lemma>.<pos>.<sense>" and the lemma may contain dots itself
    # ("st._bernard.n.01"), so only the last two dots are separators.
    lemma = synset.rsplit(".", 2)[0]
    return lemma.replace("_", " ")
