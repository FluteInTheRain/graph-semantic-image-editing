"""Which images of a split show a concept: an activity a person plays, or an object.

These sets are the candidate answers of a KG query (ADR-0050), and their sizes on
train decide which concepts become targets. Predicates must already be canonical
(mmxai.queries.generator.canonicalize_graphs), so "plays" counts as "playing".
"""

from mmxai.data.scene_graph import SceneGraph


def activity_images(
    graphs: dict[int, SceneGraph], person_synsets: list[str], activity_predicates: list[str]
) -> dict[str, set[int]]:
    """Synset of the played thing -> images where a person plays it."""
    persons = set(person_synsets)
    predicates = set(activity_predicates)
    images: dict[str, set[int]] = {}
    for image_id, graph in graphs.items():
        for relation in graph.relations:
            if relation.predicate not in predicates:
                continue
            subject = graph.objects[relation.subject_id]
            target = graph.objects[relation.object_id]
            if subject.synset in persons and target.synset:
                images.setdefault(target.synset, set()).add(image_id)
    return images


def object_images(graphs: dict[int, SceneGraph]) -> dict[str, set[int]]:
    """Object synset -> images that contain such an object."""
    images: dict[str, set[int]] = {}
    for image_id, graph in graphs.items():
        for scene_object in graph.objects.values():
            if scene_object.synset:
                images.setdefault(scene_object.synset, set()).add(image_id)
    return images
