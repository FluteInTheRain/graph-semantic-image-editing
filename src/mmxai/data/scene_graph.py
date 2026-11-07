"""Read Visual Genome scene graphs (objects.json + relationships.json) into small per-image graphs.

A scene graph describes one image: object nodes (keyed by VG object_id) and directed
relation edges (subject -predicate-> object). Two objects are never merged just
because they share a name: "two dogs" stay two nodes, which the query generator
needs to detect ambiguous references (ADR-0044).

Normalised names are used for matching; raw strings are kept for human review.
Records that cannot be trusted are dropped and counted by reason, never repaired.
"""

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

OBJECTS_FILE = "objects.json"
RELATIONSHIPS_FILE = "relationships.json"

# Characters stripped from both ends of a name; VG annotators sometimes typed "dog." or "'hat'".
EDGE_PUNCTUATION = " .,;:!?\"'`"

# Reasons a record is dropped. Counting them makes data problems visible in the report.
DROP_MISSING_IMAGE_ID = "missing_image_id"
DROP_MISSING_OBJECT_ID = "missing_object_id"
DROP_EMPTY_NAME = "empty_name"
DROP_DUPLICATE_OBJECT_ID = "duplicate_object_id"
DROP_MISSING_RELATIONSHIP_ID = "missing_relationship_id"
DROP_MISSING_ENDPOINT = "missing_subject_or_object"
DROP_EMPTY_PREDICATE = "empty_predicate"
DROP_SELF_LOOP = "self_loop"
DROP_DUPLICATE_RELATIONSHIP_ID = "duplicate_relationship_id"
DROP_DUPLICATE_TRIPLE = "duplicate_triple"
# Not a drop: a relation endpoint absent from objects.json is added from the relation itself.
ADDED_FROM_RELATIONSHIP = "objects_added_from_relationships"


def normalize_text(text: str) -> str:
    """Lowercase, turn underscores into spaces, collapse whitespace and trim edge punctuation."""
    text = text.lower().replace("_", " ")
    text = " ".join(text.split())
    return text.strip(EDGE_PUNCTUATION)


@dataclass(frozen=True)
class SceneObject:
    """One annotated object; `name` is normalised, `raw_name` is what the annotator typed.

    `synset` is the first WordNet synset VG attached to the object ("man.n.01"), or "".
    """

    object_id: int
    name: str
    raw_name: str
    synset: str = ""


@dataclass(frozen=True)
class Relation:
    """One directed relation subject -predicate-> object, with ids into the scene graph."""

    relationship_id: int
    subject_id: int
    subject_name: str
    predicate: str
    object_id: int
    object_name: str
    raw_predicate: str
    raw_subject_name: str
    raw_object_name: str


@dataclass
class SceneGraph:
    """All objects and relations of one image."""

    image_id: int
    objects: dict[int, SceneObject] = field(default_factory=dict)
    relations: list[Relation] = field(default_factory=list)

    def name_counts(self) -> Counter:
        """How many distinct objects carry each normalised name in this image."""
        return Counter(scene_object.name for scene_object in self.objects.values())

    def relation_by_id(self, relationship_id: int) -> Relation | None:
        for relation in self.relations:
            if relation.relationship_id == relationship_id:
                return relation
        return None


def first_raw_name(annotation: dict) -> str:
    """VG stores one name as "name" or several as "names"; the first listed name is used."""
    if isinstance(annotation.get("name"), str):
        return annotation["name"]
    names = annotation.get("names")
    if isinstance(names, list) and names and isinstance(names[0], str):
        return names[0]
    return ""


def first_synset(annotation: dict) -> str:
    """First WordNet synset of an object annotation, or "" when VG gives none."""
    synsets = annotation.get("synsets")
    if isinstance(synsets, list) and synsets and isinstance(synsets[0], str):
        return synsets[0]
    return ""


def match_key(scene_object: SceneObject) -> str:
    """Key that decides whether objects of two images are the same kind of thing (ADR-0044).

    The synset comes from the VG annotation itself, so "man" and "guy" can match
    without a hand-made synonym list. Objects without a synset fall back to their name.
    """
    if scene_object.synset:
        return scene_object.synset
    return scene_object.name


def entry_image_id(entry: dict) -> int | None:
    """Image id of one top-level entry. Older VG files call it "id" instead of "image_id"."""
    image_id = entry.get("image_id", entry.get("id"))
    if isinstance(image_id, int):
        return image_id
    return None


def parse_object(annotation: dict, report: Counter) -> SceneObject | None:
    """Turn one object annotation into a SceneObject, or count why it was dropped."""
    object_id = annotation.get("object_id")
    if not isinstance(object_id, int):
        report[DROP_MISSING_OBJECT_ID] += 1
        return None
    raw_name = first_raw_name(annotation)
    name = normalize_text(raw_name)
    if not name:
        report[DROP_EMPTY_NAME] += 1
        return None
    return SceneObject(object_id, name, raw_name, first_synset(annotation))


def add_objects(graph: SceneGraph, objects_entry: dict, report: Counter) -> None:
    """Add every valid object of one objects.json entry; a repeated object_id keeps the first."""
    for annotation in objects_entry.get("objects", []):
        scene_object = parse_object(annotation, report)
        if scene_object is None:
            continue
        if scene_object.object_id in graph.objects:
            report[DROP_DUPLICATE_OBJECT_ID] += 1
            continue
        graph.objects[scene_object.object_id] = scene_object


def add_relations(graph: SceneGraph, relationships_entry: dict, report: Counter) -> None:
    """Add every valid relation of one relationships.json entry.

    Relations reuse the object ids of objects.json. VG merged some objects after
    annotation, so an endpoint may be missing from objects.json; it is then added
    from the relation's own copy of the object instead of dropping the relation.
    """
    seen_ids = {relation.relationship_id for relation in graph.relations}
    seen_triples = {
        (relation.subject_id, relation.predicate, relation.object_id)
        for relation in graph.relations
    }
    for annotation in relationships_entry.get("relationships", []):
        relationship_id = annotation.get("relationship_id")
        if not isinstance(relationship_id, int):
            report[DROP_MISSING_RELATIONSHIP_ID] += 1
            continue
        subject_annotation = annotation.get("subject")
        object_annotation = annotation.get("object")
        if not isinstance(subject_annotation, dict) or not isinstance(object_annotation, dict):
            report[DROP_MISSING_ENDPOINT] += 1
            continue
        raw_predicate = annotation.get("predicate")
        if not isinstance(raw_predicate, str) or not normalize_text(raw_predicate):
            report[DROP_EMPTY_PREDICATE] += 1
            continue

        subject = parse_object(subject_annotation, report)
        target = parse_object(object_annotation, report)
        if subject is None or target is None:
            continue
        if subject.object_id == target.object_id:
            report[DROP_SELF_LOOP] += 1
            continue
        if relationship_id in seen_ids:
            report[DROP_DUPLICATE_RELATIONSHIP_ID] += 1
            continue
        predicate = normalize_text(raw_predicate)
        triple = (subject.object_id, predicate, target.object_id)
        if triple in seen_triples:
            # The same fact typed twice would let one annotation count as two hops.
            report[DROP_DUPLICATE_TRIPLE] += 1
            continue

        for endpoint in (subject, target):
            if endpoint.object_id not in graph.objects:
                graph.objects[endpoint.object_id] = endpoint
                report[ADDED_FROM_RELATIONSHIP] += 1
        # Names come from objects.json when it has the object, so one id has one name.
        subject_node = graph.objects[subject.object_id]
        target_node = graph.objects[target.object_id]
        relation = Relation(
            relationship_id=relationship_id,
            subject_id=subject_node.object_id,
            subject_name=subject_node.name,
            predicate=predicate,
            object_id=target_node.object_id,
            object_name=target_node.name,
            raw_predicate=raw_predicate,
            raw_subject_name=subject_node.raw_name,
            raw_object_name=target_node.raw_name,
        )
        graph.relations.append(relation)
        seen_ids.add(relationship_id)
        seen_triples.add(triple)


def build_scene_graphs(
    objects_entries: list[dict],
    relationships_entries: list[dict],
    image_ids: set[int] | None,
) -> tuple[dict[int, SceneGraph], Counter]:
    """Build one SceneGraph per image from already loaded JSON lists.

    image_ids restricts the result to one split (None keeps every image), so a
    caller working on test never builds graphs of train images by accident.
    Returns (graphs by image id, counts of dropped records by reason).
    """
    report: Counter = Counter()
    graphs: dict[int, SceneGraph] = {}

    for entry in objects_entries:
        image_id = entry_image_id(entry)
        if image_id is None:
            report[DROP_MISSING_IMAGE_ID] += 1
            continue
        if image_ids is not None and image_id not in image_ids:
            continue
        graph = graphs.setdefault(image_id, SceneGraph(image_id))
        add_objects(graph, entry, report)

    for entry in relationships_entries:
        image_id = entry_image_id(entry)
        if image_id is None:
            report[DROP_MISSING_IMAGE_ID] += 1
            continue
        if image_ids is not None and image_id not in image_ids:
            continue
        graph = graphs.setdefault(image_id, SceneGraph(image_id))
        add_relations(graph, entry, report)

    return graphs, report


def load_scene_graphs(
    data_dir: str | Path,
    image_ids: set[int] | None,
    objects_file: str = OBJECTS_FILE,
    relationships_file: str = RELATIONSHIPS_FILE,
) -> tuple[dict[int, SceneGraph], Counter]:
    """Read objects.json and relationships.json from data_dir and build the scene graphs.

    Both files are required: objects.json gives every object of the image, which
    is needed to tell whether a name like "dog" points to a single object.
    relationships.json is large (about 700 MB for v1.4), so this needs several GB of RAM.
    """
    data_dir = Path(data_dir)
    objects_path = data_dir / objects_file
    relationships_path = data_dir / relationships_file
    for path in (objects_path, relationships_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Download the Visual Genome objects and relationships "
                "annotations (see data/README.md and docs/query_benchmark.md) into "
                f"{data_dir}, or point data_dir in the config to them."
            )
    with open(objects_path, encoding="utf-8") as file:
        objects_entries = json.load(file)
    with open(relationships_path, encoding="utf-8") as file:
        relationships_entries = json.load(file)
    return build_scene_graphs(objects_entries, relationships_entries, image_ids)
