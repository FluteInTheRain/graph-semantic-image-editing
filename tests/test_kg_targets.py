"""Concept -> images on scene graphs (ADR-0050). Hand-built synthetic graphs."""

from mmxai.data.scene_graph import Relation, SceneGraph, SceneObject
from mmxai.queries.kg_targets import activity_images, object_images


def graph(image_id: int, people: str, predicate: str, thing: str) -> SceneGraph:
    subject = SceneObject(1, people.split(".")[0], people.split(".")[0], people)
    target = SceneObject(2, thing.split(".")[0], thing.split(".")[0], thing)
    scene = SceneGraph(image_id, {1: subject, 2: target})
    relation = Relation(
        10, 1, subject.name, predicate, 2, target.name, predicate, subject.name, target.name
    )
    scene.relations.append(relation)
    return scene


def test_activity_needs_a_person_subject_and_the_playing_predicate():
    graphs = {
        1: graph(1, "man.n.01", "playing", "tennis.n.01"),
        2: graph(2, "dog.n.01", "playing", "frisbee.n.01"),  # not a person
        3: graph(3, "girl.n.01", "holding", "tennis.n.01"),  # not playing
    }

    images = activity_images(graphs, ["man.n.01", "girl.n.01"], ["playing"])

    assert images == {"tennis.n.01": {1}}


def test_object_images_use_every_object_synset():
    graphs = {1: graph(1, "man.n.01", "near", "train.n.01")}

    assert object_images(graphs) == {"man.n.01": {1}, "train.n.01": {1}}
