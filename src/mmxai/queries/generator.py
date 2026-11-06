"""Generate direct/indirect query pairs from real Visual Genome relation chains (ADR-0044).

A chain is k >= 2 annotated relations that link distinct objects of ONE image by
object id, e.g. (person -holding-> frisbee) and (dog -looks at-> frisbee) share the
frisbee node. From one chain we write two queries with the same answers:

    direct   names every object:   "A person is holding a frisbee, and a dog looks at the frisbee."
    indirect hides inner objects:  "A person is holding the thing that a dog looks at."

The indirect query can only be resolved by following the relations, which is the
"child - plays - football" kind of query the PDF (Task 2) asks for. (The frisbee
example is a made-up test fixture, not a claim about Visual Genome.)

Text comes from fixed templates, no language model: every word is traceable to an
annotation, and the generation rules are fixed in the config before any result.
"""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mmxai.data.scene_graph import Relation, SceneGraph, load_scene_graphs
from mmxai.data.splits import load_split
from mmxai.queries.schema import STATUS_CANDIDATE, write_jsonl
from mmxai.queries.validator import (
    hidden_names_in_text,
    images_matching_pattern,
    validate_queries,
    write_review_manifest,
)
from mmxai.utils.yaml_config import load_yaml_dataclass

# Predicates starting with one of these words describe a position, not an action,
# so they need "is" to read as a sentence: "a cup is on a table".
POSITION_WORDS = {
    "above",
    "across",
    "against",
    "along",
    "among",
    "around",
    "at",
    "atop",
    "behind",
    "below",
    "beneath",
    "beside",
    "between",
    "by",
    "in",
    "inside",
    "near",
    "next",
    "on",
    "outside",
    "over",
    "under",
    "underneath",
}

# Rejection reasons, counted so the report can say why chains were thrown away.
REJECT_GENERIC_NAME = "generic_name"
REJECT_TRIVIAL_PREDICATE = "trivial_predicate"
REJECT_AMBIGUOUS_NAME = "ambiguous_name"
REJECT_HIDDEN_NAME_IN_TEXT = "hidden_name_in_indirect_text"
REJECT_DUPLICATE_TEXT = "duplicate_query_text"
REJECT_TOO_MANY_MATCHES = "too_many_candidate_images"


@dataclass(frozen=True)
class QueryBenchmarkConfig:
    """Settings for generating candidates and building the verified benchmark."""

    dataset: str
    data_dir: str
    objects_file: str
    relationships_file: str
    split_file: str
    split: str
    seed: int
    limit_pairs: int
    min_hops: int
    max_hops: int
    max_pairs_per_image: int
    max_candidate_images: int
    generic_names: list[str]
    trivial_predicates: list[str]
    author: str
    candidates_path: str
    manifest_path: str
    benchmark_path: str
    min_verified_pairs: int


def load_query_benchmark_config(path: str | Path) -> QueryBenchmarkConfig:
    """Load the YAML config and reject settings that cannot give an indirect query."""
    config = load_yaml_dataclass(path, QueryBenchmarkConfig)
    if config.split not in ("val", "test"):
        raise ValueError(f"split must be val or test, got {config.split!r}")
    if config.min_hops < 2:
        raise ValueError("min_hops must be >= 2: an indirect query needs at least two relations")
    if config.max_hops < config.min_hops:
        raise ValueError("max_hops must be >= min_hops")
    if config.limit_pairs < 1 or config.max_pairs_per_image < 1:
        raise ValueError("limit_pairs and max_pairs_per_image must be >= 1")
    return config


@dataclass(frozen=True)
class Chain:
    """Relations r1..rk of one image; relation i links node_ids[i] and node_ids[i + 1]."""

    image_id: int
    node_ids: tuple[int, ...]
    relations: tuple[Relation, ...]

    @property
    def hop_count(self) -> int:
        return len(self.relations)

    def is_subject(self, position: int, node_position: int) -> bool:
        """Is node `node_position` the subject of relation `position`?"""
        return self.relations[position].subject_id == self.node_ids[node_position]


# --- Which relations and chains are usable ------------------------------------------------


def usable_relations(
    graph: SceneGraph,
    generic_names: list[str],
    trivial_predicates: list[str],
    rejections: Counter,
) -> list[Relation]:
    """Relations whose two objects and predicate can be named without ambiguity.

    Rules (ADR-0044), kept deliberately simple:
    - no generic name ("it", "thing"...): it cannot be checked in the image;
    - no trivial predicate ("of", "and"...): it is not a real relation;
    - each object's name is unique among ALL objects of the image, so "the dog"
      points to one object and the graph by id agrees with what a reader sees.
      It also means the nodes of a chain all have different names, so an indirect
      query can never be answered by a second, named copy of its hidden object.
    """
    generic = set(generic_names)
    trivial = set(trivial_predicates)
    counts = graph.name_counts()
    usable = []
    for relation in graph.relations:
        if relation.subject_name in generic or relation.object_name in generic:
            rejections[REJECT_GENERIC_NAME] += 1
        elif relation.predicate in trivial:
            rejections[REJECT_TRIVIAL_PREDICATE] += 1
        elif counts[relation.subject_name] > 1 or counts[relation.object_name] > 1:
            rejections[REJECT_AMBIGUOUS_NAME] += 1
        else:
            usable.append(relation)
    return usable


def enumerate_chains(
    image_id: int, relations: list[Relation], min_hops: int, max_hops: int
) -> list[Chain]:
    """All simple paths of min_hops..max_hops relations through distinct object ids.

    Walking forward or backward along an edge is allowed (the bridge can be subject
    or object). A path and its reverse use the same relations; only the first one
    found is kept. Start nodes and edges are visited in id order, so the result is
    deterministic.
    """
    edges_by_node: dict[int, list[Relation]] = {}
    for relation in sorted(relations, key=lambda item: item.relationship_id):
        edges_by_node.setdefault(relation.subject_id, []).append(relation)
        edges_by_node.setdefault(relation.object_id, []).append(relation)

    chains = []
    seen_relation_sets = set()

    def walk(node_ids: list[int], path: list[Relation]) -> None:
        if min_hops <= len(path):
            key = frozenset(relation.relationship_id for relation in path)
            if key not in seen_relation_sets:
                seen_relation_sets.add(key)
                chains.append(Chain(image_id, tuple(node_ids), tuple(path)))
        if len(path) == max_hops:
            return
        current = node_ids[-1]
        for relation in edges_by_node.get(current, []):
            # Edges are walked in both directions, so step to whichever end is not `current`.
            next_id = relation.object_id if relation.subject_id == current else relation.subject_id
            if next_id in node_ids:
                continue
            walk(node_ids + [next_id], path + [relation])

    for start in sorted(edges_by_node):
        walk([start], [])
    return chains


def node_names(chain: Chain, graph: SceneGraph) -> list[str]:
    return [graph.objects[node_id].name for node_id in chain.node_ids]


# --- Templates -------------------------------------------------------------------------------


def verb_phrase(predicate: str) -> str:
    """Make a predicate read as a present-tense verb: "holding" -> "is holding", "on" -> "is on".

    Predicates that already are a verb ("has", "wears", "looks at") stay unchanged.
    """
    first_word = predicate.split()[0]
    if first_word in ("is", "are"):
        return predicate
    if first_word.endswith("ing") or first_word.endswith("ed") or first_word in POSITION_WORDS:
        return f"is {predicate}"
    return predicate


def with_article(name: str) -> str:
    if name[0] in "aeiou":
        return f"an {name}"
    return f"a {name}"


def as_sentence(text: str) -> str:
    return text[0].upper() + text[1:] + "."


def direct_text(chain: Chain, graph: SceneGraph) -> str:
    """Name every object: "a" at first mention, "the" afterwards (same object id)."""
    mentioned = set()

    def mention(node_id: int) -> str:
        name = graph.objects[node_id].name
        if node_id in mentioned:
            return f"the {name}"
        mentioned.add(node_id)
        return with_article(name)

    clauses = []
    for relation in chain.relations:
        subject = mention(relation.subject_id)
        target = mention(relation.object_id)
        clauses.append(f"{subject} {verb_phrase(relation.predicate)} {target}")
    return as_sentence(", and ".join(clauses))


def describe_node(chain: Chain, graph: SceneGraph, node_position: int) -> str:
    """Noun phrase for a node: the last node is named, inner (bridge) nodes are hidden
    behind "the thing that ..." built from the next relation of the chain."""
    if node_position == len(chain.node_ids) - 1:
        return with_article(graph.objects[chain.node_ids[node_position]].name)
    relation = chain.relations[node_position]
    rest = describe_node(chain, graph, node_position + 1)
    verb = verb_phrase(relation.predicate)
    if chain.is_subject(node_position, node_position):
        return f"the thing that {verb} {rest}"
    return f"the thing that {rest} {verb}"


def indirect_text(chain: Chain, graph: SceneGraph) -> str:
    """Name the two end objects, describe every inner object only through its relations."""
    first = with_article(graph.objects[chain.node_ids[0]].name)
    rest = describe_node(chain, graph, 1)
    verb = verb_phrase(chain.relations[0].predicate)
    if chain.is_subject(0, 0):
        return as_sentence(f"{first} {verb} {rest}")
    return as_sentence(f"{rest} {verb} {first}")


def reasoning_type(chain: Chain) -> str:
    """Name the shape of a 2-hop chain by the bridge's role in both relations."""
    if chain.hop_count != 2:
        return "multi_hop_chain"
    bridge_is_subject_of_first = chain.is_subject(0, 1)
    bridge_is_subject_of_second = chain.is_subject(1, 1)
    if not bridge_is_subject_of_first and not bridge_is_subject_of_second:
        return "shared_object"  # A -p-> X <-q- B
    if bridge_is_subject_of_first and bridge_is_subject_of_second:
        return "shared_subject"  # A <-p- X -q-> B
    return "path"  # A -p-> X -q-> B


def path_hint(chain: Chain) -> str:
    """queries/README.md style: "person - holding - frisbee; dog - looks at - frisbee"."""
    parts = []
    for relation in chain.relations:
        parts.append(f"{relation.subject_name} - {relation.predicate} - {relation.object_name}")
    return "; ".join(parts)


def relation_evidence(chain: Chain) -> list[dict]:
    """The annotations behind the query, in chain order, with raw strings for reviewers."""
    evidence = []
    for relation in chain.relations:
        evidence.append(
            {
                "relationship_id": relation.relationship_id,
                "subject_id": relation.subject_id,
                "subject_name": relation.subject_name,
                "predicate": relation.predicate,
                "object_id": relation.object_id,
                "object_name": relation.object_name,
                "raw": [
                    relation.raw_subject_name,
                    relation.raw_predicate,
                    relation.raw_object_name,
                ],
            }
        )
    return evidence


def make_pair(
    chain: Chain,
    graph: SceneGraph,
    pair_id: str,
    relevant: list[int],
    config: QueryBenchmarkConfig,
) -> list[dict]:
    """The direct and the indirect record of one chain; both share pair_id and answers."""
    names = node_names(chain, graph)
    shared = {
        "dataset": config.dataset,
        "relevant": [str(image_id) for image_id in relevant],
        "path_hint": path_hint(chain),
        "author": config.author,
        "pair_id": pair_id,
        "source_image_id": str(chain.image_id),
        "relation_evidence": relation_evidence(chain),
        "chain_object_ids": list(chain.node_ids),
        "hop_count": chain.hop_count,
        "split": config.split,
        "verification_status": STATUS_CANDIDATE,
    }
    direct = {"qid": f"{pair_id}-d", "text": direct_text(chain, graph), "type": "direct"}
    direct.update(shared)
    direct["reasoning_type"] = "explicit"
    direct["hidden_names"] = []
    indirect = {"qid": f"{pair_id}-i", "text": indirect_text(chain, graph), "type": "indirect"}
    indirect.update(shared)
    indirect["reasoning_type"] = reasoning_type(chain)
    indirect["hidden_names"] = names[1:-1]
    return [direct, indirect]


def chain_problem(chain: Chain, graph: SceneGraph, seen_texts: set[str]) -> str | None:
    """Return why a chain cannot become a query pair, or None if it can (cheap checks only)."""
    # No check for two chain nodes with one name: usable_relations already keeps only
    # names that are unique in the image, so every node of a chain has its own name.
    names = node_names(chain, graph)
    text = indirect_text(chain, graph)
    # Naming the hidden object would turn the indirect query into a reworded direct one.
    if hidden_names_in_text(text, names[1:-1]):
        return REJECT_HIDDEN_NAME_IN_TEXT
    if text in seen_texts:
        return REJECT_DUPLICATE_TEXT
    return None


def chain_pattern(chain: Chain, graph: SceneGraph) -> tuple[list[str], list[tuple[str, bool]]]:
    """Same pattern the validator rebuilds from the stored evidence."""
    steps = []
    for position, relation in enumerate(chain.relations):
        steps.append((relation.predicate, chain.is_subject(position, position)))
    return node_names(chain, graph), steps


def generate_pairs(
    graphs: dict[int, SceneGraph], config: QueryBenchmarkConfig
) -> tuple[list[dict], Counter]:
    """Generate up to config.limit_pairs candidate pairs from the given split's graphs.

    Images are visited in a seeded random order and at most max_pairs_per_image
    chains are drawn per image, which spreads the benchmark over many images.
    Candidate answers are searched in `graphs`, which must hold every image of the
    split and nothing else. Returns (records, rejection counts).
    """
    rng = np.random.default_rng(config.seed)
    image_order = sorted(graphs)
    rng.shuffle(image_order)

    rejections: Counter = Counter()
    records: list[dict] = []
    seen_texts: set[str] = set()
    num_pairs = 0
    for image_id in image_order:
        if num_pairs >= config.limit_pairs:
            break
        graph = graphs[image_id]
        relations = usable_relations(
            graph, config.generic_names, config.trivial_predicates, rejections
        )
        chains = enumerate_chains(image_id, relations, config.min_hops, config.max_hops)
        if not chains:
            continue
        # Random order inside the image, then keep the first acceptable chains.
        taken = 0
        for position in rng.permutation(len(chains)).tolist():
            if taken >= config.max_pairs_per_image or num_pairs >= config.limit_pairs:
                break
            chain = chains[position]
            problem = chain_problem(chain, graph, seen_texts)
            if problem is not None:
                rejections[problem] += 1
                continue
            names, steps = chain_pattern(chain, graph)
            matches = images_matching_pattern(names, steps, graphs)
            # Fixed before generation: a pattern found in very many images is too
            # generic to review by hand and says little about relational reasoning.
            if len(matches) > config.max_candidate_images:
                rejections[REJECT_TOO_MANY_MATCHES] += 1
                continue
            if image_id not in matches:
                raise RuntimeError(f"chain of image {image_id} does not match its own graph")
            num_pairs += 1
            pair_id = f"vg-{config.split}-{num_pairs:03d}"
            records.extend(make_pair(chain, graph, pair_id, matches, config))
            seen_texts.add(indirect_text(chain, graph))
            taken += 1
    return records, rejections


def run_generation(config: QueryBenchmarkConfig) -> dict:
    """Load the split's scene graphs, generate candidates, check them, write JSONL + manifest.

    Only images of config.split are loaded, so a test benchmark never looks at
    train or val annotations, and a val pilot never touches test.
    Returns a summary for the CLI to print.
    """
    split = load_split(config.split_file)
    split_ids = {int(image_id) for image_id in split[config.split].tolist()}
    graphs, parse_report = load_scene_graphs(
        config.data_dir, split_ids, config.objects_file, config.relationships_file
    )
    records, rejections = generate_pairs(graphs, config)
    if not records:
        raise ValueError("no candidate pair survived the filters; nothing written")

    # Self-check against the annotations before anything reaches a reviewer.
    split_id_strings = {str(image_id) for image_id in split_ids}
    problems = validate_queries(records, split_id_strings, graphs, expected_split=config.split)
    if problems:
        raise ValueError("generated candidates are invalid:\n" + "\n".join(problems))

    write_jsonl(records, config.candidates_path)
    write_review_manifest(records, config.manifest_path)
    return {
        "split": config.split,
        "num_split_images": len(split_ids),
        "num_scene_graphs": len(graphs),
        "num_pairs": len(records) // 2,
        "num_queries": len(records),
        "dropped_records": dict(sorted(parse_report.items())),
        "rejected_chains": dict(sorted(rejections.items())),
        "candidates_path": config.candidates_path,
        "manifest_path": config.manifest_path,
    }
