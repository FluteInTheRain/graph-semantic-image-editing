"""Generate KG direct/indirect query pairs for Visual Genome (ADR-0050).

For each target concept (an activity a person plays, or an object), the direct
query names it and the indirect query only describes it through ConceptNet facts
that fit no other concept of its group. Answers are the images of ONE split that
show the concept in their annotations; targets and the comparison pool come from
TRAIN counts only, so test is used for nothing but the answers.
"""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from mmxai.data.conceptnet import ConceptEdge, load_edges_tsv
from mmxai.data.scene_graph import SceneGraph, build_scene_graphs, load_annotation_files
from mmxai.data.splits import load_split
from mmxai.queries.generator import PLAIN_WORDS, canonicalize_graphs, synset_lemma
from mmxai.queries.kg_facts import Description, Fact, build_fact_index, choose_description
from mmxai.queries.kg_targets import activity_images, object_images
from mmxai.queries.kg_text import direct_text, indirect_text
from mmxai.queries.schema import STATUS_CANDIDATE, write_jsonl
from mmxai.queries.validator import check_kg_evidence, validate_queries, write_review_manifest
from mmxai.utils.yaml_config import load_yaml_dataclass

FACT_KINDS = ("uses", "location", "has", "can")

REJECT_TOO_RARE = "too_few_train_images"
REJECT_NOT_PLAIN = "name_not_plain_words"
REJECT_NO_FACTS = "no_conceptnet_facts"
REJECT_ANSWERS = "answer_count_out_of_range"
REJECT_NO_DESCRIPTION = "no_unique_description"
REJECT_USED = "concept_already_used"
REJECT_AMBIGUOUS_TERM = "term_has_several_synsets"


@dataclass(frozen=True)
class KgQueryConfig:
    dataset: str
    data_dir: str
    objects_file: str
    relationships_file: str
    split_file: str
    split: str
    conceptnet_edges: str
    predicate_aliases: dict[str, str]
    activity_predicates: list[str]
    person_synsets: list[str]
    category_words: list[str]
    activity_templates: dict[str, str]
    object_templates: dict[str, str]
    num_properties: int
    min_train_images: int
    min_answer_images: int
    max_answer_images: int
    limit_activity_pairs: int
    limit_object_pairs: int
    author: str
    candidates_path: str
    manifest_path: str
    benchmark_path: str
    min_verified_pairs: int


def load_kg_query_config(path: str | Path) -> KgQueryConfig:
    config = load_yaml_dataclass(path, KgQueryConfig)
    if config.split not in ("val", "test"):
        raise ValueError(f"split must be val or test, got {config.split!r}")
    if config.num_properties < 1:
        raise ValueError("num_properties must be >= 1")
    for name in ("activity_templates", "object_templates"):
        missing = sorted(set(FACT_KINDS) - set(getattr(config, name)))
        if missing:
            raise ValueError(f"{name} has no template for {missing}")
    return config


def edge_record(edge: ConceptEdge) -> dict:
    return {
        "uri": edge.uri,
        "rel": edge.rel,
        "start": edge.start,
        "end": edge.end,
        "weight": edge.weight,
    }


def description_facts(description: Description) -> list[Fact]:
    return [description.category, *description.properties]


def make_kg_pair(
    group: str,
    synset: str,
    description: Description,
    answers: set[int],
    pair_id: str,
    templates: dict[str, str],
    config: KgQueryConfig,
) -> list[dict]:
    """The direct and indirect record of one concept; both share pair_id and answers."""
    facts = description_facts(description)
    hints = []
    for fact in facts:
        hints.append(f"{fact.edge.start} - {fact.edge.rel} - {fact.edge.end}")
    shared = {
        "dataset": config.dataset,
        "relevant": [str(image_id) for image_id in sorted(answers)],
        "path_hint": "; ".join(hints),
        "author": config.author,
        "pair_id": pair_id,
        "split": config.split,
        "verification_status": STATUS_CANDIDATE,
        "hidden_concept": synset,
        "query_group": group,
        "kg_evidence": [edge_record(fact.edge) for fact in facts],
    }
    direct = {"qid": f"{pair_id}-d", "text": direct_text(group, description.concept)}
    direct.update(type="direct", reasoning_type="explicit", hidden_names=[])
    direct.update(shared)
    indirect = {"qid": f"{pair_id}-i", "text": indirect_text(group, description, templates)}
    indirect.update(type="indirect", reasoning_type="kg_description")
    indirect["hidden_names"] = [description.concept]
    indirect.update(shared)
    return [direct, indirect]


def generate_kg_pairs(
    train_graphs: dict[int, SceneGraph],
    eval_graphs: dict[int, SceneGraph],
    fact_index: dict[str, list[Fact]],
    config: KgQueryConfig,
) -> tuple[list[dict], Counter]:
    """Pairs for the activity group, then the object group. Graphs must be canonical.

    Targets are visited from the most to the least frequent on train, so the
    result does not depend on any random seed.
    """
    groups = [
        (
            "activity",
            activity_images(train_graphs, config.person_synsets, config.activity_predicates),
            activity_images(eval_graphs, config.person_synsets, config.activity_predicates),
            config.activity_templates,
            config.limit_activity_pairs,
        ),
        (
            "object",
            object_images(train_graphs),
            object_images(eval_graphs),
            config.object_templates,
            config.limit_object_pairs,
        ),
    ]
    rejections: Counter = Counter()
    records: list[dict] = []
    used: set[str] = set()
    for group, train_images, eval_images, templates, limit in groups:
        pool = sorted({synset_lemma(synset) for synset in train_images})
        # Synsets per term: "bat" as animal and as club would give two pairs with the same
        # text but different answers, and the pool (by term) cannot tell them apart.
        synsets_per_term = Counter(synset_lemma(synset) for synset in train_images)

        # Most frequent first; the synset name breaks ties so the order is deterministic.
        ordered = sorted(train_images.items(), key=lambda item: (-len(item[1]), item[0]))
        num_pairs = 0
        for synset, images in ordered:
            if num_pairs >= limit:
                break
            if len(images) < config.min_train_images:
                rejections[REJECT_TOO_RARE] += 1
                continue
            term = synset_lemma(synset)
            if synsets_per_term[term] > 1:
                rejections[REJECT_AMBIGUOUS_TERM] += 1
                continue
            if synset in used or term in used:
                rejections[REJECT_USED] += 1
                continue
            if not PLAIN_WORDS.fullmatch(term):
                rejections[REJECT_NOT_PLAIN] += 1
                continue
            if term not in fact_index:
                rejections[REJECT_NO_FACTS] += 1
                continue
            answers = eval_images.get(synset, set())
            if not config.min_answer_images <= len(answers) <= config.max_answer_images:
                rejections[REJECT_ANSWERS] += 1
                continue
            description = choose_description(
                term, pool, fact_index, config.category_words, config.num_properties
            )
            if description is None:
                rejections[REJECT_NO_DESCRIPTION] += 1
                continue
            num_pairs += 1
            # Both: the same term must not appear in the other group either.
            used.add(synset)
            used.add(term)
            pair_id = f"vgkg-{config.split}-{group[0]}{num_pairs:03d}"
            records.extend(
                make_kg_pair(group, synset, description, answers, pair_id, templates, config)
            )
    return records, rejections


def run_kg_generation(config: KgQueryConfig) -> dict:
    """Load train + one split, generate KG candidates, check them, write JSONL + manifest."""
    split = load_split(config.split_file)
    train_ids = {int(image_id) for image_id in split["train"].tolist()}
    eval_ids = {int(image_id) for image_id in split[config.split].tolist()}
    objects_entries, relationships_entries = load_annotation_files(
        config.data_dir, config.objects_file, config.relationships_file
    )
    # Train graphs only give counts (targets, pool); answers come from the split's graphs.
    train_graphs, _ = build_scene_graphs(objects_entries, relationships_entries, train_ids)
    eval_graphs, parse_report = build_scene_graphs(objects_entries, relationships_entries, eval_ids)
    train_graphs = canonicalize_graphs(train_graphs, config.predicate_aliases)
    eval_graphs = canonicalize_graphs(eval_graphs, config.predicate_aliases)

    edges = load_edges_tsv(config.conceptnet_edges)
    records, rejections = generate_kg_pairs(
        train_graphs, eval_graphs, build_fact_index(edges), config
    )
    if not records:
        raise ValueError("no KG candidate pair survived the filters; nothing written")

    eval_id_strings = {str(image_id) for image_id in eval_ids}
    problems = validate_queries(records, eval_id_strings, expected_split=config.split)
    problems.extend(check_kg_evidence(records, {edge.uri for edge in edges}))
    if problems:
        raise ValueError("generated KG candidates are invalid:\n" + "\n".join(problems))

    write_jsonl(records, config.candidates_path)
    write_review_manifest(records, config.manifest_path)
    groups = Counter(record["query_group"] for record in records if record["type"] == "direct")
    return {
        "split": config.split,
        "num_pairs": len(records) // 2,
        "pairs_per_group": dict(sorted(groups.items())),
        "num_conceptnet_edges": len(edges),
        "dropped_records": dict(sorted(parse_report.items())),
        "rejected_concepts": dict(sorted(rejections.items())),
        "candidates_path": config.candidates_path,
        "manifest_path": config.manifest_path,
    }
