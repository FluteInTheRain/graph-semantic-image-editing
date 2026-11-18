"""Evaluation queries for the Vietnamese MMKG, generated from names and triples (ADR-0053).

For every item that has images, four queries:

    direct   / en : the item's English name        "Ngoc Lu bronze drum"
    direct   / vi : the item's Vietnamese name     "Trống đồng Ngọc Lũ"
    indirect / en : its triples, name hidden       "a bronze drum, made of bronze, ..."
    indirect / vi : the same triples in Vietnamese

A query's correct answer is every image of every item that fits it (ADR-0053 rule 6):
for an indirect query, every item that has all the described (relation, tail) pairs.
The queries are for evaluation only: nothing is tuned on them.
"""

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from mmxai.data.mmkg_csv import read_rows

# Decision (ADR-0053): one fixed sentence fragment per relation, in English and Vietnamese,
# with the same content. {} is the tail entity's name in that language.
#   Why: written once before any run, so no wording can be tuned on the results.
#   Decided by: qphulong, 2026-11-18 (proposed by Claude Code).
# instance_of is not here: it is the head noun of the sentence (see indirect_text).
CLAUSES: dict[str, tuple[str, str]] = {
    "made_from": ("made of {}", "làm bằng {}"),
    "culture": ("of the {}", "thuộc {}"),
    "time_period": ("from the {}", "thuộc thời kỳ {}"),
    "depicts": ("depicting {}", "mô tả {}"),
    "located_in": ("located in {}", "nằm ở {}"),
    "kept_at": ("kept at {}", "lưu giữ tại {}"),
    "part_of": ("part of {}", "là một phần của {}"),
    "has_part": ("containing {}", "có {}"),
    "discovered_at": ("discovered at {}", "được phát hiện ở {}"),
    "influenced_by": ("influenced by {}", "chịu ảnh hưởng của {}"),
    "named_after": ("named after {}", "được đặt theo tên {}"),
    "heritage_designation": ("listed as {}", "được xếp hạng {}"),
}
# Head noun when an item has no instance_of triple.
DEFAULT_HEAD = ("a heritage of Vietnam", "một di sản của Việt Nam")
LANGUAGES = ("en", "vi")
JOIN_WORD = {"en": " and ", "vi": " và "}


@dataclass(frozen=True)
class Entity:
    entity_id: str
    name_en: str
    name_vi: str
    kind: str  # "item" or "concept"

    def name(self, lang: str) -> str:
        return self.name_en if lang == "en" else self.name_vi


@dataclass(frozen=True)
class Query:
    qid: str
    item_id: str  # the item the query was written from
    kind: str  # "direct" or "indirect"
    lang: str  # "en" or "vi"
    text: str
    relevant_items: tuple[str, ...]  # every item that fits the query, sorted
    pairs: tuple[tuple[str, str], ...]  # the (relation, tail_id) pairs described; () if direct


def item_pairs(triples: list[dict]) -> dict[str, set[tuple[str, str]]]:
    """(relation, tail_id) pairs of every head entity."""
    pairs: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for triple in triples:
        pairs[triple["head_id"]].add((triple["relation"], triple["tail_id"]))
    return pairs


def check_relations(triples: list[dict]) -> None:
    """Refuse a relation that has no sentence fragment, instead of silently dropping it."""
    known = set(CLAUSES) | {"instance_of"}
    unknown = sorted({triple["relation"] for triple in triples} - known)
    if unknown:
        raise ValueError(f"no query template for relations {unknown}")


def joined_names(tail_ids: list[str], entities: dict[str, Entity], lang: str) -> str:
    # Sorted by name, so the sentence does not depend on the CSV row order.
    names = sorted(entities[tail_id].name(lang) for tail_id in tail_ids)
    return JOIN_WORD[lang].join(names)


def indirect_text(pairs: set[tuple[str, str]], entities: dict[str, Entity], lang: str) -> str:
    """Describe an item by its triples without its name, e.g. "a bronze drum, made of bronze"."""
    index = LANGUAGES.index(lang)
    tails_by_relation: dict[str, list[str]] = defaultdict(list)
    for relation, tail_id in pairs:
        tails_by_relation[relation].append(tail_id)

    if "instance_of" in tails_by_relation:
        head = joined_names(tails_by_relation["instance_of"], entities, lang)
        if lang == "en":
            head = f"a {head}"
    else:
        head = DEFAULT_HEAD[index]

    parts = [head]
    # CLAUSES order, so every query lists relations the same way.
    for relation, templates in CLAUSES.items():
        if relation in tails_by_relation:
            names = joined_names(tails_by_relation[relation], entities, lang)
            parts.append(templates[index].format(names))
    return ", ".join(parts)


def matching_items(
    described: set[tuple[str, str]],
    pairs_by_item: dict[str, set[tuple[str, str]]],
    item_ids: list[str],
) -> tuple[str, ...]:
    """Items that have every described pair: all of them are correct answers (ADR-0053 rule 6)."""
    return tuple(sorted(item for item in item_ids if described <= pairs_by_item.get(item, set())))


def build_queries(
    entities: dict[str, Entity], triples: list[dict], item_ids: list[str]
) -> list[Query]:
    """Four queries per item in item_ids (items that have images), in item_ids order.

    An item without triples gets its two direct queries only.
    """
    check_relations(triples)
    pairs_by_item = item_pairs(triples)
    queries = []
    for item_id in item_ids:
        item = entities[item_id]
        for lang in LANGUAGES:
            # Direct: the name. No two items share a name (checked by a test on the real
            # MMKG), but the rule still lists every item with the same name.
            same_name = tuple(
                sorted(other for other in item_ids if entities[other].name(lang) == item.name(lang))
            )
            queries.append(
                Query(
                    f"{item_id}_direct_{lang}",
                    item_id,
                    "direct",
                    lang,
                    item.name(lang),
                    same_name,
                    (),
                )
            )
        described = pairs_by_item.get(item_id, set())
        if not described:
            # Decision (ADR-0053): no indirect query for an item without triples.
            #   Why: an empty description fits every item, so all images would be correct.
            #   Decided by: Claude Code, 2026-11-18, reported to qphulong (1 item: Q910146).
            continue
        relevant = matching_items(described, pairs_by_item, item_ids)
        for lang in LANGUAGES:
            queries.append(
                Query(
                    f"{item_id}_indirect_{lang}",
                    item_id,
                    "indirect",
                    lang,
                    indirect_text(described, entities, lang),
                    relevant,
                    tuple(sorted(described)),
                )
            )
    return queries


def query_record(query: Query, images_by_item: dict[str, list[str]]) -> dict:
    """One JSON line in the format of queries/README.md, plus the MMKG fields."""
    relevant_images = sorted(
        image_id for item in query.relevant_items for image_id in images_by_item[item]
    )
    return {
        "qid": query.qid,
        "text": query.text,
        "type": query.kind,
        "lang": query.lang,
        "dataset": "mmkg_vi",
        "item_id": query.item_id,
        "relevant_items": list(query.relevant_items),
        "relevant": relevant_images,
        "path_hint": "; ".join(f"{relation} {tail}" for relation, tail in query.pairs),
        "author": "generated (scripts/make_mmkg_queries.py, ADR-0053)",
    }


def load_mmkg(mmkg_dir: Path) -> tuple[dict[str, Entity], list[dict], dict[str, list[str]]]:
    """Entities, triples and the image ids of every item, from the official MMKG CSV files."""
    entities = {
        row["entity_id"]: Entity(row["entity_id"], row["name_en"], row["name_vi"], row["type"])
        for row in read_rows(mmkg_dir, "entities")
    }
    triples = read_rows(mmkg_dir, "triples")
    images_by_item: dict[str, list[str]] = defaultdict(list)
    for row in read_rows(mmkg_dir, "images"):
        images_by_item[row["entity_id"]].append(row["image_id"])
    return entities, triples, dict(images_by_item)


def write_query_file(records: list[dict], path: Path) -> None:
    """JSON Lines, one query per line, UTF-8 kept readable (Vietnamese is not escaped)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
