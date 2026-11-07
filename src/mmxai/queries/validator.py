"""Check query files, find candidate relevant images, and turn a human review into the benchmark.

Workflow (ADR-0044):
    generator -> candidates JSONL + review manifest CSV (status "candidate")
    member A looks at the images, fills reviewer + verified_relevant -> "reviewed"
    member B (not A) re-checks, fills checker -> "verified"
    build_verified_benchmark keeps ONLY verified pairs -> queries/visual_genome_test.jsonl

Relevance (ADR-0044): an image is a candidate answer when its scene graph holds the
same chain as the query: objects with the same match key (VG WordNet synset, else
normalised name), linked by relations
with the same normalised predicates and directions, through distinct object ids.
The source image is therefore one answer among possibly many, never assumed alone.
"""

import csv
import re
from pathlib import Path

from mmxai.data.scene_graph import SceneGraph, match_key
from mmxai.data.splits import load_split
from mmxai.queries.schema import (
    STATUS_CANDIDATE,
    STATUS_REJECTED,
    STATUS_REVIEWED,
    STATUS_VERIFIED,
    check_record_schema,
    load_jsonl,
    write_jsonl,
)

# One row per direct/indirect pair: both queries share the evidence and the answers,
# so a reviewer judges them together.
MANIFEST_COLUMNS = [
    "pair_id",
    "source_image_id",
    "hop_count",
    "reasoning_type",
    "direct_qid",
    "direct_text",
    "indirect_qid",
    "indirect_text",
    "path_hint",
    "relationship_ids",
    "hidden_names",
    "candidate_relevant",
    "num_candidates",
    "verification_status",
    "reviewer",
    "verified_relevant",
    "checker",
    "edited_direct_text",
    "edited_indirect_text",
    "notes",
]
REVIEW_STATUSES = (STATUS_CANDIDATE, STATUS_REVIEWED, STATUS_VERIFIED, STATUS_REJECTED)


# --- Chain patterns and candidate relevance ----------------------------------------------


def evidence_key(relation: dict, side: str) -> str:
    """Match key of one end ("subject"/"object") of a stored relation: synset, else name."""
    synset = relation.get(f"{side}_synset", "")
    if synset:
        return synset
    return relation[f"{side}_name"]


def pattern_from_record(record: dict) -> tuple[list[str], list[tuple[str, bool]]]:
    """Return (node match keys n0..nk, steps) of the query's chain.

    Step i links node i to node i+1 by a predicate; `forward` is True when node i is
    the subject of that relation. Everything comes from the stored evidence, so
    anyone can recompute the answers from the JSONL line alone.
    """
    node_ids = record["chain_object_ids"]
    evidence = record["relation_evidence"]
    if len(evidence) != len(node_ids) - 1:
        raise ValueError(f"{record['qid']}: {len(evidence)} relations for {len(node_ids)} nodes")

    keys = []
    steps = []
    for position, relation in enumerate(evidence):
        current_id = node_ids[position]
        next_id = node_ids[position + 1]
        if relation["subject_id"] == current_id and relation["object_id"] == next_id:
            forward = True
            current_key = evidence_key(relation, "subject")
        elif relation["object_id"] == current_id and relation["subject_id"] == next_id:
            forward = False
            current_key = evidence_key(relation, "object")
        else:
            raise ValueError(f"{record['qid']}: relation {position} does not link its chain nodes")
        keys.append(current_key)
        steps.append((relation["predicate"], forward))
    last = evidence[-1]
    if steps[-1][1]:
        keys.append(evidence_key(last, "object"))
    else:
        keys.append(evidence_key(last, "subject"))
    return keys, steps


def extend_match(
    graph: SceneGraph, used_ids: list[int], keys: list[str], steps: list[tuple[str, bool]]
) -> bool:
    """Depth-first search: can the partial match `used_ids` be extended to the whole chain?"""
    step_index = len(used_ids) - 1
    if step_index == len(steps):
        return True
    predicate, forward = steps[step_index]
    current_id = used_ids[-1]
    for relation in graph.relations:
        if relation.predicate != predicate:
            continue
        if forward:
            if relation.subject_id != current_id:
                continue
            next_id = relation.object_id
        else:
            if relation.object_id != current_id:
                continue
            next_id = relation.subject_id
        # Distinct ids: one object may not play two roles of the chain.
        if next_id in used_ids or match_key(graph.objects[next_id]) != keys[step_index + 1]:
            continue
        if extend_match(graph, used_ids + [next_id], keys, steps):
            return True
    return False


def graph_has_pattern(graph: SceneGraph, keys: list[str], steps: list[tuple[str, bool]]) -> bool:
    """True if some distinct objects of this image form the chain."""
    present_keys = {match_key(scene_object) for scene_object in graph.objects.values()}
    if not set(keys) <= present_keys:
        return False
    for scene_object in graph.objects.values():
        if match_key(scene_object) != keys[0]:
            continue
        if extend_match(graph, [scene_object.object_id], keys, steps):
            return True
    return False


def images_matching_pattern(
    keys: list[str], steps: list[tuple[str, bool]], graphs: dict[int, SceneGraph]
) -> list[int]:
    """Ids (sorted) of every image in `graphs` whose scene graph contains the chain.

    Pass only the graphs of the evaluated split, so answers never come from train.
    """
    matches = []
    for image_id in sorted(graphs):
        if graph_has_pattern(graphs[image_id], keys, steps):
            matches.append(image_id)
    return matches


def hidden_names_in_text(text: str, hidden_names: list[str]) -> list[str]:
    """Hidden (bridge) names that still appear as whole words in the text.

    An indirect query that names its bridge object is only a reworded direct query.
    """
    lowered = text.lower()
    found = []
    for name in hidden_names:
        if re.search(rf"\b{re.escape(name)}\b", lowered):
            found.append(name)
    return found


# --- Validation of a query file ------------------------------------------------------------


def check_pairs(records: list[dict]) -> list[str]:
    """Each pair_id must hold exactly one direct and one indirect query on the same answers."""
    problems = []
    by_pair: dict[str, list[dict]] = {}
    for record in records:
        if "pair_id" in record:
            by_pair.setdefault(record["pair_id"], []).append(record)
    for pair_id, members in sorted(by_pair.items()):
        types = sorted(member["type"] for member in members)
        if types != ["direct", "indirect"]:
            problems.append(f"pair {pair_id}: expected one direct + one indirect, got {types}")
            continue
        first, second = members
        if first.get("source_image_id") != second.get("source_image_id"):
            problems.append(f"pair {pair_id}: the two queries have different source images")
        if sorted(first["relevant"]) != sorted(second["relevant"]):
            problems.append(f"pair {pair_id}: the two queries have different relevant images")
    return problems


def check_evidence(record: dict, graphs: dict[int, SceneGraph]) -> list[str]:
    """Every evidence relation must exist, unchanged, in the source image's annotations."""
    qid = record["qid"]
    source = int(record["source_image_id"])
    graph = graphs.get(source)
    if graph is None:
        return [f"{qid}: source image {source} has no scene graph in this split"]
    problems = []
    for evidence in record["relation_evidence"]:
        relation = graph.relation_by_id(evidence["relationship_id"])
        if relation is None:
            problems.append(f"{qid}: relationship {evidence['relationship_id']} not annotated")
            continue
        stored = (evidence["subject_id"], evidence["predicate"], evidence["object_id"])
        annotated = (relation.subject_id, relation.predicate, relation.object_id)
        if stored != annotated:
            problems.append(f"{qid}: relationship {relation.relationship_id} differs: {stored}")
    if not problems:
        keys, steps = pattern_from_record(record)
        if not graph_has_pattern(graph, keys, steps):
            problems.append(f"{qid}: the chain does not match its own source image")
    return problems


def validate_queries(
    records: list[dict],
    split_image_ids: set[str],
    graphs: dict[int, SceneGraph] | None = None,
    require_verified: bool = False,
    expected_split: str = "test",
) -> list[str]:
    """Return every problem found (empty list = valid).

    split_image_ids are the images of expected_split. The benchmark is always
    "test" (ADR-0009); only the val pilot passes expected_split="val".
    graphs (that split's scene graphs) is optional so that the committed benchmark
    can be checked in pytest without the multi-GB annotations.
    """
    problems = []
    seen_qids = set()
    for record in records:
        schema_problems = check_record_schema(record)
        if schema_problems:
            problems.extend(schema_problems)
            continue
        qid = record["qid"]
        if qid in seen_qids:
            problems.append(f"{qid}: duplicate qid")
        seen_qids.add(qid)

        for image_id in record["relevant"]:
            if image_id not in split_image_ids:
                problems.append(f"{qid}: relevant image {image_id} is not in {expected_split}")
        source = record.get("source_image_id")
        if source is not None and str(source) not in split_image_ids:
            problems.append(f"{qid}: source image {source} is not in {expected_split}")
        if record.get("split", expected_split) != expected_split:
            problems.append(f"{qid}: split is {record['split']!r}, expected {expected_split!r}")
        if require_verified and record.get("verification_status") != STATUS_VERIFIED:
            problems.append(f"{qid}: not verified (status {record.get('verification_status')!r})")
        if record["type"] == "indirect" and "hidden_names" in record:
            leaked = hidden_names_in_text(record["text"], record["hidden_names"])
            if leaked:
                problems.append(f"{qid}: indirect text names hidden objects {leaked}")
        if graphs is not None and "relation_evidence" in record:
            problems.extend(check_evidence(record, graphs))

    problems.extend(check_pairs(records))
    return problems


# --- Review manifest -------------------------------------------------------------------------


def manifest_rows(candidates: list[dict]) -> list[dict]:
    """One review row per pair, ordered by pair_id, all starting as "candidate"."""
    by_pair: dict[str, dict[str, dict]] = {}
    for record in candidates:
        by_pair.setdefault(record["pair_id"], {})[record["type"]] = record

    rows = []
    for pair_id in sorted(by_pair):
        direct = by_pair[pair_id]["direct"]
        indirect = by_pair[pair_id]["indirect"]
        relationship_ids = [str(item["relationship_id"]) for item in direct["relation_evidence"]]
        rows.append(
            {
                "pair_id": pair_id,
                "source_image_id": direct["source_image_id"],
                "hop_count": direct["hop_count"],
                "reasoning_type": indirect["reasoning_type"],
                "direct_qid": direct["qid"],
                "direct_text": direct["text"],
                "indirect_qid": indirect["qid"],
                "indirect_text": indirect["text"],
                "path_hint": direct["path_hint"],
                "relationship_ids": " ".join(relationship_ids),
                "hidden_names": " | ".join(indirect["hidden_names"]),
                "candidate_relevant": " ".join(direct["relevant"]),
                "num_candidates": len(direct["relevant"]),
                "verification_status": STATUS_CANDIDATE,
                "reviewer": "",
                "verified_relevant": "",
                "checker": "",
                "edited_direct_text": "",
                "edited_indirect_text": "",
                "notes": "",
            }
        )
    return rows


def write_review_manifest(candidates: list[dict], path: str | Path) -> None:
    """Write the CSV reviewers fill in. Refuses to overwrite: it may already hold reviews."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists and may contain human reviews")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(manifest_rows(candidates))


def read_review_manifest(path: str | Path) -> list[dict]:
    """Read the reviewed CSV; values are stripped strings."""
    with open(path, encoding="utf-8", newline="") as file:
        rows = []
        for row in csv.DictReader(file):
            cleaned = {}
            for key, value in row.items():
                cleaned[key] = (value or "").strip()
            rows.append(cleaned)
    return rows


def check_verified_row(row: dict, test_image_ids: set[str]) -> list[str]:
    """A verified row needs two different people and human-confirmed test image ids."""
    pair_id = row["pair_id"]
    problems = []
    if not row["reviewer"] or not row["checker"]:
        problems.append(f"pair {pair_id}: verified needs both reviewer and checker")
    elif row["reviewer"].lower() == row["checker"].lower():
        # queries/README.md rule 2: another member re-checks the answers.
        problems.append(f"pair {pair_id}: checker must be a different member than reviewer")
    verified = row["verified_relevant"].split()
    if not verified:
        problems.append(f"pair {pair_id}: verified_relevant is empty")
    for image_id in verified:
        if image_id not in test_image_ids:
            problems.append(f"pair {pair_id}: verified image {image_id} is not in the test split")
    return problems


def build_verified_benchmark(
    candidates: list[dict], rows: list[dict], test_image_ids: set[str]
) -> list[dict]:
    """Return the final queries: only pairs whose row is "verified", with human answers.

    A row marked verified but incomplete raises instead of being skipped, so a
    reviewer's mistake is never silently dropped from (or let into) the benchmark.
    """
    by_pair: dict[str, list[dict]] = {}
    for record in candidates:
        by_pair.setdefault(record["pair_id"], []).append(record)

    problems = []
    final = []
    for row in rows:
        status = row["verification_status"]
        if status not in REVIEW_STATUSES:
            problems.append(f"pair {row['pair_id']}: unknown status {status!r}")
            continue
        if status != STATUS_VERIFIED:
            continue
        if row["pair_id"] not in by_pair:
            problems.append(f"pair {row['pair_id']}: not in the candidates file")
            continue
        row_problems = check_verified_row(row, test_image_ids)
        if row_problems:
            problems.extend(row_problems)
            continue

        relevant = sorted(set(row["verified_relevant"].split()), key=int)
        for candidate in by_pair[row["pair_id"]]:
            record = dict(candidate)
            record["relevant"] = relevant
            record["verification_status"] = STATUS_VERIFIED
            record["reviewed_by"] = row["reviewer"]
            record["checked_by"] = row["checker"]
            edited = row[f"edited_{record['type']}_text"]
            if edited:
                record["text"] = edited
            final.append(record)

    if problems:
        raise ValueError("review manifest has problems:\n" + "\n".join(problems))
    problems = validate_queries(final, test_image_ids, require_verified=True)
    if problems:
        raise ValueError("verified benchmark is invalid:\n" + "\n".join(problems))
    return final


def run_build_benchmark(
    candidates_path: str | Path,
    manifest_path: str | Path,
    split_file: str | Path,
    benchmark_path: str | Path,
    min_verified_pairs: int,
) -> int:
    """Write the official benchmark from the reviewed manifest; return the number of pairs.

    Refuses to write fewer than min_verified_pairs pairs (10 pairs = 20 queries, the
    PDF minimum) and never overwrites an existing benchmark file.
    """
    for path in (candidates_path, manifest_path):
        if not Path(path).exists():
            raise FileNotFoundError(
                f"{path} not found; run scripts/generate_queries.py first and review the manifest"
            )
    split = load_split(split_file)
    test_ids = {str(image_id) for image_id in split["test"].tolist()}
    candidates = load_jsonl(candidates_path)
    rows = read_review_manifest(manifest_path)
    final = build_verified_benchmark(candidates, rows, test_ids)
    num_pairs = len(final) // 2
    if num_pairs < min_verified_pairs:
        raise ValueError(
            f"only {num_pairs} verified pairs, need at least {min_verified_pairs}; "
            "review more rows of the manifest"
        )
    write_jsonl(final, benchmark_path)
    return num_pairs
