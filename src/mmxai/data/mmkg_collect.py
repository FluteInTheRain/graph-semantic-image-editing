"""Collect a raw, dated snapshot of Vietnamese heritage data from Wikidata and Commons.

The snapshot is never edited afterwards. This module deliberately applies NO filter
(no licence, size or count rule): a later step builds the official tables from the
snapshot with explicit rules, so the rules can change without collecting again.
"""

import html
import json
import re
import shutil
import subprocess
import unicodedata
import urllib.parse
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from mmxai.data.mmkg_csv import write_rows
from mmxai.data.wikimedia import WikimediaClient
from mmxai.utils.yaml_config import load_yaml_dataclass

ENTITY_PREFIX = "http://www.wikidata.org/entity/"
PROPERTY_PREFIX = "http://www.wikidata.org/prop/direct/"

SPARQL_BATCH_SIZE = 150
IMAGEINFO_BATCH_SIZE = 50
# A live run makes a few hundred Commons requests; a line every 25 shows it is alive.
PROGRESS_EVERY_REQUESTS = 25

ENTITY_COLUMNS = ["entity_id", "name_vi", "name_en", "in_domain", "commons_category", "p18_files"]
TRIPLE_COLUMNS = ["head_id", "property", "tail_id"]
CANDIDATE_COLUMNS = [
    "file_name",
    "entity_id",
    "origin",
    "mime",
    "width",
    "height",
    "license",
    "sha1",
]
CREDIT_COLUMNS = ["file_name", "artist", "uploader"]

# QLever, unlike the Wikidata Query Service, has no built-in prefixes, so every query
# declares the ones it uses (harmless on the Wikidata Query Service).
SPARQL_PREFIXES = (
    "PREFIX wd: <http://www.wikidata.org/entity/>\n"
    "PREFIX wdt: <http://www.wikidata.org/prop/direct/>\n"
    "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
)

# __X__ markers are filled with str.replace because SPARQL uses curly braces itself.
DOMAIN_QUERY = (
    SPARQL_PREFIXES
    + """SELECT DISTINCT ?item ?name_vi ?name_en ?category ?image WHERE {
  ?item wdt:__DESIGNATION__ ?designation ;
        wdt:P17 wd:__COUNTRY__ .
  OPTIONAL { ?item rdfs:label ?name_vi FILTER(LANG(?name_vi) = "vi") }
  OPTIONAL { ?item rdfs:label ?name_en FILTER(LANG(?name_en) = "en") }
  OPTIONAL { ?item wdt:P373 ?category }
  OPTIONAL { ?item wdt:P18 ?image }
}
"""
)

# No label service here: it timed out in the survey.
TRIPLES_QUERY = (
    SPARQL_PREFIXES
    + """SELECT DISTINCT ?h ?p ?t WHERE {
  ?h wdt:__DESIGNATION__ ?designation ;
     wdt:P17 wd:__COUNTRY__ .
  ?h ?p ?t .
  FILTER(STRSTARTS(STR(?p), "http://www.wikidata.org/prop/direct/P"))
  FILTER(STRSTARTS(STR(?t), "http://www.wikidata.org/entity/Q"))
}
"""
)

TARGET_LABELS_QUERY = (
    SPARQL_PREFIXES
    + """SELECT DISTINCT ?item ?name_vi ?name_en WHERE {
  VALUES ?item { __IDS__ }
  OPTIONAL { ?item rdfs:label ?name_vi FILTER(LANG(?name_vi) = "vi") }
  OPTIONAL { ?item rdfs:label ?name_en FILTER(LANG(?name_en) = "en") }
}
"""
)

CONCEPT_TRIPLES_QUERY = (
    SPARQL_PREFIXES
    + """SELECT DISTINCT ?h ?p ?t WHERE {
  VALUES ?h { __IDS__ }
  ?h ?p ?t .
  FILTER(STRSTARTS(STR(?p), "http://www.wikidata.org/prop/direct/P"))
  FILTER(STRSTARTS(STR(?t), "http://www.wikidata.org/entity/Q"))
}
"""
)


@dataclass(frozen=True)
class MmkgCollectConfig:
    """Settings of one collection run (configs/mmkg_vi_collect.yaml)."""

    output_root: str
    sparql_url: str
    country_qid: str
    designation_property: str
    user_agent: str
    request_interval_seconds: float
    max_attempts: int
    max_retry_wait_seconds: float
    max_file_bytes: int


@dataclass(frozen=True)
class RawSnapshot:
    """The four raw tables plus the metadata of one collection run."""

    entities: list[dict[str, str]]
    triples: list[dict[str, str]]
    image_candidates: list[dict[str, str]]
    image_credits: list[dict[str, str]]
    meta: dict


def load_mmkg_collect_config(path: str | Path) -> MmkgCollectConfig:
    """Read the collection config from a YAML file."""
    return load_yaml_dataclass(path, MmkgCollectConfig)


def normalize_file_name(value: str) -> str:
    """Turn a P18 URL, a `File:` title or a bare name into a bare name with spaces."""
    name = value.strip()
    if "://" in name:
        # Only URLs are percent-encoded; a bare name may contain a literal "%".
        name = urllib.parse.unquote(name.rsplit("/", 1)[-1])
    if name.startswith("File:"):
        name = name[len("File:") :]
    # Vietnamese diacritics can arrive decomposed (NFD) from one source and composed
    # from another; NFC makes both spellings the same string.
    name = unicodedata.normalize("NFC", name)
    return name.replace("_", " ")


def strip_html(value: str) -> str:
    """Remove tags, unescape entities and collapse whitespace.

    Each tag becomes a space: Commons puts "Unknown author" in two adjacent tags,
    and removing the tags outright glued them into "Unknown authorUnknown author".
    """
    without_tags = re.sub(r"<[^>]*>", " ", value)
    unescaped = html.unescape(without_tags)
    return " ".join(unescaped.split())


def _entity_id(uri: str) -> str:
    return uri.removeprefix(ENTITY_PREFIX)


def _property_id(uri: str) -> str:
    return uri.removeprefix(PROPERTY_PREFIX)


def _batches(items: list[str], size: int) -> list[list[str]]:
    batches = []
    for start in range(0, len(items), size):
        batches.append(items[start : start + size])
    return batches


def _values_list(entity_ids: list[str]) -> str:
    return " ".join("wd:" + entity_id for entity_id in entity_ids)


def _fill_domain_query(template: str, config: MmkgCollectConfig) -> str:
    query = template.replace("__DESIGNATION__", config.designation_property)
    return query.replace("__COUNTRY__", config.country_qid)


def _collect_domain(client, config, queries) -> dict[str, dict]:
    """Group the domain rows per item (SPARQL returns one row per P18/P373 combination)."""
    query = _fill_domain_query(DOMAIN_QUERY, config)
    queries["domain"] = query
    items: dict[str, dict] = {}
    for row in client.sparql(query):
        entity_id = _entity_id(row["item"])
        if entity_id not in items:
            items[entity_id] = {"name_vi": "", "name_en": "", "categories": set(), "files": set()}
        item = items[entity_id]
        if row.get("name_vi"):
            item["name_vi"] = row["name_vi"]
        if row.get("name_en"):
            item["name_en"] = row["name_en"]
        if row.get("category"):
            item["categories"].add(row["category"])
        if row.get("image"):
            item["files"].add(normalize_file_name(row["image"]))
    return items


def _collect_domain_triples(client, config, queries) -> set[tuple[str, str, str]]:
    query = _fill_domain_query(TRIPLES_QUERY, config)
    queries["triples"] = query
    triples = set()
    for row in client.sparql(query):
        triples.add((_entity_id(row["h"]), _property_id(row["p"]), _entity_id(row["t"])))
    return triples


def _collect_target_labels(client, target_ids, queries) -> dict[str, dict[str, str]]:
    queries["target_labels"] = TARGET_LABELS_QUERY
    labels: dict[str, dict[str, str]] = {}
    for batch in _batches(target_ids, SPARQL_BATCH_SIZE):
        query = TARGET_LABELS_QUERY.replace("__IDS__", _values_list(batch))
        for row in client.sparql(query):
            entity_id = _entity_id(row["item"])
            entry = labels.setdefault(entity_id, {"name_vi": "", "name_en": ""})
            if row.get("name_vi"):
                entry["name_vi"] = row["name_vi"]
            if row.get("name_en"):
                entry["name_en"] = row["name_en"]
    return labels


def _collect_concept_triples(client, target_ids, entity_ids, queries) -> set[tuple[str, str, str]]:
    """Statements of targets, kept only when the tail is already known (no new entities)."""
    queries["concept_triples"] = CONCEPT_TRIPLES_QUERY
    triples = set()
    for batch in _batches(target_ids, SPARQL_BATCH_SIZE):
        query = CONCEPT_TRIPLES_QUERY.replace("__IDS__", _values_list(batch))
        for row in client.sparql(query):
            head = _entity_id(row["h"])
            tail = _entity_id(row["t"])
            if tail in entity_ids:
                triples.add((head, _property_id(row["p"]), tail))
    return triples


def _category_files(client: WikimediaClient, category: str) -> list[str]:
    """List the files of a Commons category, following `continue` until the end."""
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": "Category:" + category,
        "cmtype": "file",
        "cmlimit": "500",
    }
    names = []
    while True:
        answer = client.commons(params)
        for member in answer["query"]["categorymembers"]:
            names.append(normalize_file_name(member["title"]))
        if "continue" not in answer:
            return names
        # Send back every key of `continue`, as the MediaWiki API requires.
        params = dict(params)
        params.update(answer["continue"])


def _collect_candidate_pairs(client, domain_items, log) -> list[tuple[str, str, str]]:
    """Return (file_name, entity_id, origin); P18 wins when a file is both P18 and category."""
    pairs = []
    category_total = 0
    for item in domain_items.values():
        if item["categories"]:
            category_total += 1
    categories_done = 0
    for entity_id in sorted(domain_items):
        item = domain_items[entity_id]
        seen = set()
        for file_name in sorted(item["files"]):
            pairs.append((file_name, entity_id, "p18"))
            seen.add(file_name)
        if item["categories"]:
            category = sorted(item["categories"])[0]
            for file_name in _category_files(client, category):
                if file_name not in seen:
                    pairs.append((file_name, entity_id, "category"))
                    seen.add(file_name)
            categories_done += 1
            if categories_done % PROGRESS_EVERY_REQUESTS == 0:
                log(f"category files: {categories_done}/{category_total} categories")
    return pairs


def _fetch_image_info(client, file_names: list[str], log) -> dict[str, tuple[str, dict]]:
    """Return {requested name: (final name, imageinfo)}; unknown files are left out.

    Commons may answer under another title than the one asked: `normalized` fixes the
    spelling (e.g. a lower-case first letter), then `redirects` follows a renamed file
    to its current page. The final name is the one the file is downloaded under.
    """
    resolved = {}
    files_done = 0
    for request_number, batch in enumerate(_batches(file_names, IMAGEINFO_BATCH_SIZE), start=1):
        titles = "|".join("File:" + name for name in batch)
        params = {
            "action": "query",
            "prop": "imageinfo",
            "iiprop": "extmetadata|size|sha1|mime|user",
            "titles": titles,
            "redirects": "1",
        }
        answer = client.commons(params)
        query = answer["query"]
        normalized = {}
        for change in query.get("normalized", []):
            normalized[normalize_file_name(change["from"])] = normalize_file_name(change["to"])
        redirects = {}
        for change in query.get("redirects", []):
            redirects[normalize_file_name(change["from"])] = normalize_file_name(change["to"])
        info_by_final_name = {}
        for page in query["pages"].values():
            if "missing" in page or "imageinfo" not in page:
                continue
            info_by_final_name[normalize_file_name(page["title"])] = page["imageinfo"][0]

        for name in batch:
            # Normalization happens first, then the redirect is followed (one step only).
            final_name = normalized.get(name, name)
            final_name = redirects.get(final_name, final_name)
            if final_name in info_by_final_name:
                resolved[name] = (final_name, info_by_final_name[final_name])
        files_done += len(batch)
        if request_number % PROGRESS_EVERY_REQUESTS == 0:
            log(f"image info: {files_done}/{len(file_names)} files")
    return resolved


def _metadata_value(info: dict, key: str) -> str:
    entry = info.get("extmetadata", {}).get(key)
    if entry is None:
        return ""
    return entry.get("value", "")


def _sparql_index(client: WikimediaClient, log) -> str:
    """Name of the data dump behind the SPARQL endpoint, or "unknown" if it cannot be read.

    QLever serves a Wikidata dump, not live Wikidata, so the dump name tells readers how
    fresh the data is. It is only provenance, so the run does not fail without it.
    """
    try:
        return client.sparql_endpoint_stats()["name-index"]
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        log(f"warning: could not read the SPARQL endpoint stats (name-index): {error!r}")
        return "unknown"


def _git_commit() -> str:
    """Current commit, or "unknown" when git is not available (the only tolerated fallback)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def collect_snapshot(
    client: WikimediaClient, config: MmkgCollectConfig, today: date, log=print
) -> RawSnapshot:
    """Query Wikidata and Commons and return the raw tables, without any filtering.

    `log` receives one line when each phase starts and ends, plus progress lines,
    so a run of several minutes can be followed in a log file.
    """
    queries: dict[str, str] = {}
    log("domain items: start")
    domain_items = _collect_domain(client, config, queries)
    log(f"domain items: done, {len(domain_items)} items")
    log("triples: start")
    domain_triples = _collect_domain_triples(client, config, queries)
    log(f"triples: done, {len(domain_triples)} triples")

    domain_ids = set(domain_items)
    target_ids = set()
    for _head, _property, tail in domain_triples:
        if tail not in domain_ids:
            target_ids.add(tail)
    sorted_targets = sorted(target_ids)

    log(f"target labels: start, {len(sorted_targets)} targets")
    labels = _collect_target_labels(client, sorted_targets, queries)
    log(f"target labels: done, {len(labels)} labelled")
    entity_ids = domain_ids | target_ids
    log("concept triples: start")
    concept_triples = _collect_concept_triples(client, sorted_targets, entity_ids, queries)
    log(f"concept triples: done, {len(concept_triples)} triples")

    entities = []
    for entity_id in sorted(domain_items):
        item = domain_items[entity_id]
        category = ""
        if item["categories"]:
            category = sorted(item["categories"])[0]
        entities.append(
            {
                "entity_id": entity_id,
                "name_vi": item["name_vi"],
                "name_en": item["name_en"],
                "in_domain": "1",
                "commons_category": category,
                "p18_files": "|".join(sorted(item["files"])),
            }
        )
    for entity_id in sorted_targets:
        label = labels.get(entity_id, {"name_vi": "", "name_en": ""})
        entities.append(
            {
                "entity_id": entity_id,
                "name_vi": label["name_vi"],
                "name_en": label["name_en"],
                "in_domain": "0",
                "commons_category": "",
                "p18_files": "",
            }
        )

    triples = []
    for head, property_id, tail in sorted(domain_triples | concept_triples):
        triples.append({"head_id": head, "property": property_id, "tail_id": tail})

    log("category files: start")
    pairs = _collect_candidate_pairs(client, domain_items, log)
    log(f"category files: done, {len(pairs)} entity-file pairs")
    requested_files = sorted({file_name for file_name, _entity, _origin in pairs})
    log(f"image info: start, {len(requested_files)} files")
    resolved = _fetch_image_info(client, requested_files, log)
    log(f"image info: done, {len(resolved)} found")

    image_candidates = []
    seen_pairs = set()
    # Rows are recorded under the final (canonical) name, so a redirect can make two
    # rows of one entity point at the same file. `pairs` lists each entity's P18 files
    # before its category files, so keeping the first row lets P18 win.
    for requested_name, entity_id, origin in pairs:
        if requested_name not in resolved:
            continue
        file_name, info = resolved[requested_name]
        if (file_name, entity_id) in seen_pairs:
            continue
        seen_pairs.add((file_name, entity_id))
        image_candidates.append(
            {
                "file_name": file_name,
                "entity_id": entity_id,
                "origin": origin,
                "mime": info.get("mime", ""),
                "width": str(info.get("width", "")),
                "height": str(info.get("height", "")),
                "license": _metadata_value(info, "LicenseShortName"),
                "sha1": info.get("sha1", ""),
            }
        )

    missing_files = []
    info_by_file_name = {}
    for requested_name in requested_files:
        if requested_name not in resolved:
            missing_files.append(requested_name)
            continue
        file_name, info = resolved[requested_name]
        info_by_file_name[file_name] = info

    image_credits = []
    for file_name in sorted(info_by_file_name):
        info = info_by_file_name[file_name]
        image_credits.append(
            {
                "file_name": file_name,
                "artist": strip_html(_metadata_value(info, "Artist")),
                "uploader": info.get("user", ""),
            }
        )

    meta = {
        "collected_on": today.isoformat(),
        "queries": queries,
        "row_counts": {
            "entities_raw": len(entities),
            "triples_raw": len(triples),
            "image_candidates": len(image_candidates),
            "image_credits": len(image_credits),
        },
        "missing_files": missing_files,
        "git_commit": _git_commit(),
        "user_agent": config.user_agent,
        "sparql_url": config.sparql_url,
        "sparql_index": _sparql_index(client, log),
        "config": asdict(config),
    }
    return RawSnapshot(entities, triples, image_candidates, image_credits, meta)


def write_snapshot(snapshot: RawSnapshot, output_root: Path, max_file_bytes: int) -> Path:
    """Write `output_root/<YYYYMMDD>/`; all-or-nothing, and never over an existing snapshot."""
    collected_on = date.fromisoformat(snapshot.meta["collected_on"])
    folder_name = collected_on.strftime("%Y%m%d")
    final_folder = output_root / folder_name
    partial_folder = output_root / (folder_name + ".partial")

    if final_folder.exists():
        raise FileExistsError(f"snapshot already exists: {final_folder}")
    output_root.mkdir(parents=True, exist_ok=True)
    # A partial folder can only be left by a crash, so it is safe to discard.
    if partial_folder.exists():
        shutil.rmtree(partial_folder)
    partial_folder.mkdir()

    try:
        write_rows(
            partial_folder, "entities_raw", ENTITY_COLUMNS, snapshot.entities, max_file_bytes
        )
        write_rows(partial_folder, "triples_raw", TRIPLE_COLUMNS, snapshot.triples, max_file_bytes)
        write_rows(
            partial_folder,
            "image_candidates",
            CANDIDATE_COLUMNS,
            snapshot.image_candidates,
            max_file_bytes,
        )
        write_rows(
            partial_folder, "image_credits", CREDIT_COLUMNS, snapshot.image_credits, max_file_bytes
        )
        meta_text = json.dumps(snapshot.meta, indent=2, ensure_ascii=False)
        (partial_folder / "meta.json").write_text(meta_text + "\n", encoding="utf-8")
        # Rename last, so a folder named by date is always complete.
        partial_folder.rename(final_folder)
    except BaseException:
        shutil.rmtree(partial_folder, ignore_errors=True)
        raise
    return final_folder
