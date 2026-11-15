# VG KG Query Benchmark (v2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate direct/indirect query pairs for Visual Genome where the indirect query hides a concept (an activity like *tennis* or an object like *train*) and describes it only through ConceptNet facts that point to that concept alone (ADR-0050).

**Architecture:** ConceptNet is filtered once into a small TSV (`src/mmxai/data/conceptnet.py`). Facts per concept and the "unique description" choice live in `src/mmxai/queries/kg_facts.py`; sentences in `kg_text.py`; concept → image lookups on scene graphs in `kg_targets.py`; the run loop in `kg_generator.py`. Validation, review manifest, build and evaluation reuse the v1 code (ADR-0044), with small generalisations.

**Tech Stack:** Python 3.11 standard library (`csv`, `gzip`, `json`, `hashlib`), numpy, PyYAML, pytest, ruff. No new dependency.

**Spec:** `docs/decisions/0050-truy-van-gian-tiep-mo-ta-khai-niem-qua-conceptnet.md`.

**Repo rules (CLAUDE.md):** branch `feat/vg-kg-query-benchmark` (already checked out, ADR-0050 committed as `83e795f`). Run every command as `python -m uv run ...` (uv is not on PATH on this machine). Code/comments in English, docs in Vietnamese, simple viva-explainable code. Commit after each task, never push. **Never run generation with the test config in this plan** — only unit tests and, if the ConceptNet file is present, the val pilot.

**Known pre-existing failures (ignore):** 10 tests in `tests/test_feature_pipeline.py` and `tests/test_notebooks_match_src.py`; 2 ruff I001 in `tests/test_eval_metrics.py`, `tests/test_eval_relevance.py`.

**Query test command used below** (`QTESTS`):
`python -m uv run pytest -q tests/test_conceptnet.py tests/test_kg_facts.py tests/test_kg_text.py tests/test_kg_targets.py tests/test_kg_generator.py tests/test_query_validator.py tests/test_query_generator.py tests/test_scene_graph.py tests/test_query_benchmark_eval.py tests/test_query_configs.py`
(Files that do not exist yet are created by the tasks; until then drop them from the command.)

---

## File map

| File | Responsibility |
|---|---|
| `src/mmxai/data/conceptnet.py` (new) | Parse ConceptNet assertion lines, filter English edges, read/write the filtered TSV |
| `scripts/prepare_conceptnet.py`, `configs/conceptnet_en.yaml`, `data_cards/conceptnet.md` (new) | One-time filtering CLI + its config + data card |
| `src/mmxai/queries/kg_facts.py` (new) | Turn edges into facts about a concept; choose 1 category + N properties unique to the concept |
| `src/mmxai/queries/kg_text.py` (new) | Direct/indirect sentences from a description |
| `src/mmxai/queries/kg_targets.py` (new) | Which images show which activity / object synset |
| `src/mmxai/queries/kg_generator.py` (new) | Config, pair records, run loop |
| `src/mmxai/data/scene_graph.py` (modify) | `load_annotation_files()` so both splits are built from one JSON read |
| `src/mmxai/queries/validator.py` (modify) | Manifest rows for KG records; `check_kg_evidence()` |
| `src/mmxai/eval/query_benchmark.py` (modify) | Metrics per `query_group` (activity / object) |
| `scripts/generate_kg_queries.py`, `scripts/build_kg_query_benchmark.py` (new) | Thin CLIs |
| `configs/query_benchmark_vg_kg.yaml`, `configs/query_benchmark_vg_kg_pilot.yaml`, `configs/eval_query_benchmark_vg_kg.yaml` (new) | Settings |
| `tests/conceptnet_fixtures.py` + `tests/test_conceptnet.py`, `test_kg_facts.py`, `test_kg_text.py`, `test_kg_targets.py`, `test_kg_generator.py` (new) | Synthetic tests |

---

### Task 1: ConceptNet parsing and the filtered TSV

**Files:**
- Create: `src/mmxai/data/conceptnet.py`
- Create: `tests/conceptnet_fixtures.py`
- Test: `tests/test_conceptnet.py`

ConceptNet 5.7 assertion files are tab-separated with 5 columns: assertion URI, relation URI, start node URI, end node URI, JSON info (holds `"weight"`). Example line:
`/a/[/r/UsedFor/,/c/en/racket/n/,/c/en/tennis/]	/r/UsedFor	/c/en/racket/n	/c/en/tennis	{"dataset": "/d/conceptnet/4/en", "weight": 2.0}`

- [ ] **Step 1: Write the fixture** `tests/conceptnet_fixtures.py`

```python
"""Tiny SYNTHETIC ConceptNet edges for the KG query tests.

Nothing here is copied from ConceptNet; the facts are made up so that the answers
are known (tennis and badminton share "racket", only tennis has "court").
"""

import gzip
import json
from pathlib import Path

from mmxai.data.conceptnet import ConceptEdge


def assertion_line(relation: str, start: str, end: str, weight: float = 1.0) -> str:
    """One line in the ConceptNet 5.7 assertions format."""
    uri = f"/a/[/r/{relation}/,{start}/,{end}/]"
    info = json.dumps({"dataset": "/d/test", "weight": weight})
    return "\t".join([uri, f"/r/{relation}", start, end, info])


def edge(relation: str, start: str, end: str, weight: float = 1.0) -> ConceptEdge:
    uri = f"/a/[/r/{relation}/,/c/en/{start.replace(' ', '_')}/,/c/en/{end.replace(' ', '_')}/]"
    return ConceptEdge(uri, relation, start, end, weight)


# Activities: tennis is the only sport that is played on a court.
# Objects: train is the only vehicle found at a railway; "car" has no unique fact.
EDGES = [
    edge("IsA", "tennis", "sport"),
    edge("IsA", "badminton", "sport"),
    edge("IsA", "baseball", "sport"),
    edge("UsedFor", "racket", "tennis", 2.0),
    edge("UsedFor", "racket", "badminton"),
    edge("UsedFor", "tennis ball", "tennis"),
    edge("AtLocation", "tennis", "court"),
    edge("UsedFor", "shuttlecock", "badminton"),
    edge("UsedFor", "bat", "baseball"),
    edge("UsedFor", "glove", "baseball"),
    edge("IsA", "train", "vehicle"),
    edge("IsA", "bus", "vehicle"),
    edge("IsA", "car", "vehicle"),
    edge("AtLocation", "train", "railway"),
    edge("CapableOf", "train", "carry passengers"),
    edge("CapableOf", "bus", "carry passengers"),
    edge("CapableOf", "car", "carry passengers"),
    edge("AtLocation", "bus", "street"),
    edge("AtLocation", "car", "street"),
]


def write_assertions_gz(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as file:
        for line in lines:
            file.write(line + "\n")
```

- [ ] **Step 2: Write the failing tests** `tests/test_conceptnet.py`

```python
"""ConceptNet parsing and filtering (ADR-0050). Synthetic lines only."""

from mmxai.data.conceptnet import (
    ConceptEdge,
    concept_term,
    load_edges_tsv,
    parse_assertion_line,
    prepare_conceptnet,
    write_edges_tsv,
)
from conceptnet_fixtures import assertion_line, write_assertions_gz

RELATIONS = ["IsA", "UsedFor"]


def test_concept_term_keeps_english_and_drops_the_part_of_speech():
    assert concept_term("/c/en/tennis_racket/n") == "tennis racket"
    assert concept_term("/c/en/tennis") == "tennis"
    assert concept_term("/c/fr/raquette") is None


def test_parse_keeps_only_english_edges_of_wanted_relations_and_weight():
    good = assertion_line("UsedFor", "/c/en/racket/n", "/c/en/tennis", 2.0)
    french = assertion_line("UsedFor", "/c/fr/raquette", "/c/en/tennis")
    other_relation = assertion_line("Synonym", "/c/en/racket", "/c/en/racquet")
    light = assertion_line("IsA", "/c/en/tennis", "/c/en/sport", 0.5)

    edge = parse_assertion_line(good, RELATIONS, min_weight=1.0)

    assert edge == ConceptEdge(good.split("\t")[0], "UsedFor", "racket", "tennis", 2.0)
    assert parse_assertion_line(french, RELATIONS, 1.0) is None
    assert parse_assertion_line(other_relation, RELATIONS, 1.0) is None
    assert parse_assertion_line(light, RELATIONS, 1.0) is None


def test_tsv_round_trip(tmp_path):
    edges = [ConceptEdge("/a/x", "IsA", "tennis", "sport", 1.0)]
    path = tmp_path / "edges.tsv"

    write_edges_tsv(edges, path)

    assert load_edges_tsv(path) == edges


def test_prepare_filters_a_gzip_file_and_reports_counts(tmp_path):
    source = tmp_path / "assertions.csv.gz"
    write_assertions_gz(
        source,
        [
            assertion_line("IsA", "/c/en/tennis", "/c/en/sport"),
            assertion_line("IsA", "/c/de/tennis", "/c/de/sport"),
        ],
    )
    output = tmp_path / "out" / "edges.tsv"

    summary = prepare_conceptnet(source, output, RELATIONS, min_weight=1.0)

    assert summary["lines_read"] == 2
    assert summary["edges_kept"] == 1
    assert len(summary["sha256"]) == 64
    assert load_edges_tsv(output)[0].start == "tennis"
```

- [ ] **Step 3: Run, expect FAIL** (`ModuleNotFoundError: mmxai.data.conceptnet`)

Run: `python -m uv run pytest -q tests/test_conceptnet.py`

- [ ] **Step 4: Implement** `src/mmxai/data/conceptnet.py`

```python
"""Read ConceptNet 5.7 assertions and keep the small English part the KG queries need.

ConceptNet is a public commonsense knowledge graph: each edge is
(start, relation, end, weight), e.g. racket -UsedFor-> tennis. The full assertions
file is large, so it is filtered ONCE (scripts/prepare_conceptnet.py) into a TSV
with only English-English edges of a few relations (ADR-0050). The original edge
URI is kept so every query can cite the exact edge it was built from.
"""

import csv
import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

TSV_COLUMNS = ["uri", "rel", "start", "end", "weight"]


@dataclass(frozen=True)
class ConceptEdge:
    """One kept edge; start/end are plain terms ("tennis racket"), rel has no "/r/"."""

    uri: str
    rel: str
    start: str
    end: str
    weight: float


def concept_term(node: str) -> str | None:
    """"/c/en/tennis_racket/n" -> "tennis racket"; None for non-English nodes."""
    parts = node.split("/")
    # ["", "c", "en", "tennis_racket", "n", ...]
    if len(parts) < 4 or parts[1] != "c" or parts[2] != "en":
        return None
    return parts[3].replace("_", " ")


def parse_assertion_line(line: str, relations: list[str], min_weight: float) -> ConceptEdge | None:
    """Turn one assertions line into a ConceptEdge, or None if the edge is not kept."""
    columns = line.rstrip("\n").split("\t")
    if len(columns) != 5:
        return None
    uri, relation_uri, start_uri, end_uri, info = columns
    relation = relation_uri.removeprefix("/r/")
    if relation not in relations:
        return None
    start = concept_term(start_uri)
    end = concept_term(end_uri)
    if start is None or end is None or start == end:
        return None
    weight = float(json.loads(info).get("weight", 0.0))
    if weight < min_weight:
        return None
    return ConceptEdge(uri, relation, start, end, weight)


def write_edges_tsv(edges: list[ConceptEdge], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(TSV_COLUMNS)
        for edge in edges:
            writer.writerow([edge.uri, edge.rel, edge.start, edge.end, edge.weight])


def load_edges_tsv(path: str | Path) -> list[ConceptEdge]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download conceptnet-assertions-5.7.0.csv.gz into data/conceptnet/ "
            "and run scripts/prepare_conceptnet.py (see data_cards/conceptnet.md)."
        )
    edges = []
    with open(path, encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file, delimiter="\t"):
            edges.append(
                ConceptEdge(row["uri"], row["rel"], row["start"], row["end"], float(row["weight"]))
            )
    return edges


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_conceptnet(
    source_path: str | Path, output_path: str | Path, relations: list[str], min_weight: float
) -> dict:
    """Filter the gzipped assertions file into the TSV; return counts and the TSV sha256."""
    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(
            f"{source_path} not found. Download it from "
            "https://github.com/commonsense/conceptnet5/wiki/Downloads (see data_cards/conceptnet.md)."
        )
    edges = []
    lines_read = 0
    with gzip.open(source_path, "rt", encoding="utf-8") as file:
        for line in file:
            lines_read += 1
            edge = parse_assertion_line(line, relations, min_weight)
            if edge is not None:
                edges.append(edge)
    write_edges_tsv(edges, output_path)
    return {
        "source": str(source_path),
        "output": str(output_path),
        "lines_read": lines_read,
        "edges_kept": len(edges),
        "sha256": file_sha256(output_path),
    }
```

Wrap any line over 100 characters (ruff E501) — e.g. split the error-message strings.

- [ ] **Step 5: Run, expect PASS**

Run: `python -m uv run pytest -q tests/test_conceptnet.py` → 4 passed. Then `python -m uv run ruff check src/mmxai/data/conceptnet.py tests/conceptnet_fixtures.py tests/test_conceptnet.py` and `python -m uv run ruff format` on the same files.

- [ ] **Step 6: Commit**

```bash
git add src/mmxai/data/conceptnet.py tests/conceptnet_fixtures.py tests/test_conceptnet.py
git commit -m "feat(data): parse and filter ConceptNet assertions" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Prepare CLI, config and data card

**Files:**
- Create: `scripts/prepare_conceptnet.py`, `configs/conceptnet_en.yaml`, `data_cards/conceptnet.md`
- Modify: `src/mmxai/data/conceptnet.py` (config dataclass)
- Test: `tests/test_conceptnet.py`

- [ ] **Step 1: Failing test** (append to `tests/test_conceptnet.py`, import `load_conceptnet_prepare_config`)

```python
def test_prepare_config_loads_the_repo_file():
    config = load_conceptnet_prepare_config("configs/conceptnet_en.yaml")

    assert config.relations == ["IsA", "UsedFor", "AtLocation", "HasA", "PartOf", "CapableOf"]
    assert config.min_weight == 1.0
```

- [ ] **Step 2: Run, expect FAIL** (ImportError)

- [ ] **Step 3: Implement.** In `conceptnet.py` add (import `load_yaml_dataclass` from `mmxai.utils.yaml_config`):

```python
@dataclass(frozen=True)
class ConceptNetPrepareConfig:
    source_path: str
    output_path: str
    relations: list[str]
    min_weight: float


def load_conceptnet_prepare_config(path: str | Path) -> ConceptNetPrepareConfig:
    return load_yaml_dataclass(path, ConceptNetPrepareConfig)
```

`configs/conceptnet_en.yaml`:

```yaml
# Filter ConceptNet 5.7 once for the KG query benchmark (ADR-0050).
# How to run: uv run python scripts/prepare_conceptnet.py --config configs/conceptnet_en.yaml
source_path: data/conceptnet/conceptnet-assertions-5.7.0.csv.gz
output_path: artifacts/conceptnet/en_edges.tsv
# Decision (ADR-0050): the relations the description templates can read.
relations: [IsA, UsedFor, AtLocation, HasA, PartOf, CapableOf]
# Decision (ADR-0050): drop edges asserted with weight < 1 (weakly supported).
min_weight: 1.0
```

`scripts/prepare_conceptnet.py`:

```python
"""Filter ConceptNet 5.7 into the small English TSV used by the KG queries (ADR-0050).

Logic: mmxai.data.conceptnet.prepare_conceptnet.

Before running:
    Download conceptnet-assertions-5.7.0.csv.gz into data/conceptnet/
    (https://github.com/commonsense/conceptnet5/wiki/Downloads; see data_cards/conceptnet.md).

Run (from the repo root):
    uv run python scripts/prepare_conceptnet.py --config configs/conceptnet_en.yaml

Output:
    <output_path> (TSV, in artifacts/, gitignored); the printed sha256 goes into the data card.
"""

import argparse
import json

from mmxai.data.conceptnet import load_conceptnet_prepare_config, prepare_conceptnet


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="YAML file in configs/")
    args = parser.parse_args()

    config = load_conceptnet_prepare_config(args.config)
    summary = prepare_conceptnet(
        config.source_path, config.output_path, config.relations, config.min_weight
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
```

`data_cards/conceptnet.md` (Vietnamese; read `data_cards/visual_genome.md` first and follow its section layout). Content to include:
- Nguồn: ConceptNet 5.7, file `conceptnet-assertions-5.7.0.csv.gz`, trang tải https://github.com/commonsense/conceptnet5/wiki/Downloads.
- Giấy phép: CC BY-SA 4.0 (theo trang ConceptNet; một số nguồn con có điều khoản riêng, xem trang gốc). Không commit file dữ liệu.
- Cách dùng trong đồ án: lọc bằng `scripts/prepare_conceptnet.py`, chỉ giữ cạnh tiếng Anh, 6 quan hệ, weight ≥ 1; output `artifacts/conceptnet/en_edges.tsv`.
- Ô trống để thành viên điền: ngày tải, dung lượng file, sha256 file tải về, sha256 TSV (in ra bởi script), số dòng đọc / số cạnh giữ.
- Rò rỉ: tri thức ngoài, không phụ thuộc split.

- [ ] **Step 4: Run, expect PASS** — `python -m uv run pytest -q tests/test_conceptnet.py` → 5 passed; ruff check + format on changed files.

- [ ] **Step 5: Commit**

```bash
git add src/mmxai/data/conceptnet.py scripts/prepare_conceptnet.py configs/conceptnet_en.yaml data_cards/conceptnet.md tests/test_conceptnet.py
git commit -m "feat(data): ConceptNet prepare CLI, config and data card" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Facts and the unique description

**Files:**
- Create: `src/mmxai/queries/kg_facts.py`
- Test: `tests/test_kg_facts.py`

How facts are read from edges (direction matters):

| Edge | Concept | Fact kind | Value |
|---|---|---|---|
| `C IsA K` | C | `isa` | K |
| `X UsedFor C` | C | `uses` | X |
| `C AtLocation Y` | C | `location` | Y |
| `C HasA X` | C | `has` | X |
| `X PartOf C` | C | `has` | X |
| `C CapableOf V` | C | `can` | V |

- [ ] **Step 1: Failing tests** `tests/test_kg_facts.py`

```python
"""Facts per concept and the unique description (ADR-0050). Synthetic edges only."""

from mmxai.queries.kg_facts import build_fact_index, choose_description, leaks_name
from conceptnet_fixtures import EDGES

CATEGORIES = ["sport", "vehicle"]
SPORTS = ["tennis", "badminton", "baseball"]
VEHICLES = ["train", "bus", "car"]


def test_facts_follow_the_edge_direction():
    index = build_fact_index(EDGES)

    kinds = sorted((fact.kind, fact.value) for fact in index["tennis"])

    assert kinds == [
        ("isa", "sport"),
        ("location", "court"),
        ("uses", "racket"),
        ("uses", "tennis ball"),
    ]
    assert ("can", "carry passengers") in {(fact.kind, fact.value) for fact in index["train"]}


def test_description_is_unique_and_prefers_rare_properties():
    index = build_fact_index(EDGES)

    description = choose_description("tennis", SPORTS, index, CATEGORIES, num_properties=2)

    assert description.category.value == "sport"
    # "court" belongs to tennis only, so it is picked first; "tennis ball" leaks the name.
    assert [(fact.kind, fact.value) for fact in description.properties] == [
        ("location", "court"),
        ("uses", "racket"),
    ]


def test_no_description_when_every_property_is_shared():
    index = build_fact_index(EDGES)

    # car: vehicle + carry passengers + street, all shared with bus.
    assert choose_description("car", VEHICLES, index, CATEGORIES, num_properties=2) is None


def test_no_description_without_a_known_category():
    index = build_fact_index(EDGES)

    assert choose_description("tennis", SPORTS, index, ["animal"], num_properties=2) is None


def test_leaks_name_checks_whole_words():
    assert leaks_name("tennis ball", "tennis")
    assert leaks_name("tennis", "tennis")
    assert not leaks_name("court", "tennis")
```

- [ ] **Step 2: Run, expect FAIL** (ModuleNotFoundError)

Run: `python -m uv run pytest -q tests/test_kg_facts.py`

- [ ] **Step 3: Implement** `src/mmxai/queries/kg_facts.py`

```python
"""Facts about a concept from ConceptNet, and a description that fits only that concept.

A fact is something true about a concept C read from one edge: "C is a sport",
"C uses a racket" (racket -UsedFor-> C). An indirect query names one category and a
few properties of C but never C itself (ADR-0050). To be a fair query it must point
to C alone: no other concept of the same group may have the same category and
properties, otherwise images of that other concept would also be correct answers.
"""

from dataclasses import dataclass

from mmxai.data.conceptnet import ConceptEdge
from mmxai.queries.generator import PLAIN_WORDS


@dataclass(frozen=True)
class Fact:
    """kind is "isa", "uses", "location", "has" or "can"; value is the other end."""

    kind: str
    value: str
    edge: ConceptEdge


@dataclass(frozen=True)
class Description:
    concept: str
    category: Fact
    properties: tuple[Fact, ...]


def facts_from_edge(edge: ConceptEdge) -> list[tuple[str, Fact]]:
    """(concept, fact) pairs one edge gives; the table in the plan/ADR fixes the direction."""
    if edge.rel == "IsA":
        return [(edge.start, Fact("isa", edge.end, edge))]
    if edge.rel == "UsedFor":
        return [(edge.end, Fact("uses", edge.start, edge))]
    if edge.rel == "AtLocation":
        return [(edge.start, Fact("location", edge.end, edge))]
    if edge.rel == "HasA":
        return [(edge.start, Fact("has", edge.end, edge))]
    if edge.rel == "PartOf":
        return [(edge.end, Fact("has", edge.start, edge))]
    if edge.rel == "CapableOf":
        return [(edge.start, Fact("can", edge.end, edge))]
    return []


def build_fact_index(edges: list[ConceptEdge]) -> dict[str, list[Fact]]:
    """Facts per concept. The same (kind, value) from several edges keeps the heaviest edge."""
    best: dict[tuple[str, str, str], Fact] = {}
    for edge in edges:
        for concept, fact in facts_from_edge(edge):
            key = (concept, fact.kind, fact.value)
            if key not in best or fact.edge.weight > best[key].edge.weight:
                best[key] = fact
    index: dict[str, list[Fact]] = {}
    for (concept, _, _), fact in sorted(best.items()):
        index.setdefault(concept, []).append(fact)
    return index


def has_fact(index: dict[str, list[Fact]], concept: str, kind: str, value: str) -> bool:
    for fact in index.get(concept, []):
        if fact.kind == kind and fact.value == value:
            return True
    return False


def concepts_with(index: dict[str, list[Fact]], fact: Fact, concepts: list[str]) -> list[str]:
    """The concepts (from the given list) that share this fact."""
    sharing = []
    for concept in concepts:
        if has_fact(index, concept, fact.kind, fact.value):
            sharing.append(concept)
    return sharing


def leaks_name(value: str, concept: str) -> bool:
    """True if the value repeats a word of the concept ("tennis ball" for tennis)."""
    return bool(set(value.split()) & set(concept.split()))


def choose_description(
    concept: str,
    pool: list[str],
    index: dict[str, list[Fact]],
    category_words: list[str],
    num_properties: int,
) -> Description | None:
    """One category + num_properties properties that together fit `concept` only.

    pool holds every concept of the same group (not only the targets), so a property
    shared with any of them is not unique. Greedy: always add the property that the
    fewest remaining look-alikes share; ties go to the heavier edge, then the
    alphabet, so the result is deterministic. None if no unique description exists.
    """
    facts = index.get(concept, [])
    others = [other for other in pool if other != concept]

    categories = []
    for fact in facts:
        if fact.kind == "isa" and fact.value in category_words:
            categories.append(fact)
    if not categories:
        return None

    def category_order(fact: Fact) -> tuple:
        return (len(concepts_with(index, fact, others)), fact.value)

    category = min(categories, key=category_order)
    look_alikes = concepts_with(index, category, others)

    candidates = []
    for fact in facts:
        if fact.kind == "isa" or not PLAIN_WORDS.fullmatch(fact.value):
            continue
        if leaks_name(fact.value, concept):
            continue
        candidates.append(fact)

    chosen: list[Fact] = []
    while len(chosen) < num_properties and candidates:

        def property_order(fact: Fact) -> tuple:
            return (len(concepts_with(index, fact, look_alikes)), -fact.edge.weight, fact.value)

        best = min(candidates, key=property_order)
        chosen.append(best)
        candidates.remove(best)
        look_alikes = concepts_with(index, best, look_alikes)

    if len(chosen) < num_properties or look_alikes:
        return None
    return Description(concept, category, tuple(chosen))
```

- [ ] **Step 4: Run, expect PASS** → 5 passed; ruff check + format.

- [ ] **Step 5: Commit**

```bash
git add src/mmxai/queries/kg_facts.py tests/test_kg_facts.py
git commit -m "feat(queries): ConceptNet facts and unique concept descriptions" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Sentences

**Files:**
- Create: `src/mmxai/queries/kg_text.py`
- Test: `tests/test_kg_text.py`

- [ ] **Step 1: Failing tests** `tests/test_kg_text.py`

```python
"""Direct/indirect sentences for KG queries (ADR-0050)."""

from mmxai.queries.kg_facts import build_fact_index, choose_description
from mmxai.queries.kg_text import direct_text, indirect_text, with_article
from conceptnet_fixtures import EDGES

ACTIVITY_TEMPLATES = {"uses": "uses", "location": "is played on", "has": "has", "can": "can"}
OBJECT_TEMPLATES = {"uses": "is used with", "location": "is found at", "has": "has", "can": "can"}


def test_with_article():
    assert with_article("racket") == "a racket"
    assert with_article("apple") == "an apple"


def test_activity_sentences():
    index = build_fact_index(EDGES)
    description = choose_description("tennis", ["tennis", "badminton"], index, ["sport"], 2)

    assert direct_text("activity", "tennis") == "A person playing tennis."
    assert indirect_text("activity", description, ACTIVITY_TEMPLATES) == (
        "A person playing a sport that is played on a court and uses a racket."
    )


def test_object_sentences_keep_verb_values_without_article():
    index = build_fact_index(EDGES)
    description = choose_description("train", ["train", "bus", "car"], index, ["vehicle"], 2)

    assert direct_text("object", "train") == "A train."
    assert indirect_text("object", description, OBJECT_TEMPLATES) == (
        "A vehicle that is found at a railway and can carry passengers."
    )


def test_two_properties_of_one_kind_share_the_verb():
    index = build_fact_index(EDGES)
    description = choose_description("baseball", ["baseball", "tennis"], index, ["sport"], 2)

    assert indirect_text("activity", description, ACTIVITY_TEMPLATES) == (
        "A person playing a sport that uses a bat and a glove."
    )
```

Check the expected orders against `choose_description`'s tie-break before running: for train, `railway` (0 look-alikes) is chosen before `carry passengers`; for baseball both properties have 0 look-alikes and weight 1, so the alphabet gives `bat` then `glove`.

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement** `src/mmxai/queries/kg_text.py`

```python
"""Sentences for KG queries: name the concept (direct) or only describe it (indirect)."""

from mmxai.queries.generator import as_sentence
from mmxai.queries.kg_facts import Description

# "can" is followed by a verb phrase ("can carry passengers"), so it takes no article.
VERB_KINDS = ("can",)


def with_article(term: str) -> str:
    if term[0] in "aeiou":
        return f"an {term}"
    return f"a {term}"


def property_phrase(description: Description, templates: dict[str, str]) -> str:
    """"is played on a court and uses a racket"; values of one kind share the verb."""
    values_by_kind: dict[str, list[str]] = {}
    for fact in description.properties:
        values_by_kind.setdefault(fact.kind, []).append(fact.value)
    phrases = []
    for kind, values in values_by_kind.items():
        if kind in VERB_KINDS:
            objects = values
        else:
            objects = [with_article(value) for value in values]
        phrases.append(f"{templates[kind]} {' and '.join(objects)}")
    return " and ".join(phrases)


def direct_text(group: str, concept: str) -> str:
    if group == "activity":
        return as_sentence(f"a person playing {concept}")
    return as_sentence(with_article(concept))


def indirect_text(group: str, description: Description, templates: dict[str, str]) -> str:
    category = with_article(description.category.value)
    properties = property_phrase(description, templates)
    if group == "activity":
        return as_sentence(f"a person playing {category} that {properties}")
    return as_sentence(f"{category} that {properties}")
```

- [ ] **Step 4: Run, expect PASS** → 4 passed; ruff.

- [ ] **Step 5: Commit**

```bash
git add src/mmxai/queries/kg_text.py tests/test_kg_text.py
git commit -m "feat(queries): direct and indirect sentences for KG queries" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Which images show which concept

**Files:**
- Create: `src/mmxai/queries/kg_targets.py`
- Test: `tests/test_kg_targets.py`

- [ ] **Step 1: Failing tests** `tests/test_kg_targets.py`

```python
"""Concept -> images on scene graphs (ADR-0050). Hand-built synthetic graphs."""

from mmxai.data.scene_graph import Relation, SceneGraph, SceneObject
from mmxai.queries.kg_targets import activity_images, object_images


def graph(image_id: int, people: str, predicate: str, thing: str) -> SceneGraph:
    subject = SceneObject(1, people.split(".")[0], people.split(".")[0], people)
    target = SceneObject(2, thing.split(".")[0], thing.split(".")[0], thing)
    scene = SceneGraph(image_id, {1: subject, 2: target})
    scene.relations.append(
        Relation(10, 1, subject.name, predicate, 2, target.name, predicate, subject.name, target.name)
    )
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
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement** `src/mmxai/queries/kg_targets.py`

```python
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
```

- [ ] **Step 4: Run, expect PASS** → 2 passed; ruff.

- [ ] **Step 5: Commit**

```bash
git add src/mmxai/queries/kg_targets.py tests/test_kg_targets.py
git commit -m "feat(queries): activity and object image lookups for KG queries" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Validator support for KG records

**Files:**
- Modify: `src/mmxai/queries/validator.py` (`MANIFEST_COLUMNS`, `manifest_rows`, new `check_kg_evidence`)
- Test: `tests/test_query_validator.py`

- [ ] **Step 1: Failing tests** (append to `tests/test_query_validator.py`; import `check_kg_evidence`)

```python
def kg_pair() -> list[dict]:
    shared = {
        "dataset": "visual_genome",
        "relevant": ["100"],
        "path_hint": "racket - UsedFor - tennis",
        "author": "test",
        "pair_id": "vgkg-test-a001",
        "split": "test",
        "verification_status": "candidate",
        "hidden_concept": "tennis.n.01",
        "query_group": "activity",
        "kg_evidence": [
            {"uri": "/a/x", "rel": "UsedFor", "start": "racket", "end": "tennis", "weight": 2.0}
        ],
    }
    direct = {"qid": "vgkg-test-a001-d", "text": "A person playing tennis.", "type": "direct"}
    direct.update(shared, reasoning_type="explicit", hidden_names=[])
    indirect = {
        "qid": "vgkg-test-a001-i",
        "text": "A person playing a sport that uses a racket.",
        "type": "indirect",
    }
    indirect.update(shared, reasoning_type="kg_description", hidden_names=["tennis"])
    return [direct, indirect]


def test_manifest_rows_accept_kg_records():
    rows = manifest_rows(kg_pair())

    assert rows[0]["query_group"] == "activity"
    assert rows[0]["hidden_concept"] == "tennis.n.01"
    assert rows[0]["kg_edges"] == "/a/x"
    assert rows[0]["relationship_ids"] == ""
    assert list(rows[0]) == MANIFEST_COLUMNS


def test_kg_pair_passes_the_generic_checks():
    assert validate_queries(kg_pair(), {"100"}) == []


def test_check_kg_evidence_rejects_unknown_edges():
    assert check_kg_evidence(kg_pair(), {"/a/x"}) == []
    problems = check_kg_evidence(kg_pair(), {"/a/other"})
    assert len(problems) == 2
    assert "/a/x" in problems[0]
```

- [ ] **Step 2: Run, expect FAIL** (ImportError / KeyError `source_image_id`)

Run: `python -m uv run pytest -q tests/test_query_validator.py`

- [ ] **Step 3: Implement.** In `MANIFEST_COLUMNS` insert after `"relationship_ids",`:

```python
    "query_group",
    "hidden_concept",
    "kg_edges",
```

In `manifest_rows`, replace the `relationship_ids = ...` line and the dict fields that assume v1 with:

```python
        # v1 records cite scene-graph relations, KG records cite ConceptNet edges (ADR-0050).
        relationship_ids = []
        for item in direct.get("relation_evidence", []):
            relationship_ids.append(str(item["relationship_id"]))
        kg_edges = []
        for item in direct.get("kg_evidence", []):
            kg_edges.append(item["uri"])
```

and in the row dict use `direct.get("source_image_id", "")`, `direct.get("hop_count", "")`, `indirect.get("reasoning_type", "")`, `" | ".join(indirect.get("hidden_names", []))`, plus

```python
                "relationship_ids": " ".join(relationship_ids),
                "query_group": direct.get("query_group", ""),
                "hidden_concept": direct.get("hidden_concept", ""),
                "kg_edges": " ".join(kg_edges),
```

Keep the key order identical to `MANIFEST_COLUMNS`. Add after `check_evidence`:

```python
def check_kg_evidence(records: list[dict], edge_uris: set[str]) -> list[str]:
    """Every ConceptNet edge a KG query cites must exist in the filtered edge file."""
    problems = []
    for record in records:
        for item in record.get("kg_evidence", []):
            if item["uri"] not in edge_uris:
                problems.append(f"{record['qid']}: ConceptNet edge {item['uri']} not found")
    return problems
```

- [ ] **Step 4: Run, expect PASS** — `python -m uv run pytest -q tests/test_query_validator.py tests/test_query_generator.py` all pass (v1 tests must still pass: v1 rows get `""` in the 3 new columns). Ruff.

- [ ] **Step 5: Commit**

```bash
git add src/mmxai/queries/validator.py tests/test_query_validator.py
git commit -m "feat(queries): review manifest and evidence check for KG queries" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Generator run loop

**Files:**
- Modify: `src/mmxai/data/scene_graph.py` (extract `load_annotation_files`)
- Create: `src/mmxai/queries/kg_generator.py`
- Test: `tests/test_kg_generator.py`

- [ ] **Step 1: Refactor `scene_graph.py` (no behaviour change).** Split the file-reading half of `load_scene_graphs` into:

```python
def load_annotation_files(
    data_dir: str | Path,
    objects_file: str = OBJECTS_FILE,
    relationships_file: str = RELATIONSHIPS_FILE,
) -> tuple[list[dict], list[dict]]:
    """Read objects.json and relationships.json once (several GB of RAM for v1.4)."""
```

containing the existing existence check and the two `json.load` calls, and make `load_scene_graphs` call it and then `build_scene_graphs`. Run `python -m uv run pytest -q tests/test_scene_graph.py tests/test_query_validator.py` → still pass.

- [ ] **Step 2: Failing tests** `tests/test_kg_generator.py`

```python
"""KG query generation end to end on synthetic data (ADR-0050)."""

from dataclasses import replace

from mmxai.data.scene_graph import Relation, SceneGraph, SceneObject
from mmxai.queries.kg_facts import build_fact_index
from mmxai.queries.kg_generator import KgQueryConfig, generate_kg_pairs
from conceptnet_fixtures import EDGES

CONFIG = KgQueryConfig(
    dataset="visual_genome",
    data_dir="unused",
    objects_file="objects.json",
    relationships_file="relationships.json",
    split_file="unused",
    split="test",
    conceptnet_edges="unused",
    predicate_aliases={"plays": "playing"},
    activity_predicates=["playing"],
    person_synsets=["man.n.01"],
    category_words=["sport", "vehicle"],
    activity_templates={"uses": "uses", "location": "is played on", "has": "has", "can": "can"},
    object_templates={"uses": "is used with", "location": "is found at", "has": "has", "can": "can"},
    num_properties=2,
    min_train_images=1,
    min_answer_images=1,
    max_answer_images=10,
    limit_activity_pairs=5,
    limit_object_pairs=5,
    author="test",
    candidates_path="unused",
    manifest_path="unused",
    benchmark_path="unused",
    min_verified_pairs=1,
)


def scene(image_id: int, thing: str, predicate: str = "playing") -> SceneGraph:
    man = SceneObject(1, "man", "man", "man.n.01")
    target = SceneObject(2, thing.split(".")[0], thing.split(".")[0], thing)
    graph = SceneGraph(image_id, {1: man, 2: target})
    graph.relations.append(
        Relation(image_id, 1, "man", predicate, 2, target.name, predicate, "man", target.name)
    )
    return graph


def fixture_graphs() -> tuple[dict, dict]:
    """Canonical graphs, as run_kg_generation passes them ("plays" becomes "playing")."""
    train = {
        1: scene(1, "tennis.n.01"),
        2: scene(2, "badminton.n.01"),
        3: scene(3, "train.n.01", "near"),
        4: scene(4, "car.n.01", "near"),
        5: scene(5, "bus.n.01", "near"),  # makes car ambiguous (shared facts)
    }
    test = {
        100: scene(100, "tennis.n.01"),
        101: scene(101, "tennis.n.01", "plays"),
        102: scene(102, "train.n.01", "near"),
        103: scene(103, "car.n.01", "near"),  # car has answers, so it fails on the description
    }
    aliases = CONFIG.predicate_aliases
    return canonicalize_graphs(train, aliases), canonicalize_graphs(test, aliases)


def test_pairs_hide_the_concept_and_share_answers():
    train, test = fixture_graphs()

    records, rejections = generate_kg_pairs(train, test, build_fact_index(EDGES), CONFIG)

    tennis = [record for record in records if record.get("hidden_concept") == "tennis.n.01"]
    direct, indirect = sorted(tennis, key=lambda record: record["type"])
    assert direct["text"] == "A person playing tennis."
    assert "tennis" not in indirect["text"].lower()
    # "plays" is canonicalised to "playing", so image 101 is an answer too.
    assert direct["relevant"] == indirect["relevant"] == ["100", "101"]
    assert indirect["query_group"] == "activity"
    assert {item["start"] for item in indirect["kg_evidence"]} >= {"racket"}


def test_concepts_without_unique_description_or_answers_are_rejected():
    train, test = fixture_graphs()

    records, rejections = generate_kg_pairs(train, test, build_fact_index(EDGES), CONFIG)

    concepts = {record["hidden_concept"] for record in records}
    assert "car.n.01" not in concepts  # every fact shared with bus/car look-alikes
    assert "badminton.n.01" not in concepts  # no test image
    assert rejections["no_unique_description"] >= 1
    assert rejections["answer_count_out_of_range"] >= 1


def test_answers_never_leave_the_given_split():
    train, test = fixture_graphs()

    records, _ = generate_kg_pairs(train, test, build_fact_index(EDGES), CONFIG)

    for record in records:
        assert set(record["relevant"]) <= {str(image_id) for image_id in test}


def test_limits_per_group():
    train, test = fixture_graphs()
    config = replace(CONFIG, limit_activity_pairs=0)

    records, _ = generate_kg_pairs(train, test, build_fact_index(EDGES), config)

    assert {record["query_group"] for record in records} == {"object"}
```

Add `from mmxai.queries.generator import canonicalize_graphs` to the imports. Hand-derived expectations: activity targets are badminton (no test answer → `answer_count_out_of_range`) and tennis (answers 100, 101; description court + racket). Object targets: `man` has no ConceptNet facts (`no_conceptnet_facts`), badminton and bus have no test answer, car has an answer but every fact is shared with bus (`no_unique_description`), tennis is already used (`concept_already_used`), train gets railway + carry passengers. With `limit_activity_pairs=0`, tennis is generated in the object group instead.

- [ ] **Step 3: Run, expect FAIL** (ModuleNotFoundError)

- [ ] **Step 4: Implement** `src/mmxai/queries/kg_generator.py`

```python
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

        def train_order(synset: str) -> tuple:
            return (-len(train_images[synset]), synset)

        num_pairs = 0
        for synset in sorted(train_images, key=train_order):
            if num_pairs >= limit:
                break
            if len(train_images[synset]) < config.min_train_images:
                rejections[REJECT_TOO_RARE] += 1
                continue
            term = synset_lemma(synset)
            if synset in used:
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
            used.add(synset)
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
    eval_graphs, parse_report = build_scene_graphs(
        objects_entries, relationships_entries, eval_ids
    )
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
```

- [ ] **Step 5: Run, expect PASS** — `python -m uv run pytest -q tests/test_kg_generator.py tests/test_kg_facts.py tests/test_kg_text.py tests/test_kg_targets.py`. If an expected sentence or rejection count differs, re-derive it by hand from `conceptnet_fixtures.EDGES` before changing either the test or the code, and say which in the commit message. Ruff.

- [ ] **Step 6: Commit**

```bash
git add src/mmxai/data/scene_graph.py src/mmxai/queries/kg_generator.py tests/test_kg_generator.py
git commit -m "feat(queries): generate KG direct/indirect query pairs" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Metrics per query group

**Files:**
- Modify: `src/mmxai/eval/query_benchmark.py` (`score_rankings`)
- Test: `tests/test_query_benchmark_eval.py`

- [ ] **Step 1: Failing test** (append; read the file first to reuse its helpers/imports for `score_rankings`)

```python
def test_scores_are_split_by_query_group():
    queries = [
        {"qid": "a-d", "type": "direct", "relevant": ["1"], "query_group": "activity"},
        {"qid": "a-i", "type": "indirect", "relevant": ["1"], "query_group": "activity"},
        {"qid": "o-d", "type": "direct", "relevant": ["2"], "query_group": "object"},
    ]
    retrieved = np.array([[1, 2], [2, 1], [3, 4]])

    metrics = score_rankings(queries, retrieved, recall_ks=[1], top_k=2)

    assert metrics["by_query_group"]["activity"]["direct"]["R@1"] == 1.0
    assert metrics["by_query_group"]["activity"]["indirect"]["R@1"] == 0.0
    assert metrics["by_query_group"]["object"]["indirect"] is None
    assert metrics["per_query"][0]["query_group"] == "activity"
```

Check the key name `compute_metrics` returns for recall@1 (`R@1` or similar) in `src/mmxai/eval/metrics.py` and use that exact key.

- [ ] **Step 2: Run, expect FAIL** (KeyError `by_query_group`)

- [ ] **Step 3: Implement.** In `score_rankings`, inside the loop also collect

```python
        group = query.get("query_group")
        if group:
            ranks_by_query_group.setdefault(group, {"direct": [], "indirect": []})
            ranks_by_query_group[group][query["type"]].append(rank)
```

(initialise `ranks_by_query_group: dict[str, dict[str, list]] = {}` before the loop), add `"query_group": query.get("query_group")` to each `per_query` item, and before returning:

```python
    # ADR-0050: activity and object queries are reported apart; v1 files have no groups.
    metrics["by_query_group"] = {}
    for group, ranks_by_type in sorted(ranks_by_query_group.items()):
        metrics["by_query_group"][group] = {}
        for query_type, ranks in ranks_by_type.items():
            if ranks:
                metrics["by_query_group"][group][query_type] = compute_metrics(
                    ranks, recall_ks, top_k
                )
            else:
                metrics["by_query_group"][group][query_type] = None
```

- [ ] **Step 4: Run, expect PASS** — `python -m uv run pytest -q tests/test_query_benchmark_eval.py tests/test_eval_query_groups.py tests/test_eval_query_ranks.py`. If an existing test compares the full metrics dict, add the new key to its expectation (an empty `{}` for v1 queries) and say so in the commit. Ruff.

- [ ] **Step 5: Commit**

```bash
git add src/mmxai/eval/query_benchmark.py tests/test_query_benchmark_eval.py
git commit -m "feat(eval): query benchmark metrics per query group" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: CLIs and configs

**Files:**
- Create: `scripts/generate_kg_queries.py`, `scripts/build_kg_query_benchmark.py`
- Create: `configs/query_benchmark_vg_kg.yaml`, `configs/query_benchmark_vg_kg_pilot.yaml`, `configs/eval_query_benchmark_vg_kg.yaml`
- Modify: `tests/test_query_configs.py`, `configs/README.md`, `scripts/README.md`

- [ ] **Step 1: Failing test** (append to `tests/test_query_configs.py`; reuse its `ALLOWED_TO_DIFFER` and path style)

```python
def test_kg_pilot_and_test_configs_share_every_rule():
    from mmxai.queries.kg_generator import load_kg_query_config

    test_config = load_kg_query_config(REPO / "configs/query_benchmark_vg_kg.yaml")
    pilot_config = load_kg_query_config(REPO / "configs/query_benchmark_vg_kg_pilot.yaml")

    allowed = {"split", "candidates_path", "manifest_path", "benchmark_path"}
    for field in test_config.__dataclass_fields__:
        if field in allowed:
            continue
        assert getattr(test_config, field) == getattr(pilot_config, field), field
```

(`REPO` = whatever the file already uses for the repo root, e.g. `Path(__file__).parent.parent`; read the file first.)

- [ ] **Step 2: Run, expect FAIL** (FileNotFoundError)

- [ ] **Step 3: Write the configs.** `configs/query_benchmark_vg_kg.yaml`:

```yaml
# KG query benchmark for Visual Genome, test split (ADR-0050, status: Proposed).
# How to run: see scripts/generate_kg_queries.py and scripts/build_kg_query_benchmark.py.
# Every value below must be fixed on the val pilot BEFORE generating on test, and never
# changed after looking at retrieval results (PDF section 6, ADR-0009).
dataset: visual_genome
data_dir: data/visual_genome
objects_file: objects.json
relationships_file: relationships.json
split_file: splits/visual_genome/split_seed0.json
split: test
conceptnet_edges: artifacts/conceptnet/en_edges.tsv
# Same canonicalisation as ADR-0044, only for the activity predicate.
predicate_aliases: {plays: playing, play: playing, playing with: playing}
activity_predicates: [playing]
# Decision (ADR-0050, proposed): subjects that count as "a person".
person_synsets: [man.n.01, woman.n.01, boy.n.01, girl.n.01, child.n.01, person.n.01,
                 player.n.01, guy.n.01, lady.n.01, people.n.01, kid.n.01]
# Decision (ADR-0050, proposed): category words an indirect query may name.
category_words: [sport, game, activity, vehicle, animal, instrument, fruit, vegetable,
                 furniture, tool, food, toy, appliance, device, container, clothing]
activity_templates: {uses: uses, location: is played on, has: has, can: can}
object_templates: {uses: is used with, location: is found at, has: has, can: can}
num_properties: 2
# Decision (ADR-0050, proposed): thresholds, fixed on the val pilot.
min_train_images: 20
min_answer_images: 3
max_answer_images: 50
limit_activity_pairs: 10
limit_object_pairs: 20
author: kg-template-v1
candidates_path: queries/visual_genome_kg_candidates.jsonl
manifest_path: queries/query_review_manifest_kg.csv
benchmark_path: queries/visual_genome_kg_test.jsonl
# PDF Task 2: at least 20 queries = 10 direct/indirect pairs.
min_verified_pairs: 10
```

`configs/query_benchmark_vg_kg_pilot.yaml`: identical except

```yaml
# Val PILOT of the KG generator (ADR-0050): check rules and templates without touching test.
split: val
candidates_path: results/runs/query_kg_pilot_val/visual_genome_kg_candidates_val.jsonl
manifest_path: results/runs/query_kg_pilot_val/query_review_manifest_kg_val.csv
benchmark_path: results/runs/query_kg_pilot_val/unused_benchmark.jsonl  # never built from val
```

(The val split also has 5,000 images, so the same answer thresholds apply.)

`configs/eval_query_benchmark_vg_kg.yaml`: copy `configs/eval_query_benchmark_vg.yaml`, change `experiment: clip_vg_kg_query_benchmark` and `benchmark_path: queries/visual_genome_kg_test.jsonl`.

- [ ] **Step 4: Write the CLIs.** `scripts/generate_kg_queries.py`:

```python
"""Generate candidate KG direct/indirect query pairs for Visual Genome (ADR-0050).

Logic: mmxai.queries.kg_generator.

Before running:
    uv sync
    data/visual_genome/objects.json, data/visual_genome/relationships.json
    artifacts/conceptnet/en_edges.tsv (scripts/prepare_conceptnet.py)

Run (from the repo root):
    Pilot on val (output in results/runs/, gitignored):
        uv run python scripts/generate_kg_queries.py --config configs/query_benchmark_vg_kg_pilot.yaml
    Candidates on test, only after the team fixed the config (writes into queries/):
        uv run python scripts/generate_kg_queries.py --config configs/query_benchmark_vg_kg.yaml
    Both refuse to overwrite existing outputs.

Output:
    <candidates_path> (JSONL) and <manifest_path> (CSV). Next: two members review the
    manifest, then scripts/build_kg_query_benchmark.py.
"""

import argparse
import json

from mmxai.queries.kg_generator import load_kg_query_config, run_kg_generation


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="YAML file in configs/")
    args = parser.parse_args()

    summary = run_kg_generation(load_kg_query_config(args.config))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
```

`scripts/build_kg_query_benchmark.py`: copy `scripts/build_query_benchmark.py`, replace the import with `from mmxai.queries.kg_generator import load_kg_query_config`, call `load_kg_query_config`, and update the docstring to name the KG configs and output `queries/visual_genome_kg_test.jsonl`. Wrap docstring lines > 100 chars.

Add one line per new config to `configs/README.md` and per new script to `scripts/README.md`, following their existing format.

- [ ] **Step 5: Run, expect PASS** — `QTESTS` all pass; `python -m uv run python scripts/generate_kg_queries.py --help` prints the docstring. Ruff on new Python files.

- [ ] **Step 6: Commit**

```bash
git add scripts/generate_kg_queries.py scripts/build_kg_query_benchmark.py configs/query_benchmark_vg_kg.yaml configs/query_benchmark_vg_kg_pilot.yaml configs/eval_query_benchmark_vg_kg.yaml tests/test_query_configs.py configs/README.md scripts/README.md
git commit -m "feat(queries): CLIs and configs for the KG query benchmark" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Docs, full checks, optional val pilot

**Files:**
- Modify: `docs/query_benchmark.md`, `docs/decisions/0050-truy-van-gian-tiep-mo-ta-khai-niem-qua-conceptnet.md`, `docs/ai-usage-log.md`

- [ ] **Step 1: ADR fix.** In ADR-0050, row "Mô tả duy nhất", the text says "tất định theo seed"; the code is deterministic without a seed (targets by train frequency, ties by weight then alphabet). Change it to "tất định: duyệt khái niệm theo tần suất trên train, hoà thì theo trọng số cạnh rồi theo thứ tự chữ". Also add the file names actually created (`src/mmxai/data/conceptnet.py`, `src/mmxai/queries/kg_*.py`, the 4 configs, 3 scripts) in a "Code:" line under the decision table.

- [ ] **Step 2: Docs.** Add a section "Bộ truy vấn v2: khái niệm mô tả qua ConceptNet (ADR-0050)" to `docs/query_benchmark.md` (Vietnamese, beginner-first): khái niệm; bảng fact kinds; cách chọn mô tả duy nhất; ví dụ **tổng hợp** từ test fixture (ghi rõ là dữ liệu giả); lệnh chạy theo thứ tự (prepare → pilot val → chốt config → test → review → build → evaluate); lưu ý rò rỉ (mô hình đồ thị không dựng cạnh từ annotation ảnh test). Add one row to `docs/ai-usage-log.md` (skill `/log-ai-usage`), column "Đã tự kiểm tra/hiểu chưa" = `Chưa`.

- [ ] **Step 3: Full checks**

Run: `python -m uv run ruff check` → only the 2 known I001.
Run: `python -m uv run ruff format --check` → only `.md` plan files may be listed.
Run: `python -m uv run pytest -q` → only the 10 known failures.

- [ ] **Step 4: Optional val pilot (only if `data/conceptnet/conceptnet-assertions-5.7.0.csv.gz` exists)**

```bash
python -m uv run python scripts/prepare_conceptnet.py --config configs/conceptnet_en.yaml
python -m uv run python scripts/generate_kg_queries.py --config configs/query_benchmark_vg_kg_pilot.yaml
```

Print the pilot pairs (`type`, `query_group`, number of answers, `text`, `path_hint`) and show them to the user. Do NOT tune anything yet; the user decides rule changes (val only). Never run the test config.

- [ ] **Step 5: Commit**

```bash
git add docs/query_benchmark.md docs/decisions/0050-truy-van-gian-tiep-mo-ta-khai-niem-qua-conceptnet.md docs/ai-usage-log.md
git commit -m "docs(queries): document the KG query benchmark v2" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Out of scope
- Generating on test, human review, building `queries/visual_genome_kg_test.jsonl`, CLIP/GraphSAGE/G-Retriever evaluation runs.
- Flickr30k answers (only the answer lookup will need a Flickr30k version later).
- Using ConceptNet edges inside a graph model (G-Retriever plan).
