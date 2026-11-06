# VG Query Rules v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply the ADR-0044 revisions from the 2026-11-06 val pilot: predicate whitelist + hand-written templates, only "shared_object"/"path" 2-hop chains, ≥ 1 action predicate per chain, plural-aware articles, and candidate relevance matched by VG WordNet synsets.

**Architecture:** The parser (`scene_graph.py`) additionally reads each object's first synset. The generator canonicalises predicates once per run (strip leading "is/are", trailing article, apply aliases), filters by whitelist/shape/action, and writes text from per-shape templates. The validator matches chains across images by `match_key` (synset, else name) and reads the same keys back from `relation_evidence`, so answers stay recomputable from the JSONL alone.

**Tech Stack:** Python 3.11, numpy, PyYAML, pytest, ruff. No new dependency.

**Spec:** `docs/decisions/0044-bo-truy-van-danh-gia-tu-scene-graph-visual-genome.md` (rows *Chuỗi, Vị từ, Lọc, Sinh câu, Đáp án ứng viên* and section *Pilot val lần 1*).

**Repo rules (CLAUDE.md):** branch `feat/vg-query-benchmark` (already checked out). Run every command as `python -m uv run ...` (uv is not on PATH on this machine). Commit only if the user has agreed to commit; never push. Code/comments in English, docs in Vietnamese. Never generate on the **test** split in this plan — only the val pilot.

---

## File map

| File | Change |
|---|---|
| `src/mmxai/data/scene_graph.py` | `SceneObject.synset`, `first_synset()`, `match_key()` |
| `src/mmxai/queries/generator.py` | config fields, `canonical_predicate`, `canonicalize_graphs`, whitelist filter, shape + action filter, new templates; remove `POSITION_WORDS`, `verb_phrase`, `describe_node`, `with_article`, `trivial_predicates` |
| `src/mmxai/queries/validator.py` | match by key (synset or name); read keys from evidence |
| `configs/query_benchmark_vg.yaml`, `configs/query_benchmark_vg_pilot.yaml` | new keys, identical rules |
| `tests/scene_graph_fixtures.py` | new config fields, optional synset in `obj()` |
| `tests/test_scene_graph.py`, `tests/test_query_generator.py`, `tests/test_query_validator.py` | new/updated tests |
| `tests/test_query_configs.py` | new: pilot and test configs share every rule |
| `docs/query_benchmark.md`, `docs/ai-usage-log.md` | docs |

---

### Task 1: Read the object synset in the parser

**Files:**
- Modify: `src/mmxai/data/scene_graph.py` (`SceneObject` ~line 45, `parse_object` ~line 106)
- Test: `tests/test_scene_graph.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_scene_graph.py`; add `match_key` to its import from `mmxai.data.scene_graph`)

```python
def test_object_synset_is_read_and_used_as_match_key():
    objects = [
        {
            "image_id": 1,
            "objects": [
                {"object_id": 1, "names": ["guy"], "synsets": ["man.n.01"]},
                {"object_id": 2, "names": ["horse"]},
            ],
        }
    ]

    graphs, _ = build_scene_graphs(objects, [], image_ids=None)

    assert graphs[1].objects[1].synset == "man.n.01"
    assert match_key(graphs[1].objects[1]) == "man.n.01"
    # No synset annotated: fall back to the normalised name.
    assert graphs[1].objects[2].synset == ""
    assert match_key(graphs[1].objects[2]) == "horse"
```

- [ ] **Step 2: Run it, expect FAIL** (`ImportError: cannot import name 'match_key'`)

Run: `python -m uv run pytest tests/test_scene_graph.py -q`

- [ ] **Step 3: Implement**

In `SceneObject`, add a defaulted last field (existing `SceneObject(id, name, raw)` calls keep working):

```python
@dataclass(frozen=True)
class SceneObject:
    """One annotated object; `name` is normalised, `raw_name` is what the annotator typed.

    `synset` is the first WordNet synset VG attached to the object ("man.n.01"), or "".
    """

    object_id: int
    name: str
    raw_name: str
    synset: str = ""
```

After `first_raw_name`, add:

```python
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
```

In `parse_object`, change the return to:

```python
    return SceneObject(object_id, name, raw_name, first_synset(annotation))
```

- [ ] **Step 4: Run, expect PASS**

Run: `python -m uv run pytest tests/test_scene_graph.py -q`

- [ ] **Step 5: Commit (only if agreed)**

```bash
git add src/mmxai/data/scene_graph.py tests/test_scene_graph.py
git commit -m "feat(data): read object synsets from Visual Genome scene graphs"
```

---

### Task 2: Config fields and canonical predicates

**Files:**
- Modify: `src/mmxai/queries/generator.py` (`QueryBenchmarkConfig`, `load_query_benchmark_config`, `usable_relations`, `run_generation`, constants)
- Modify: `tests/scene_graph_fixtures.py` (`CONFIG`)
- Test: `tests/test_query_generator.py`

- [ ] **Step 1: Update the test fixture config** in `tests/scene_graph_fixtures.py`: replace `trivial_predicates=["of"],` with

```python
    predicate_aliases={"looks at": "looking at", "rides": "riding"},
    predicate_templates={
        "holding": "is holding",
        "looking at": "is looking at",
        "riding": "is riding",
        "wearing": "is wearing",
        "sitting on": "is sitting on",
        "on": "is on",
        "near": "is near",
    },
    action_predicates=["holding", "looking at", "riding", "wearing", "sitting on"],
```

Also add to `tests/scene_graph_fixtures.py` a helper that returns graphs the way `run_generation` sees them (canonical predicates — the fixture's raw "looks_at" must become "looking at", otherwise it is not in the whitelist):

```python
def canonical_fixture_graphs(image_ids: list[int]) -> dict[int, SceneGraph]:
    """Fixture graphs of the given images, with predicates canonicalised as in run_generation."""
    graphs, _ = build_scene_graphs(*synthetic_annotations(), image_ids=set(image_ids))
    return canonicalize_graphs(graphs, CONFIG.predicate_aliases)
```

(imports: `from mmxai.data.scene_graph import SceneGraph, build_scene_graphs` and `canonicalize_graphs` from `mmxai.queries.generator`.) Then:
- in `tests/test_query_generator.py`, make `fixture_graphs` return `canonical_fixture_graphs(image_ids)`;
- in `tests/test_query_validator.py`, replace every `graphs, _ = build_scene_graphs(*synthetic_annotations(), image_ids=set(TEST_IDS))` (e.g. line ~40) with `graphs = canonical_fixture_graphs(TEST_IDS)`. `grep -n build_scene_graphs tests/test_query_validator.py` lists them.

- [ ] **Step 2: Write failing tests** in `tests/test_query_generator.py`. Replace the import of `REJECT_TRIVIAL_PREDICATE` and `verb_phrase` with `REJECT_PREDICATE_NOT_ALLOWED`, `canonical_predicate`, `canonicalize_graphs`. Replace `test_ambiguous_names_generic_names_and_trivial_predicates_are_rejected` and delete `test_verb_phrase_adds_is_only_where_needed`. Add:

```python
def test_canonical_predicate_strips_copula_article_and_applies_aliases():
    aliases = {"wears": "wearing"}

    assert canonical_predicate("on a", aliases) == "on"
    assert canonical_predicate("are on", aliases) == "on"
    assert canonical_predicate("is sitting on the", aliases) == "sitting on"
    assert canonical_predicate("wears", aliases) == "wearing"
    # Only whole trailing words are articles: "near" must not lose its "a".
    assert canonical_predicate("near", aliases) == "near"


def test_canonicalize_graphs_keeps_the_raw_predicate():
    graph = make_graph([(1, 1, "wears", 2)], {1: "man", 2: "hat"})

    canonical = canonicalize_graphs({1: graph}, {"wears": "wearing"})

    assert canonical[1].relations[0].predicate == "wearing"
    assert canonical[1].relations[0].raw_predicate == "wears"
    assert graph.relations[0].predicate == "wears"  # input left unchanged


def test_ambiguous_generic_and_non_whitelisted_relations_are_rejected():
    names = {1: "tree", 2: "tree", 3: "dog", 4: "it", 5: "leg"}
    graph = make_graph([(1, 1, "near", 3), (2, 4, "on", 3), (3, 5, "of", 3)], names)
    rejections = Counter()

    usable = usable_relations(graph, ["it"], {"near": "is near", "on": "is on"}, rejections)

    assert usable == []
    assert rejections == Counter(
        {REJECT_AMBIGUOUS_NAME: 1, REJECT_GENERIC_NAME: 1, REJECT_PREDICATE_NOT_ALLOWED: 1}
    )


def test_config_rejects_action_predicates_without_template(tmp_path):
    values = {field: getattr(CONFIG, field) for field in CONFIG.__dataclass_fields__}
    values["action_predicates"] = ["holding", "flying"]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")

    with pytest.raises(ValueError, match="flying"):
        load_query_benchmark_config(path)


def test_config_rejects_more_than_two_hops(tmp_path):
    values = {field: getattr(CONFIG, field) for field in CONFIG.__dataclass_fields__}
    values["max_hops"] = 3
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")

    with pytest.raises(ValueError, match="2 hops"):
        load_query_benchmark_config(path)
```

- [ ] **Step 3: Run, expect FAIL** (import errors)

Run: `python -m uv run pytest tests/test_query_generator.py -q`

- [ ] **Step 4: Implement in `generator.py`**

Add `replace` to the dataclasses import: `from dataclasses import dataclass, replace`.

Delete `POSITION_WORDS`. Replace `REJECT_TRIVIAL_PREDICATE = "trivial_predicate"` with:

```python
REJECT_PREDICATE_NOT_ALLOWED = "predicate_not_in_whitelist"
```

Add after the reject constants:

```python
# Removed before the whitelist lookup so "are on a" and "on" are one predicate.
LEADING_COPULAS = ("is ", "are ")
TRAILING_ARTICLES = (" a", " an", " the")
```

In `QueryBenchmarkConfig`, replace `trivial_predicates: list[str]` with:

```python
    # canonical predicate -> hand-written verb phrase; predicates not listed are dropped.
    predicate_templates: dict[str, str]
    # variant -> canonical predicate, e.g. "wears" -> "wearing".
    predicate_aliases: dict[str, str]
    # Subset of predicate_templates that are actions; every chain needs at least one.
    action_predicates: list[str]
```

In `load_query_benchmark_config`, replace the two hop checks with:

```python
    if config.min_hops != 2 or config.max_hops != 2:
        raise ValueError("min_hops and max_hops must be 2: the templates cover 2 hops only")
    unknown_actions = sorted(set(config.action_predicates) - set(config.predicate_templates))
    if unknown_actions:
        raise ValueError(f"action_predicates without a template: {unknown_actions}")
    unknown_targets = sorted(set(config.predicate_aliases.values()) - set(config.predicate_templates))
    if unknown_targets:
        raise ValueError(f"predicate_aliases point to predicates without a template: {unknown_targets}")
```

Then update `test_config_rejects_fewer_than_two_hops` to `match="2 hops"`.

Add before `usable_relations`:

```python
def canonical_predicate(predicate: str, aliases: dict[str, str]) -> str:
    """One key per meaning: "are on a" -> "on", "wears" -> "wearing" (with that alias)."""
    for copula in LEADING_COPULAS:
        if predicate.startswith(copula):
            predicate = predicate[len(copula) :]
    for article in TRAILING_ARTICLES:
        if predicate.endswith(article):
            predicate = predicate[: -len(article)]
    return aliases.get(predicate, predicate)


def canonicalize_graphs(
    graphs: dict[int, SceneGraph], aliases: dict[str, str]
) -> dict[int, SceneGraph]:
    """Copies of the graphs with canonical predicates; raw_predicate keeps the annotation.

    Done once per run so that generation, answer matching and the evidence check
    all compare the same predicate strings.
    """
    result = {}
    for image_id, graph in graphs.items():
        relations = []
        for relation in graph.relations:
            predicate = canonical_predicate(relation.predicate, aliases)
            relations.append(replace(relation, predicate=predicate))
        result[image_id] = SceneGraph(image_id, graph.objects, relations)
    return result
```

Change `usable_relations` signature and the predicate branch:

```python
def usable_relations(
    graph: SceneGraph,
    generic_names: list[str],
    predicate_templates: dict[str, str],
    rejections: Counter,
) -> list[Relation]:
```

Docstring bullet: `- predicate in the whitelist (predicate_templates): every query reads correctly;`. Body: replace `trivial = set(trivial_predicates)` (delete it) and

```python
        elif relation.predicate not in predicate_templates:
            rejections[REJECT_PREDICATE_NOT_ALLOWED] += 1
```

In `generate_pairs`, change the call to `usable_relations(graph, config.generic_names, config.predicate_templates, rejections)`.

In `run_generation`, right after `load_scene_graphs(...)`:

```python
    graphs = canonicalize_graphs(graphs, config.predicate_aliases)
```

- [ ] **Step 5: Run the new tests, expect PASS**

Run: `python -m uv run pytest tests/test_query_generator.py -q -k "canonical or rejected or config"`

(Text-level tests still fail until Task 4 — expected.)

- [ ] **Step 6: Commit (only if agreed)**

```bash
git add src/mmxai/queries/generator.py tests/scene_graph_fixtures.py tests/test_query_generator.py
git commit -m "feat(queries): predicate whitelist with canonical predicates"
```

---

### Task 3: Chain shape and action filters

**Files:**
- Modify: `src/mmxai/queries/generator.py` (`chain_problem`, `generate_pairs`)
- Test: `tests/test_query_generator.py`

- [ ] **Step 1: Write failing tests** (import `REJECT_SHAPE`, `REJECT_NO_ACTION`, `chain_problem`)

```python
def test_shared_subject_chains_are_rejected():
    graph = make_graph(
        [(1, 1, "holding", 2), (2, 1, "wearing", 3)], {1: "man", 2: "umbrella", 3: "hat"}
    )
    chain = enumerate_chains(1, graph.relations, 2, 2)[0]

    assert reasoning_type(chain) == "shared_subject"
    assert chain_problem(chain, graph, set(), CONFIG) == REJECT_SHAPE


def test_chain_without_an_action_predicate_is_rejected():
    graph = make_graph([(1, 1, "on", 2), (2, 3, "near", 2)], {1: "cup", 2: "table", 3: "plate"})
    chain = enumerate_chains(1, graph.relations, 2, 2)[0]

    assert reasoning_type(chain) == "shared_object"
    assert chain_problem(chain, graph, set(), CONFIG) == REJECT_NO_ACTION


def test_path_chain_with_an_action_is_accepted():
    graph = make_graph([(1, 1, "riding", 2), (2, 2, "on", 3)], {1: "man", 2: "horse", 3: "beach"})
    chain = enumerate_chains(1, graph.relations, 2, 2)[0]

    assert reasoning_type(chain) == "path"
    assert chain_problem(chain, graph, set(), CONFIG) is None
```

- [ ] **Step 2: Run, expect FAIL** (import error)

- [ ] **Step 3: Implement.** Add constants:

```python
REJECT_SHAPE = "shared_subject_chain"
REJECT_NO_ACTION = "no_action_predicate"
# Pilot val 2026-11-06: shared-subject chains gave vague text and joined two names of
# one person ("baseball player" / "batter"), so only these shapes are kept (ADR-0044).
ALLOWED_SHAPES = ("shared_object", "path")
```

Replace `chain_problem` with:

```python
def chain_problem(
    chain: Chain, graph: SceneGraph, seen_texts: set[str], config: QueryBenchmarkConfig
) -> str | None:
    """Return why a chain cannot become a query pair, or None if it can (cheap checks only)."""
    if reasoning_type(chain) not in ALLOWED_SHAPES:
        return REJECT_SHAPE
    # Only positions ("cup on table near plate") is not the "child - plays - football"
    # kind of relation the PDF asks the indirect group to need.
    actions = set(config.action_predicates)
    if not any(relation.predicate in actions for relation in chain.relations):
        return REJECT_NO_ACTION
    # No check for two chain nodes with one name: usable_relations already keeps only
    # names that are unique in the image, so every node of a chain has its own name.
    names = node_names(chain, graph)
    text = indirect_text(chain, graph, config.predicate_templates)
    # Naming the hidden object would turn the indirect query into a reworded direct one.
    if hidden_names_in_text(text, names[1:-1]):
        return REJECT_HIDDEN_NAME_IN_TEXT
    if text in seen_texts:
        return REJECT_DUPLICATE_TEXT
    return None
```

In `generate_pairs`: `problem = chain_problem(chain, graph, seen_texts, config)` and `seen_texts.add(indirect_text(chain, graph, config.predicate_templates))`.

(`indirect_text` gets its `templates` argument in Task 4; do Task 4 before running.)

---

### Task 4: Per-shape templates with plural-aware articles

**Files:**
- Modify: `src/mmxai/queries/generator.py` (template section, `make_pair`)
- Test: `tests/test_query_generator.py`

- [ ] **Step 1: Write/update tests.** Import `is_plural`, `first_mention`, `direct_text` and `SceneObject` (already imported). Replace `test_fixture_gives_the_expected_direct_and_indirect_text` assertions with:

```python
    assert direct["text"] == "A person is holding a frisbee, and a dog is looking at the frisbee."
    assert indirect["text"] == "A person is holding the thing that a dog is looking at."
```

Replace `test_path_and_shared_subject_chains_read_correctly` with:

```python
def test_path_chain_reads_correctly():
    graph = make_graph([(1, 1, "riding", 2), (2, 2, "on", 3)], {1: "man", 2: "horse", 3: "beach"})
    chain = enumerate_chains(1, graph.relations, 2, 2)[0]
    templates = CONFIG.predicate_templates

    assert direct_text(chain, graph, templates) == (
        "A man is riding a horse, and the horse is on a beach."
    )
    assert indirect_text(chain, graph, templates) == "A man is riding something that is on a beach."


def test_path_chain_reads_the_same_when_found_backwards():
    # The walk starts at the lowest id (beach), so node order is beach, horse, man.
    graph = make_graph([(1, 3, "on", 1), (2, 2, "riding", 3)], {1: "beach", 2: "man", 3: "horse"})
    chain = enumerate_chains(1, graph.relations, 2, 2)[0]

    assert indirect_text(chain, graph, CONFIG.predicate_templates) == (
        "A man is riding something that is on a beach."
    )


def test_plural_names_get_some_and_are():
    shoes = SceneObject(1, "shoes", "shoes", "shoe.n.01")
    glass = SceneObject(2, "glass", "glass", "glass.n.02")
    man = SceneObject(3, "man", "man", "")
    apple = SceneObject(4, "apple", "apple", "apple.n.01")

    assert is_plural(shoes)
    assert not is_plural(glass)
    assert not is_plural(man)
    assert first_mention(shoes) == "some shoes"
    assert first_mention(apple) == "an apple"
    assert first_mention(man) == "a man"


def test_plural_subject_takes_are():
    graph = make_graph([(1, 1, "on", 2), (2, 3, "wearing", 1)], {1: "shoes", 2: "floor", 3: "man"})
    graph.objects[1] = SceneObject(1, "shoes", "shoes", "shoe.n.01")
    chain = enumerate_chains(1, graph.relations, 2, 2)[0]

    assert direct_text(chain, graph, CONFIG.predicate_templates) == (
        "A man is wearing some shoes, and the shoes are on a floor."
    )
```

- [ ] **Step 2: Run, expect FAIL**

Run: `python -m uv run pytest tests/test_query_generator.py -q`

- [ ] **Step 3: Implement.** Replace everything from `def verb_phrase` through `def indirect_text` (including `with_article`, `describe_node`) with:

```python
def synset_lemma(synset: str) -> str:
    """"fire_hydrant.n.01" -> "fire hydrant"."""
    return synset.split(".")[0].replace("_", " ")


def is_plural(scene_object: SceneObject) -> bool:
    """True when the name is the synset lemma + "s"/"es" ("shoes" vs shoe.n.01).

    Deliberately narrow: irregular plurals ("men") read as singular and are left
    to the reviewer, instead of guessing English grammar from a word list.
    """
    if not scene_object.synset:
        return False
    last_word = scene_object.name.split()[-1]
    lemma_last_word = synset_lemma(scene_object.synset).split()[-1]
    return last_word in (lemma_last_word + "s", lemma_last_word + "es")


def first_mention(scene_object: SceneObject) -> str:
    """"a man", "an apple", "some shoes"."""
    if is_plural(scene_object):
        return f"some {scene_object.name}"
    if scene_object.name[0] in "aeiou":
        return f"an {scene_object.name}"
    return f"a {scene_object.name}"


def verb_for(phrase: str, plural_subject: bool) -> str:
    """Templates are written for one subject ("is holding"); a plural subject takes "are"."""
    if plural_subject and phrase.startswith("is "):
        return "are " + phrase[len("is ") :]
    return phrase


def as_sentence(text: str) -> str:
    return text[0].upper() + text[1:] + "."


def chain_roles(chain: Chain, graph: SceneGraph) -> tuple:
    """(first, first_relation, bridge, last, last_relation) in reading order.

    shared_object A -p-> X <-q- B: first = A, last = B.
    path          A -p-> X -q-> C: first = A (the subject pointing INTO the bridge),
                                   last = C, whatever direction the walk found it in.
    """
    bridge_id = chain.node_ids[1]
    bridge = graph.objects[bridge_id]
    if reasoning_type(chain) == "shared_object":
        first_relation, last_relation = chain.relations
        first = graph.objects[first_relation.subject_id]
        last = graph.objects[last_relation.subject_id]
        return first, first_relation, bridge, last, last_relation
    if chain.relations[0].object_id == bridge_id:
        first_relation, last_relation = chain.relations
    else:
        last_relation, first_relation = chain.relations
    first = graph.objects[first_relation.subject_id]
    last = graph.objects[last_relation.object_id]
    return first, first_relation, bridge, last, last_relation


def direct_text(chain: Chain, graph: SceneGraph, templates: dict[str, str]) -> str:
    """Name every object; the bridge is "a X" at first mention and "the X" afterwards."""
    first, first_relation, bridge, last, last_relation = chain_roles(chain, graph)
    first_verb = verb_for(templates[first_relation.predicate], is_plural(first))
    opening = f"{first_mention(first)} {first_verb} {first_mention(bridge)}"
    if reasoning_type(chain) == "shared_object":
        last_verb = verb_for(templates[last_relation.predicate], is_plural(last))
        closing = f"{first_mention(last)} {last_verb} the {bridge.name}"
    else:
        last_verb = verb_for(templates[last_relation.predicate], is_plural(bridge))
        closing = f"the {bridge.name} {last_verb} {first_mention(last)}"
    return as_sentence(f"{opening}, and {closing}")


def indirect_text(chain: Chain, graph: SceneGraph, templates: dict[str, str]) -> str:
    """Name the two end objects; the bridge is described only through its relations."""
    first, first_relation, bridge, last, last_relation = chain_roles(chain, graph)
    first_verb = verb_for(templates[first_relation.predicate], is_plural(first))
    if reasoning_type(chain) == "shared_object":
        last_verb = verb_for(templates[last_relation.predicate], is_plural(last))
        hidden = f"the thing that {first_mention(last)} {last_verb}"
    else:
        # "something" is singular, so the template is used as written.
        hidden = f"something that {templates[last_relation.predicate]} {first_mention(last)}"
    return as_sentence(f"{first_mention(first)} {first_verb} {hidden}")
```

Add `SceneObject` to the import from `mmxai.data.scene_graph`. Move `reasoning_type` **above** `chain_roles` (it is now used by it). In `make_pair`, pass `config.predicate_templates` to both `direct_text` and `indirect_text`.

- [ ] **Step 4: Run, expect PASS**

Run: `python -m uv run pytest tests/test_query_generator.py -q`

- [ ] **Step 5: Commit (only if agreed)**

```bash
git add src/mmxai/queries/generator.py tests/test_query_generator.py
git commit -m "feat(queries): per-shape templates, shape and action filters"
```

---

### Task 5: Match candidate answers by synset

**Files:**
- Modify: `src/mmxai/queries/validator.py` (`pattern_from_record`, `extend_match`, `graph_has_pattern`, module docstring)
- Modify: `src/mmxai/queries/generator.py` (`relation_evidence`, `chain_pattern`)
- Test: `tests/test_query_validator.py`, `tests/scene_graph_fixtures.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_query_validator.py`; import `Relation, SceneGraph, SceneObject` from `mmxai.data.scene_graph`)

```python
def synset_graph(image_id: int, rider: SceneObject) -> SceneGraph:
    horse = SceneObject(2, "horse", "horse", "horse.n.01")
    graph = SceneGraph(image_id, {1: rider, 2: horse})
    graph.relations.append(
        Relation(9, 1, rider.name, "riding", 2, "horse", "riding", rider.name, "horse")
    )
    return graph


def test_answers_match_by_synset_not_only_by_name():
    graphs = {
        1: synset_graph(1, SceneObject(1, "man", "man", "man.n.01")),
        2: synset_graph(2, SceneObject(1, "guy", "guy", "man.n.01")),
        3: synset_graph(3, SceneObject(1, "man", "man", "")),  # no synset: key "man"
    }

    assert images_matching_pattern(["man.n.01", "horse.n.01"], [("riding", True)], graphs) == [
        1,
        2,
    ]


def test_pattern_from_record_reads_synset_keys_from_evidence():
    record = {
        "qid": "q",
        "chain_object_ids": [1, 2],
        "relation_evidence": [
            {
                "relationship_id": 9,
                "subject_id": 1,
                "subject_name": "guy",
                "subject_synset": "man.n.01",
                "predicate": "riding",
                "object_id": 2,
                "object_name": "horse",
                "object_synset": "",
            }
        ],
    }

    assert pattern_from_record(record) == (["man.n.01", "horse"], [("riding", True)])
```

- [ ] **Step 2: Run, expect FAIL**

Run: `python -m uv run pytest tests/test_query_validator.py -q`

- [ ] **Step 3: Implement in `validator.py`.** Import `match_key` from `mmxai.data.scene_graph`. Add:

```python
def evidence_key(relation: dict, side: str) -> str:
    """Match key of one end ("subject"/"object") of a stored relation: synset, else name."""
    synset = relation.get(f"{side}_synset", "")
    if synset:
        return synset
    return relation[f"{side}_name"]
```

In `pattern_from_record`: rename `names` → `keys`; use `evidence_key(relation, "subject")` / `evidence_key(relation, "object")` instead of `relation["subject_name"]` / `relation["object_name"]` (also for the last node: `evidence_key(last, "object")` / `evidence_key(last, "subject")`). Docstring: "Return (node match keys n0..nk, steps)".

In `extend_match` (rename parameter `names` → `keys`):

```python
        if next_id in used_ids or match_key(graph.objects[next_id]) != keys[step_index + 1]:
            continue
```

In `graph_has_pattern` (rename `names` → `keys`):

```python
    present_keys = {match_key(scene_object) for scene_object in graph.objects.values()}
    if not set(keys) <= present_keys:
        return False
    for scene_object in graph.objects.values():
        if match_key(scene_object) != keys[0]:
            continue
```

Rename `names` → `keys` in `images_matching_pattern`. Update the module docstring "Relevance" paragraph: "objects with the same match key (VG WordNet synset, else normalised name)".

In `generator.py`, `relation_evidence`: needs the graph for synsets. Change to `relation_evidence(chain: Chain, graph: SceneGraph)` and add two keys per entry:

```python
                "subject_synset": graph.objects[relation.subject_id].synset,
                ...
                "object_synset": graph.objects[relation.object_id].synset,
```

Update its call in `make_pair` to `relation_evidence(chain, graph)`. In `chain_pattern`, return keys instead of names (import `match_key`):

```python
    keys = [match_key(graph.objects[node_id]) for node_id in chain.node_ids]
    return keys, steps
```

and rename `names, steps = chain_pattern(...)` → `keys, steps = ...` / `images_matching_pattern(keys, steps, graphs)` in `generate_pairs`.

In `tests/scene_graph_fixtures.py`, give `obj()` an optional synset so fixtures can exercise it:

```python
def obj(object_id: int, name: str, synset: str = "") -> dict:
    annotation = {"object_id": object_id, "names": [name], "x": 0, "y": 0, "w": 1, "h": 1}
    if synset:
        annotation["synsets"] = [synset]
    return annotation
```

- [ ] **Step 4: Run all query tests, expect PASS**

Run: `python -m uv run pytest tests/test_scene_graph.py tests/test_query_generator.py tests/test_query_validator.py tests/test_query_benchmark_eval.py tests/test_queries_committed.py -q`

If an older validator test that called `images_matching_pattern(names, ...)` with plain names fails, it is because the fixture has no synsets → keys equal names, so it should still pass; investigate before changing it.

- [ ] **Step 5: Commit (only if agreed)**

```bash
git add src/mmxai/queries tests/test_query_validator.py tests/scene_graph_fixtures.py
git commit -m "feat(queries): match candidate answers by Visual Genome synset"
```

---

### Task 6: Real configs (pilot and test share every rule)

**Files:**
- Modify: `configs/query_benchmark_vg.yaml`, `configs/query_benchmark_vg_pilot.yaml`
- Create: `tests/test_query_configs.py`

- [ ] **Step 1: Write the failing test** `tests/test_query_configs.py`:

```python
"""The val pilot must test exactly the rules later used on test (ADR-0044)."""

from mmxai.queries.generator import load_query_benchmark_config

# Only these may differ: which split, how many pairs, where files go.
ALLOWED_TO_DIFFER = {
    "split",
    "limit_pairs",
    "candidates_path",
    "manifest_path",
    "benchmark_path",
}


def test_pilot_and_test_configs_share_every_rule():
    test_config = load_query_benchmark_config("configs/query_benchmark_vg.yaml")
    pilot_config = load_query_benchmark_config("configs/query_benchmark_vg_pilot.yaml")

    for field in test_config.__dataclass_fields__:
        if field in ALLOWED_TO_DIFFER:
            continue
        assert getattr(test_config, field) == getattr(pilot_config, field), field
```

- [ ] **Step 2: Run, expect FAIL** (`unknown config keys: ['trivial_predicates']`)

- [ ] **Step 3: Edit both YAML files identically.** Change `author: generator-template-v1` → `author: generator-template-v2`. Replace the `trivial_predicates` comment + line with:

```yaml
# Decision (ADR-0044, pilot val 2026-11-06, Thuat Nguyen): whitelist of predicates with a
#   hand-written verb phrase. Chosen from predicate frequencies on TRAIN only.
#   Dropped on purpose: has/have, of, with, by, for, and, at, from, belonging to, between, along.
predicate_templates:
  wearing: is wearing
  holding: is holding
  riding: is riding
  carrying: is carrying
  eating: is eating
  playing: is playing
  watching: is watching
  looking at: is looking at
  pulling: is pulling
  sitting on: is sitting on
  sitting in: is sitting in
  standing on: is standing on
  standing in: is standing in
  walking on: is walking on
  laying on: is lying on
  hanging on: is hanging on
  hanging from: is hanging from
  covering: is covering
  attached to: is attached to
  painted on: is painted on
  "on": is on
  on top of: is on top of
  in: is in
  inside: is inside
  under: is under
  above: is above
  behind: is behind
  in front of: is in front of
  next to: is next to
  near: is near
  beside: is beside
# Variants mapped to one canonical predicate (after stripping "is/are" and a trailing article).
predicate_aliases:
  wears: wearing
  holds: holding
  rides: riding
  carries: carrying
  eats: eating
  plays: playing
  watches: watching
  looks at: looking at
  pulls: pulling
  sits on: sitting on
  stands on: standing on
  lying on: laying on
  hangs on: hanging on
  inside of: inside
# Decision (ADR-0044): every chain needs at least one of these actions.
action_predicates: [wearing, holding, riding, carrying, eating, playing, watching, looking at,
                    pulling, sitting on, sitting in, standing on, standing in, walking on,
                    laying on, hanging on, hanging from, covering]
```

Note: `"on"` must be quoted — bare `on` is parsed by PyYAML as boolean `True`.

- [ ] **Step 4: Run, expect PASS**

Run: `python -m uv run pytest tests/test_query_configs.py -q`

- [ ] **Step 5: Commit (only if agreed)**

```bash
git add configs/query_benchmark_vg.yaml configs/query_benchmark_vg_pilot.yaml tests/test_query_configs.py
git commit -m "feat(configs): predicate whitelist for the VG query benchmark"
```

---

### Task 7: Val pilot v2, full checks, docs

**Files:**
- Modify: `docs/query_benchmark.md`, `docs/ai-usage-log.md`

- [ ] **Step 1: Remove the old pilot output** (gitignored, generated by us; `write_jsonl` refuses to overwrite):

```bash
rm -r results/runs/query_pilot_val
```

- [ ] **Step 2: Run the val pilot** (never the test config in this plan)

Run: `python -m uv run python scripts/generate_queries.py --config configs/query_benchmark_vg_pilot.yaml`
Expected: JSON summary with `"num_pairs": 5` and `rejected_chains` containing `predicate_not_in_whitelist`, `shared_subject_chain`, `no_action_predicate`.

- [ ] **Step 3: Print the 5 pairs and check them by eye**

```bash
python -m uv run python -c "import json; [print(r['type'][:3], len(r['relevant']), '|', r['text'], '|', r['path_hint']) for r in map(json.loads, open('results/runs/query_pilot_val/visual_genome_candidates_val.jsonl', encoding='utf-8'))]"
```

Expected: grammatical sentences, no "a a"/"a the", no "has", indirect hides the bridge. Show them to the user. If a template reads badly, fix it in BOTH configs (val only — allowed) and repeat Steps 1–3. Never run the test config to judge templates.

- [ ] **Step 4: Full checks**

Run: `python -m uv run ruff check` → only the 2 pre-existing I001 in `tests/test_eval_metrics.py`, `tests/test_eval_relevance.py`.
Run: `python -m uv run ruff format --check` → all formatted.
Run: `python -m uv run pytest -q` → the same 10 pre-existing failures (`test_feature_pipeline.py`, `test_notebooks_match_src.py`), everything else passes.

- [ ] **Step 5: Docs.** In `docs/query_benchmark.md` update the rules/templates section to match ADR-0044 (whitelist, two shapes, action rule, synset matching, `some`/`are` for plurals) and replace the example with one pilot **val** pair, clearly labelled "ví dụ từ pilot val, không thuộc bộ test". Add a row to `docs/ai-usage-log.md` (skill `/log-ai-usage`), column "Đã tự kiểm tra/hiểu chưa" = `Chưa`.

- [ ] **Step 6: Commit (only if agreed)**

```bash
git add docs/query_benchmark.md docs/ai-usage-log.md docs/decisions/0044-bo-truy-van-danh-gia-tu-scene-graph-visual-genome.md
git commit -m "docs(queries): document query rules v2 and the val pilot"
```

---

## Out of scope (next steps for the user, not this plan)
- Generating 40 candidate pairs on test, human review by two members, building `queries/visual_genome_test.jsonl`.
- Evaluating CLIP on the benchmark (needs test image embeddings in `artifacts/`).
- Calling `leakage-auditor` / `experiment-reviewer` before the PR.
