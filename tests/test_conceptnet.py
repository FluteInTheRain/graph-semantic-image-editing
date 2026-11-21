"""ConceptNet parsing and filtering (ADR-0057). Synthetic lines only."""

from conceptnet_fixtures import assertion_line, write_assertions_gz
from mmxai.data.conceptnet import (
    ConceptEdge,
    concept_term,
    is_malformed,
    load_conceptnet_prepare_config,
    load_edges_tsv,
    parse_assertion_line,
    prepare_conceptnet,
    write_edges_tsv,
)

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


def test_prepare_config_loads_the_repo_file():
    config = load_conceptnet_prepare_config("configs/conceptnet_en.yaml")

    assert config.relations == ["IsA", "UsedFor", "AtLocation", "HasA", "PartOf", "CapableOf"]
    assert config.min_weight == 1.0


def broken_json_line() -> str:
    columns = assertion_line("IsA", "/c/en/tennis", "/c/en/sport").split("	")
    columns[4] = "{not json"
    return "	".join(columns)


def text_weight_line() -> str:
    columns = assertion_line("IsA", "/c/en/tennis", "/c/en/sport").split("	")
    columns[4] = '{"weight": "abc"}'
    return "	".join(columns)


def test_parse_does_not_crash_on_bad_json_or_a_non_numeric_weight():
    assert parse_assertion_line(broken_json_line(), RELATIONS, 1.0) is None
    assert parse_assertion_line(text_weight_line(), RELATIONS, 1.0) is None
    assert parse_assertion_line("only	two", RELATIONS, 1.0) is None


def test_info_that_is_not_an_object_is_malformed():
    columns = assertion_line("IsA", "/c/en/tennis", "/c/en/sport").split("	")
    columns[4] = "[1, 2]"
    line = "	".join(columns)

    assert parse_assertion_line(line, RELATIONS, 1.0) is None
    assert is_malformed(line, RELATIONS)


def test_is_malformed_tells_broken_lines_from_filtered_ones():
    good_but_french = assertion_line("IsA", "/c/fr/tennis", "/c/fr/sport")
    other_relation = assertion_line("Synonym", "/c/en/racket", "/c/en/racquet")

    assert is_malformed(broken_json_line(), RELATIONS)
    assert is_malformed(text_weight_line(), RELATIONS)
    assert is_malformed("only	two", RELATIONS)
    assert not is_malformed(good_but_french, RELATIONS)
    assert not is_malformed(other_relation, RELATIONS)


def test_prepare_counts_malformed_lines(tmp_path):
    source = tmp_path / "assertions.csv.gz"
    write_assertions_gz(
        source,
        [
            assertion_line("IsA", "/c/en/tennis", "/c/en/sport"),
            broken_json_line(),
            text_weight_line(),
        ],
    )

    summary = prepare_conceptnet(source, tmp_path / "edges.tsv", RELATIONS, min_weight=1.0)

    assert summary["malformed_lines"] == 2
    assert summary["edges_kept"] == 1
