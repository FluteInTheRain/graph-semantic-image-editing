"""Read ConceptNet 5.7 assertions and keep the small English part the KG queries need.

ConceptNet is a public commonsense knowledge graph: each edge is
(start, relation, end, weight), e.g. racket -UsedFor-> tennis. The full assertions
file is large, so it is filtered ONCE (scripts/prepare_conceptnet.py) into a TSV
with only English-English edges of a few relations (ADR-0057). The original edge
URI is kept so every query can cite the exact edge it was built from.
"""

import csv
import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from mmxai.utils.yaml_config import load_yaml_dataclass

TSV_COLUMNS = ["uri", "rel", "start", "end", "weight"]


@dataclass(frozen=True)
class ConceptNetPrepareConfig:
    source_path: str
    output_path: str
    relations: list[str]
    min_weight: float


def load_conceptnet_prepare_config(path: str | Path) -> ConceptNetPrepareConfig:
    return load_yaml_dataclass(path, ConceptNetPrepareConfig)


@dataclass(frozen=True)
class ConceptEdge:
    """One kept edge; start/end are plain terms ("tennis racket"), rel has no "/r/"."""

    uri: str
    rel: str
    start: str
    end: str
    weight: float


def concept_term(node: str) -> str | None:
    """Turn "/c/en/tennis_racket/n" into "tennis racket"; None for non-English nodes."""
    parts = node.split("/")
    # ["", "c", "en", "tennis_racket", "n", ...]
    if len(parts) < 4 or parts[1] != "c" or parts[2] != "en":
        return None
    return parts[3].replace("_", " ")


def read_weight(info: str) -> float | None:
    """The numeric weight in the info column, or None if the column is malformed."""
    try:
        data = json.loads(info)
        if not isinstance(data, dict):
            return None
        return float(data.get("weight", 0.0))
    except (ValueError, TypeError):
        # ValueError covers json.JSONDecodeError and float("abc").
        return None


def parse_assertion_line(line: str, relations: list[str], min_weight: float) -> ConceptEdge | None:
    """Turn one assertions line into a ConceptEdge, or None if the edge is not kept.

    Never raises on a bad line: malformed and filtered-out lines both give None.
    is_malformed tells the two apart, so the prepare step can count the broken ones.
    """
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
    weight = read_weight(info)
    if weight is None or weight < min_weight:
        return None
    return ConceptEdge(uri, relation, start, end, weight)


def is_malformed(line: str, relations: list[str]) -> bool:
    """True if the line is broken: not 5 columns, or an unreadable info column.

    The info JSON is only checked for wanted relations: parsing it for all ~34M lines
    would be slow, and a broken line of an unwanted relation does not matter.
    """
    columns = line.rstrip("\n").split("\t")
    if len(columns) != 5:
        return True
    if columns[1].removeprefix("/r/") not in relations:
        return False
    return read_weight(columns[4]) is None


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
            f"{path} not found. Download conceptnet-assertions-5.7.0.csv.gz into "
            "data/conceptnet/ and run scripts/prepare_conceptnet.py "
            "(see data_cards/conceptnet.md)."
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
            "https://github.com/commonsense/conceptnet5/wiki/Downloads "
            "(see data_cards/conceptnet.md)."
        )
    edges = []
    lines_read = 0
    malformed_lines = 0
    with gzip.open(source_path, "rt", encoding="utf-8") as file:
        for line in file:
            lines_read += 1
            edge = parse_assertion_line(line, relations, min_weight)
            if edge is not None:
                edges.append(edge)
            elif is_malformed(line, relations):
                malformed_lines += 1
    write_edges_tsv(edges, output_path)
    return {
        "source": str(source_path),
        "output": str(output_path),
        "lines_read": lines_read,
        "malformed_lines": malformed_lines,
        "edges_kept": len(edges),
        "sha256": file_sha256(output_path),
    }
