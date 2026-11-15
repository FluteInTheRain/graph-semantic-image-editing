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
    start_part = start.replace(" ", "_")
    end_part = end.replace(" ", "_")
    uri = f"/a/[/r/{relation}/,/c/en/{start_part}/,/c/en/{end_part}/]"
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
    # Ridden things: only the horse is found in a barn and can carry a rider.
    edge("IsA", "horse", "animal"),
    edge("AtLocation", "horse", "barn"),
    edge("CapableOf", "horse", "carry rider"),
    edge("IsA", "elephant", "animal"),
    edge("AtLocation", "elephant", "zoo"),
]


def write_assertions_gz(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as file:
        for line in lines:
            file.write(line + "\n")
