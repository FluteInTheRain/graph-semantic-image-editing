"""The JSONL query format of queries/README.md, plus the optional fields of ADR-0044."""

import json
from pathlib import Path

# Fields required by queries/README.md.
REQUIRED_FIELDS = ("qid", "text", "type", "dataset", "relevant")
QUERY_TYPES = ("direct", "indirect")

# Review stages (ADR-0044). Only "verified" queries may enter the final benchmark.
STATUS_CANDIDATE = "candidate"  # produced by the generator, nobody has looked at it
STATUS_REVIEWED = "reviewed"  # one member checked the images, a second check is missing
STATUS_VERIFIED = "verified"  # a second, different member confirmed the review
STATUS_REJECTED = "rejected"  # a reviewer threw it out
VERIFICATION_STATUSES = (STATUS_CANDIDATE, STATUS_REVIEWED, STATUS_VERIFIED, STATUS_REJECTED)


def check_record_schema(record: dict) -> list[str]:
    """Return the schema problems of one query record (empty list = valid)."""
    qid = record.get("qid", "<no qid>")
    problems = []
    for name in REQUIRED_FIELDS:
        if name not in record:
            problems.append(f"{qid}: missing field '{name}'")
    if problems:
        return problems

    if not isinstance(record["qid"], str) or not record["qid"]:
        problems.append(f"{qid}: qid must be a non-empty string")
    if not isinstance(record["text"], str) or not record["text"].strip():
        problems.append(f"{qid}: text must be a non-empty string")
    if record["type"] not in QUERY_TYPES:
        problems.append(f"{qid}: type must be one of {QUERY_TYPES}, got {record['type']!r}")
    relevant = record["relevant"]
    if not isinstance(relevant, list) or not relevant:
        problems.append(f"{qid}: relevant must be a non-empty list of image ids")
    else:
        for image_id in relevant:
            # queries/README.md and splits/README.md store ids as strings.
            if not isinstance(image_id, str) or not image_id.isdigit():
                problems.append(f"{qid}: relevant id {image_id!r} is not a digit string")
    status = record.get("verification_status")
    if status is not None and status not in VERIFICATION_STATUSES:
        problems.append(f"{qid}: unknown verification_status {status!r}")
    return problems


def load_jsonl(path: str | Path) -> list[dict]:
    """Read one JSON object per non-empty line."""
    records = []
    with open(path, encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON ({error})") from error
    return records


def write_jsonl(records: list[dict], path: str | Path) -> None:
    """Write one JSON object per line. Refuses to overwrite: query files are frozen once made."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists; delete it yourself to regenerate")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False))
            file.write("\n")
