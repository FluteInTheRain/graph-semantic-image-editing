"""Tests for the MMKG raw snapshot collector. A fake client answers; no network."""

import json
import os
import subprocess
import sys
import unicodedata
from datetime import date
from pathlib import Path

import pytest

from mmxai.data import mmkg_collect
from mmxai.data.mmkg_collect import (
    MmkgCollectConfig,
    RawSnapshot,
    collect_snapshot,
    normalize_file_name,
    strip_html,
    write_snapshot,
)
from mmxai.data.mmkg_csv import read_rows

REPO_ROOT = Path(__file__).resolve().parents[1]
ENTITY = "http://www.wikidata.org/entity/"
PROP = "http://www.wikidata.org/prop/direct/"
TODAY = date(2026, 11, 16)

CONFIG = MmkgCollectConfig(
    output_root="mmkg/raw",
    country_qid="Q881",
    designation_property="P1435",
    user_agent="test-agent",
    request_interval_seconds=0.0,
    max_attempts=1,
    max_file_bytes=1048576,
)


def domain_row(qid, vi="", en="", category="", image=""):
    row = {"item": ENTITY + qid}
    if vi:
        row["name_vi"] = vi
    if en:
        row["name_en"] = en
    if category:
        row["category"] = category
    if image:
        row["image"] = "http://commons.wikimedia.org/wiki/Special:FilePath/" + image
    return row


def triple_row(head, prop, tail):
    return {"h": ENTITY + head, "p": PROP + prop, "t": ENTITY + tail}


def info_page(name, license_name="CC BY-SA 4.0"):
    return {
        "title": "File:" + name,
        "imageinfo": [
            {
                "mime": "image/jpeg",
                "width": 800,
                "height": 600,
                "sha1": "sha-" + name,
                "user": "uploader1",
                "extmetadata": {
                    "LicenseShortName": {"value": license_name},
                    "Artist": {"value": '<a href="x">Ann&amp;Bob</a>'},
                },
            }
        ],
    }


class FakeClient:
    """Answers `sparql` and `commons` from canned data."""

    def __init__(
        self,
        domain,
        triples,
        labels=(),
        concepts=(),
        categories=None,
        pages=None,
        normalized=None,
        redirects=None,
    ):
        self.domain = list(domain)
        self.triples = list(triples)
        self.labels = list(labels)
        self.concepts = list(concepts)
        self.categories = categories or {}
        self.pages = pages or {}
        # Like Commons: titles are always normalized, redirects followed only on request.
        self.normalized = normalized or {}
        self.redirects = redirects or {}
        self.user_agent = "test-agent"

    def sparql(self, query):
        if "?image" in query:
            return self.domain
        if "VALUES" in query and "rdfs:label" in query:
            return self.labels
        if "VALUES" in query:
            return self.concepts
        return self.triples

    def commons(self, params):
        if params.get("list") == "categorymembers":
            pages = self.categories[params["cmtitle"]]
            index = 1 if "cmcontinue" in params else 0
            members = [{"title": "File:" + name} for name in pages[index]]
            answer = {"query": {"categorymembers": members}}
            if index + 1 < len(pages):
                answer["continue"] = {"cmcontinue": "next", "continue": "-||"}
            return answer
        names = [title[len("File:") :] for title in params["titles"].split("|")]
        normalized = []
        redirects = []
        result = {}
        for number, name in enumerate(names):
            if name in self.normalized:
                normalized.append({"from": "File:" + name, "to": "File:" + self.normalized[name]})
                name = self.normalized[name]
            if params.get("redirects") and name in self.redirects:
                redirects.append({"from": "File:" + name, "to": "File:" + self.redirects[name]})
                name = self.redirects[name]
            if name in self.pages:
                result[str(number)] = self.pages[name]
            else:
                result["-" + str(number + 1)] = {"title": "File:" + name, "missing": ""}
        query = {"pages": result}
        if normalized:
            query["normalized"] = normalized
        if redirects:
            query["redirects"] = redirects
        return {"query": query}


def test_file_names_from_p18_and_category_match():
    url = "http://commons.wikimedia.org/wiki/Special:FilePath/Ngoc%20Lu_drum.jpg"
    assert normalize_file_name(url) == "Ngoc Lu drum.jpg"
    assert normalize_file_name("File:Ngoc Lu drum.jpg") == "Ngoc Lu drum.jpg"
    assert normalize_file_name("Ngoc_Lu drum.jpg") == "Ngoc Lu drum.jpg"


def test_strip_html_unescapes():
    assert strip_html('<a href="x">Nguyễn&nbsp;Văn&amp;A</a>\n') == "Nguyễn Văn&A"


def test_collect_keeps_targets_and_concept_triples():
    client = FakeClient(
        domain=[domain_row("Q1", vi="Một"), domain_row("Q2", en="Two")],
        triples=[triple_row("Q1", "P186", "Q9"), triple_row("Q2", "P186", "Q9")],
        labels=[{"item": ENTITY + "Q9", "name_vi": "Gỗ", "name_en": "Wood"}],
        concepts=[triple_row("Q9", "P279", "Q8"), triple_row("Q9", "P361", "Q1")],
    )
    snapshot = collect_snapshot(client, CONFIG, TODAY)
    by_id = {row["entity_id"]: row for row in snapshot.entities}
    assert set(by_id) == {"Q1", "Q2", "Q9"}
    assert by_id["Q1"]["in_domain"] == "1"
    assert by_id["Q1"]["name_en"] == ""
    assert by_id["Q9"]["in_domain"] == "0"
    assert by_id["Q9"]["name_vi"] == "Gỗ"
    triples = {(r["head_id"], r["property"], r["tail_id"]) for r in snapshot.triples}
    assert ("Q9", "P361", "Q1") in triples
    assert ("Q9", "P279", "Q8") not in triples
    assert ("Q1", "P186", "Q9") in triples


def test_category_files_follow_continuation():
    client = FakeClient(
        domain=[domain_row("Q1", category="Temple")],
        triples=[],
        categories={"Category:Temple": [["a.jpg"], ["b.jpg"]]},
        pages={"a.jpg": info_page("a.jpg"), "b.jpg": info_page("b.jpg")},
    )
    snapshot = collect_snapshot(client, CONFIG, TODAY)
    names = sorted(row["file_name"] for row in snapshot.image_candidates)
    assert names == ["a.jpg", "b.jpg"]
    assert {row["origin"] for row in snapshot.image_candidates} == {"category"}


def test_shared_file_gets_one_row_per_entity():
    client = FakeClient(
        domain=[
            domain_row("Q1", image="shared.jpg"),
            domain_row("Q2", category="Cat"),
        ],
        triples=[],
        categories={"Category:Cat": [["shared.jpg"]]},
        pages={"shared.jpg": info_page("shared.jpg")},
    )
    snapshot = collect_snapshot(client, CONFIG, TODAY)
    rows = sorted((r["entity_id"], r["origin"]) for r in snapshot.image_candidates)
    assert rows == [("Q1", "p18"), ("Q2", "category")]
    assert len(snapshot.image_credits) == 1
    credit = snapshot.image_credits[0]
    assert credit["artist"] == "Ann&Bob"
    assert credit["uploader"] == "uploader1"
    assert snapshot.image_candidates[0]["license"] == "CC BY-SA 4.0"
    assert snapshot.image_candidates[0]["width"] == "800"


def test_p18_wins_over_category_for_same_entity():
    client = FakeClient(
        domain=[domain_row("Q1", category="Cat", image="x.jpg")],
        triples=[],
        categories={"Category:Cat": [["x.jpg"]]},
        pages={"x.jpg": info_page("x.jpg")},
    )
    snapshot = collect_snapshot(client, CONFIG, TODAY)
    assert [(r["entity_id"], r["origin"]) for r in snapshot.image_candidates] == [("Q1", "p18")]
    assert snapshot.entities[0]["p18_files"] == "x.jpg"
    assert snapshot.entities[0]["commons_category"] == "Cat"


def test_deleted_file_is_skipped_and_reported():
    client = FakeClient(
        domain=[domain_row("Q1", image="gone.jpg"), domain_row("Q2", image="here.jpg")],
        triples=[],
        pages={"here.jpg": info_page("here.jpg")},
    )
    snapshot = collect_snapshot(client, CONFIG, TODAY)
    assert [r["file_name"] for r in snapshot.image_candidates] == ["here.jpg"]
    assert snapshot.meta["missing_files"] == ["gone.jpg"]


def test_normalize_file_name_applies_nfc():
    decomposed = unicodedata.normalize("NFD", "Trống.jpg")
    assert decomposed != "Trống.jpg"
    assert normalize_file_name(decomposed) == "Trống.jpg"


def test_redirected_p18_file_is_recorded_under_target_name():
    client = FakeClient(
        domain=[domain_row("Q1", image="Old.jpg")],
        triples=[],
        redirects={"Old.jpg": "New.jpg"},
        pages={"New.jpg": info_page("New.jpg")},
    )
    snapshot = collect_snapshot(client, CONFIG, TODAY)
    rows = [(r["file_name"], r["entity_id"], r["origin"]) for r in snapshot.image_candidates]
    assert rows == [("New.jpg", "Q1", "p18")]
    assert [credit["file_name"] for credit in snapshot.image_credits] == ["New.jpg"]
    assert snapshot.meta["missing_files"] == []


def test_normalized_title_is_not_reported_missing():
    client = FakeClient(
        domain=[domain_row("Q1", image="ngoc%20lu.jpg")],
        triples=[],
        normalized={"ngoc lu.jpg": "Ngoc lu.jpg"},
        pages={"Ngoc lu.jpg": info_page("Ngoc lu.jpg")},
    )
    snapshot = collect_snapshot(client, CONFIG, TODAY)
    assert [r["file_name"] for r in snapshot.image_candidates] == ["Ngoc lu.jpg"]
    assert snapshot.meta["missing_files"] == []


def test_redirect_into_category_file_keeps_one_p18_row():
    client = FakeClient(
        domain=[domain_row("Q1", category="Cat", image="Old.jpg")],
        triples=[],
        categories={"Category:Cat": [["New.jpg"]]},
        redirects={"Old.jpg": "New.jpg"},
        pages={"New.jpg": info_page("New.jpg")},
    )
    snapshot = collect_snapshot(client, CONFIG, TODAY)
    rows = [(r["file_name"], r["entity_id"], r["origin"]) for r in snapshot.image_candidates]
    assert rows == [("New.jpg", "Q1", "p18")]
    assert len(snapshot.image_credits) == 1


def test_cli_refuses_when_todays_snapshot_exists(tmp_path):
    output_root = tmp_path / "raw"
    (output_root / date.today().strftime("%Y%m%d")).mkdir(parents=True)
    config_file = tmp_path / "collect.yaml"
    config_file.write_text(
        f"output_root: {output_root}\n"
        "country_qid: Q881\n"
        "designation_property: P1435\n"
        "user_agent: test-agent\n"
        "request_interval_seconds: 0\n"
        "max_attempts: 1\n"
        "max_file_bytes: 1048576\n",
        encoding="utf-8",
    )
    # A dead proxy guarantees this test never reaches Wikimedia, even if the check breaks.
    env = dict(os.environ, HTTPS_PROXY="http://127.0.0.1:9", HTTP_PROXY="http://127.0.0.1:9")

    command = [sys.executable, str(REPO_ROOT / "scripts" / "collect_mmkg.py")]
    command += ["--config", str(config_file)]
    result = subprocess.run(
        command, capture_output=True, text=True, cwd=REPO_ROOT, env=env, check=False
    )

    assert result.returncode == 1
    assert "already exists" in result.stderr


def small_snapshot() -> RawSnapshot:
    return RawSnapshot(
        entities=[
            {
                "entity_id": "Q1",
                "name_vi": "Một",
                "name_en": "",
                "in_domain": "1",
                "commons_category": "",
                "p18_files": "",
            }
        ],
        triples=[{"head_id": "Q1", "property": "P186", "tail_id": "Q9"}],
        image_candidates=[],
        image_credits=[],
        meta={"collected_on": "2026-11-16", "row_counts": {}},
    )


def test_write_snapshot_writes_tables_and_meta(tmp_path):
    folder = write_snapshot(small_snapshot(), tmp_path, 1048576)
    assert folder == tmp_path / "20261116"
    assert read_rows(folder, "entities_raw")[0]["name_vi"] == "Một"
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    assert meta["collected_on"] == "2026-11-16"
    assert not (tmp_path / "20261116.partial").exists()


def test_write_snapshot_is_all_or_nothing(tmp_path, monkeypatch):
    real_write_rows = mmkg_collect.write_rows
    calls = []

    def failing_write_rows(*args, **kwargs):
        calls.append(1)
        if len(calls) == 3:
            raise OSError("disk full")
        return real_write_rows(*args, **kwargs)

    monkeypatch.setattr(mmkg_collect, "write_rows", failing_write_rows)
    with pytest.raises(OSError):
        write_snapshot(small_snapshot(), tmp_path, 1048576)
    assert not (tmp_path / "20261116").exists()
    assert not (tmp_path / "20261116.partial").exists()


def test_write_snapshot_refuses_existing_folder(tmp_path):
    write_snapshot(small_snapshot(), tmp_path, 1048576)
    with pytest.raises(FileExistsError):
        write_snapshot(small_snapshot(), tmp_path, 1048576)


def test_write_snapshot_removes_stale_partial(tmp_path):
    stale = Path(tmp_path) / "20261116.partial"
    stale.mkdir()
    (stale / "old.txt").write_text("x")
    folder = write_snapshot(small_snapshot(), tmp_path, 1048576)
    assert not (folder / "old.txt").exists()
