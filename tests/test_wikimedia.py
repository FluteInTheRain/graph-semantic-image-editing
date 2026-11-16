"""Tests for the polite Wikimedia client. A fake opener replaces the network."""

import json
import urllib.error
from email.message import Message
from urllib.parse import parse_qs, urlparse

import pytest
from mmxai.data.wikimedia import COMMONS_API_URL, SPARQL_URL, WikimediaClient


class FakeResponse:
    """Minimal stand-in for the object urlopen returns."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class FakeOpener:
    """Callable that records requests and replays scripted outcomes in order."""

    def __init__(self, outcomes: list) -> None:
        self.outcomes = list(outcomes)
        self.requests = []
        self.timeouts = []

    def __call__(self, request, timeout):
        self.requests.append(request)
        self.timeouts.append(timeout)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeTime:
    """Fake clock and sleep: sleeping advances the clock."""

    def __init__(self) -> None:
        self.now = 100.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def json_response(payload: dict) -> FakeResponse:
    return FakeResponse(json.dumps(payload).encode("utf-8"))


def http_error(code: int, retry_after: str | None = None) -> urllib.error.HTTPError:
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError("http://example.org/x", code, "err", headers, fp=None)


def make_client(opener: FakeOpener, fake_time: FakeTime, max_attempts: int = 5):
    return WikimediaClient(
        user_agent="test-agent/1.0",
        interval_seconds=2.0,
        max_attempts=max_attempts,
        opener=opener,
        sleep=fake_time.sleep,
        clock=fake_time.clock,
    )


def test_sends_user_agent_and_waits_between_requests():
    opener = FakeOpener([json_response({"a": 1}), json_response({"b": 2})])
    fake_time = FakeTime()
    client = make_client(opener, fake_time)

    client.commons({"action": "query"})
    client.commons({"action": "query"})

    assert len(opener.requests) == 2
    for request in opener.requests:
        assert request.get_header("User-agent") == "test-agent/1.0"
    assert opener.timeouts == [120, 120]
    # First request does not sleep; the second waits the full interval.
    assert fake_time.sleeps == [2.0]
    url = opener.requests[0].full_url
    assert url.startswith(COMMONS_API_URL)
    assert parse_qs(urlparse(url).query)["format"] == ["json"]


def test_retries_429_using_retry_after():
    opener = FakeOpener([http_error(429, retry_after="7"), json_response({"ok": True})])
    fake_time = FakeTime()
    client = make_client(opener, fake_time)

    result = client.commons({"action": "query"})

    assert result == {"ok": True}
    assert 7.0 in fake_time.sleeps


def test_retries_server_errors_then_raises():
    opener = FakeOpener([http_error(504) for _ in range(5)])
    fake_time = FakeTime()
    client = make_client(opener, fake_time, max_attempts=5)

    with pytest.raises(RuntimeError, match="504"):
        client.sparql("SELECT * WHERE {}")

    assert len(opener.requests) == 5


def test_does_not_retry_client_errors():
    opener = FakeOpener([http_error(400), json_response({})])
    fake_time = FakeTime()
    client = make_client(opener, fake_time)

    with pytest.raises(urllib.error.HTTPError):
        client.commons({"action": "query"})

    assert len(opener.requests) == 1


def test_sparql_flattens_bindings():
    payload = {
        "results": {
            "bindings": [
                {"x": {"type": "uri", "value": "http://www.wikidata.org/entity/Q1"}},
            ]
        }
    }
    opener = FakeOpener([json_response(payload)])
    client = make_client(opener, FakeTime())

    rows = client.sparql("SELECT ?x WHERE {}")

    assert rows == [{"x": "http://www.wikidata.org/entity/Q1"}]
    request = opener.requests[0]
    assert request.full_url.startswith(SPARQL_URL)
    assert request.get_header("Accept") == "application/sparql-results+json"


def test_commons_raises_on_error_key():
    opener = FakeOpener([json_response({"error": {"code": "badvalue"}})])
    client = make_client(opener, FakeTime())

    with pytest.raises(RuntimeError, match="badvalue"):
        client.commons({"action": "query"})


def test_download_returns_raw_bytes():
    opener = FakeOpener([FakeResponse(b"\x89PNG")])
    client = make_client(opener, FakeTime())

    assert client.download("http://example.org/a.png") == b"\x89PNG"
