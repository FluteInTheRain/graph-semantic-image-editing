"""Tests for the polite Wikimedia client. A fake opener replaces the network."""

import http.client
import json
import re
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


def make_client(opener: FakeOpener, fake_time: FakeTime, max_attempts: int = 5, log=None):
    if log is None:
        log = [].append
    return WikimediaClient(
        user_agent="test-agent/1.0",
        interval_seconds=2.0,
        max_attempts=max_attempts,
        max_retry_wait_seconds=1800.0,
        opener=opener,
        sleep=fake_time.sleep,
        clock=fake_time.clock,
        log=log,
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


def test_retry_after_is_honoured_and_logged():
    opener = FakeOpener([http_error(429, retry_after="1000"), json_response({"ok": True})])
    fake_time = FakeTime()
    messages = []
    client = make_client(opener, fake_time, log=messages.append)

    client.commons({"action": "query"})

    assert fake_time.sleeps == [1000.0]
    assert len(messages) == 1
    assert messages[0].startswith("commons.wikimedia.org 429: waiting 1000 s")
    assert re.search(r"resuming at \d\d:\d\d:\d\d$", messages[0])


def test_retry_after_above_limit_raises():
    opener = FakeOpener([http_error(429, retry_after="3600"), json_response({"ok": True})])
    fake_time = FakeTime()
    client = make_client(opener, fake_time)

    with pytest.raises(RuntimeError, match="commons.wikimedia.org.*3600"):
        client.commons({"action": "query"})

    assert fake_time.sleeps == []
    assert len(opener.requests) == 1


@pytest.mark.parametrize("retry_after", ["-5", "nan", "inf"])
def test_invalid_retry_after_falls_back_to_backoff(retry_after):
    opener = FakeOpener([http_error(429, retry_after=retry_after), json_response({})])
    fake_time = FakeTime()
    client = make_client(opener, fake_time)

    client.commons({"action": "query"})

    # First backoff step = interval_seconds * 2**0.
    assert fake_time.sleeps == [2.0]


def test_retried_error_is_closed():
    error = http_error(503)
    opener = FakeOpener([error, json_response({})])
    client = make_client(opener, FakeTime())

    client.commons({"action": "query"})

    assert error.fp.closed


def test_sparql_goes_to_the_configured_endpoint():
    opener = FakeOpener([json_response({"results": {"bindings": []}})])
    client = WikimediaClient(
        user_agent="test-agent/1.0",
        interval_seconds=2.0,
        max_attempts=1,
        max_retry_wait_seconds=1800.0,
        sparql_url="https://example.org/api/wikidata",
        opener=opener,
    )

    client.sparql("SELECT ?x WHERE {}")

    assert opener.requests[0].full_url.startswith("https://example.org/api/wikidata?query=")


def test_sparql_endpoint_stats_reads_cmd_stats():
    opener = FakeOpener([json_response({"name-index": "wikidata"})])
    client = WikimediaClient(
        user_agent="test-agent/1.0",
        interval_seconds=2.0,
        max_attempts=1,
        max_retry_wait_seconds=1800.0,
        sparql_url="https://example.org/api/wikidata",
        opener=opener,
    )

    assert client.sparql_endpoint_stats() == {"name-index": "wikidata"}
    assert opener.requests[0].full_url == "https://example.org/api/wikidata?cmd=stats"


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.URLError("connection refused"),
        TimeoutError("timed out"),
        ConnectionResetError("reset by peer"),
        http.client.IncompleteRead(b"partial"),
    ],
)
def test_transient_network_errors_are_retried_with_backoff(error):
    opener = FakeOpener([error, json_response({"ok": True})])
    fake_time = FakeTime()
    messages = []
    client = make_client(opener, fake_time, log=messages.append)

    assert client.commons({"action": "query"}) == {"ok": True}

    # First backoff step = interval_seconds * 2**0.
    assert fake_time.sleeps == [2.0]
    assert len(messages) == 1
    assert messages[0].startswith("commons.wikimedia.org network error")


def test_network_error_on_last_attempt_names_host_and_path_only():
    opener = FakeOpener([TimeoutError("timed out") for _ in range(3)])
    client = make_client(opener, FakeTime(), max_attempts=3)

    with pytest.raises(RuntimeError) as raised:
        client.commons({"action": "query", "titles": "File:Secret.jpg"})

    message = str(raised.value)
    assert "commons.wikimedia.org/w/api.php" in message
    assert "after 3 attempts" in message
    assert "Secret" not in message
    assert isinstance(raised.value.__cause__, TimeoutError)


def test_http_error_on_last_attempt_names_host_and_path_only():
    opener = FakeOpener([http_error(503) for _ in range(2)])
    client = make_client(opener, FakeTime(), max_attempts=2)

    with pytest.raises(RuntimeError) as raised:
        client.sparql("SELECT ?secret WHERE {}")

    message = str(raised.value)
    assert "HTTP 503" in message
    assert "query.wikidata.org/sparql" in message
    assert "secret" not in message


@pytest.mark.parametrize("code", [501, 505, 520])
def test_other_5xx_codes_are_not_retried(code):
    opener = FakeOpener([http_error(code), json_response({})])
    client = make_client(opener, FakeTime())

    with pytest.raises(urllib.error.HTTPError):
        client.commons({"action": "query"})

    assert len(opener.requests) == 1


@pytest.mark.parametrize("code", [429, 500, 502, 503, 504])
def test_listed_codes_are_retried(code):
    opener = FakeOpener([http_error(code), json_response({"ok": True})])
    client = make_client(opener, FakeTime())

    assert client.commons({"action": "query"}) == {"ok": True}
