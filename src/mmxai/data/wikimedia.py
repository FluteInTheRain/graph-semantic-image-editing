"""Polite HTTP client for Wikidata SPARQL and the Wikimedia Commons API.

This is the only code that touches the network for the MMKG collection.
Tests replace `opener` with a fake so they never go online.
"""

import json
import time
import urllib.error
import urllib.parse
import urllib.request

SPARQL_URL = "https://query.wikidata.org/sparql"
COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"

# SPARQL queries can be slow, so the timeout is generous.
REQUEST_TIMEOUT_SECONDS = 120


class WikimediaClient:
    """Sends requests with a User-Agent, a minimum gap, and retry on 429/5xx."""

    def __init__(
        self,
        user_agent: str,
        interval_seconds: float,
        max_attempts: int,
        opener=urllib.request.urlopen,
        sleep=time.sleep,
        clock=time.monotonic,
    ) -> None:
        self.user_agent = user_agent
        self.interval_seconds = interval_seconds
        self.max_attempts = max_attempts
        self._opener = opener
        self._sleep = sleep
        self._clock = clock
        # None until the first attempt, so the first request never sleeps.
        self._last_attempt_start = None

    def sparql(self, query: str) -> list[dict[str, str]]:
        """Run a SPARQL query and return each binding as {variable: value}."""
        query_string = urllib.parse.urlencode({"query": query})
        url = SPARQL_URL + "?" + query_string
        body = self._get(url, accept="application/sparql-results+json")
        payload = json.loads(body)
        rows = []
        for binding in payload["results"]["bindings"]:
            row = {}
            for variable, cell in binding.items():
                row[variable] = cell["value"]
            rows.append(row)
        return rows

    def commons(self, params: dict[str, str]) -> dict:
        """Call the Commons API (format=json is added) and return the JSON."""
        all_params = dict(params)
        all_params["format"] = "json"
        url = COMMONS_API_URL + "?" + urllib.parse.urlencode(all_params)
        payload = json.loads(self._get(url))
        if "error" in payload:
            raise RuntimeError(f"Commons API error: {payload['error']}")
        return payload

    def download(self, url: str) -> bytes:
        """Download a file and return its raw bytes."""
        return self._get(url)

    def _get(self, url: str, accept: str | None = None) -> bytes:
        """GET a URL. Interval and retry rules live here and nowhere else."""
        last_status = None
        for attempt in range(self.max_attempts):
            self._wait_for_interval()
            request = urllib.request.Request(url)
            request.add_header("User-Agent", self.user_agent)
            if accept is not None:
                request.add_header("Accept", accept)
            try:
                with self._opener(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                    return response.read()
            except urllib.error.HTTPError as error:
                is_retryable = error.code == 429 or 500 <= error.code <= 599
                if not is_retryable:
                    raise
                last_status = error.code
                # No point sleeping after the final attempt.
                if attempt < self.max_attempts - 1:
                    self._sleep(self._retry_delay(error, attempt))
        raise RuntimeError(f"HTTP {last_status} after {self.max_attempts} attempts for {url}")

    def _wait_for_interval(self) -> None:
        """Sleep so attempts start at least interval_seconds apart."""
        if self._last_attempt_start is not None:
            elapsed = self._clock() - self._last_attempt_start
            if elapsed < self.interval_seconds:
                self._sleep(self.interval_seconds - elapsed)
        self._last_attempt_start = self._clock()

    def _retry_delay(self, error: urllib.error.HTTPError, attempt: int) -> float:
        """Use Retry-After when it is a number, else exponential backoff."""
        retry_after = error.headers.get("Retry-After") if error.headers else None
        if retry_after is not None:
            try:
                return float(retry_after)
            except ValueError:
                pass
        return self.interval_seconds * 2**attempt
