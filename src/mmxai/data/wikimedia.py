"""Polite HTTP client for Wikidata SPARQL and the Wikimedia Commons API.

This is the only code that touches the network for the MMKG collection.
Tests replace `opener` with a fake so they never go online.
"""

import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta

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
        max_retry_wait_seconds: float,
        opener=urllib.request.urlopen,
        sleep=time.sleep,
        clock=time.monotonic,
        log=print,
    ) -> None:
        self.user_agent = user_agent
        self.interval_seconds = interval_seconds
        self.max_attempts = max_attempts
        self.max_retry_wait_seconds = max_retry_wait_seconds
        self._log = log
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
                # Close the error response so its connection is not held open while we wait.
                error.close()
                # No point sleeping after the final attempt.
                if attempt < self.max_attempts - 1:
                    delay = self._retry_delay(error, attempt, url)
                    resume_at = datetime.now() + timedelta(seconds=delay)
                    host = urllib.parse.urlparse(url).hostname
                    self._log(
                        f"{host} {error.code}: waiting {delay:g} s, "
                        f"resuming at {resume_at:%H:%M:%S}"
                    )
                    self._sleep(delay)
        raise RuntimeError(f"HTTP {last_status} after {self.max_attempts} attempts for {url}")

    def _wait_for_interval(self) -> None:
        """Sleep so attempts start at least interval_seconds apart."""
        if self._last_attempt_start is not None:
            elapsed = self._clock() - self._last_attempt_start
            if elapsed < self.interval_seconds:
                self._sleep(self.interval_seconds - elapsed)
        self._last_attempt_start = self._clock()

    def _retry_delay(self, error: urllib.error.HTTPError, attempt: int, url: str) -> float:
        """Use Retry-After when it is a sane number, else exponential backoff.

        Wikimedia asks clients to honour Retry-After fully; retrying earlier can extend
        the block. A wait above max_retry_wait_seconds is refused with an error
        instead, so a run never stalls silently for a very long time.
        """
        delay = self.interval_seconds * 2**attempt
        retry_after = error.headers.get("Retry-After") if error.headers else None
        if retry_after is not None:
            try:
                requested = float(retry_after)
            except ValueError:
                requested = None
            # "nan", "inf" and negative values parse as floats but are not real waits.
            if requested is not None and math.isfinite(requested) and requested >= 0:
                delay = requested
        if delay > self.max_retry_wait_seconds:
            host = urllib.parse.urlparse(url).hostname
            raise RuntimeError(
                f"{host} asked to wait {delay:g} s, more than the limit of "
                f"{self.max_retry_wait_seconds:g} s; try again later"
            )
        return delay
