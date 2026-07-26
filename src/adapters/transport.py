"""Shared HTTP transport for source adapters (plan 0001 V2.x).

Availability policy only, mirroring canon.py's caller split: pace requests to a
per-source courtesy interval, back off on 429/5xx within the same source, and
raise ``SourceUnavailable`` when retries run out — a source outage defers the
run, it never produces half-parsed data. Any other non-200 means the request
itself is wrong (``SourceError``), which is a bug to fix, not to retry.

httpx is imported lazily so the pure logic in each adapter stays testable with
a stdlib-only runner, mirroring routing.py / canon.py.
"""
from __future__ import annotations

import time
from typing import Callable

# Public sources ask API consumers to identify themselves; this UA does.
USER_AGENT = "vocaloid-radar/0.1 (+https://github.com/moongioh/vocaloid-radar)"
_ATTEMPTS = 3
_BACKOFF_BASE = 2.0  # seconds; doubles per retry


class SourceUnavailable(Exception):
    """The source kept answering 429/5xx. Defer to the next run — don't guess."""


class SourceError(Exception):
    """Non-retryable response (4xx / unparseable): the request is wrong."""


# raw_get(path, params) -> (http_status, parsed_json_or_None). Injected in tests.
RawGet = Callable[[str, dict], "tuple[int, object | None]"]


def _httpx_get(base_url: str) -> RawGet:
    import httpx  # lazy: keeps fixture tests stdlib-only

    client = httpx.Client(
        base_url=base_url,
        headers={"User-Agent": USER_AGENT},
        timeout=30.0,
        follow_redirects=True,
    )

    def raw_get(path: str, params: dict) -> "tuple[int, object | None]":
        r = client.get(path, params=params)
        try:
            return r.status_code, r.json()
        except ValueError:
            return r.status_code, None

    return raw_get


def make_getter(
    base_url: str,
    *,
    min_interval: float,
    raw_get: "RawGet | None" = None,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
):
    """Build a paced, retrying ``get(path, params) -> parsed_json``."""
    if raw_get is None:
        raw_get = _httpx_get(base_url)
    last = [float("-inf")]

    def get(path: str, params: dict):
        wait = min_interval - (monotonic() - last[0])
        if wait > 0:
            sleep(wait)
        for attempt in range(_ATTEMPTS):
            status, data = raw_get(path, params)
            if status == 429 or status >= 500:
                if attempt < _ATTEMPTS - 1:
                    sleep(_BACKOFF_BASE * 2**attempt)
                continue
            last[0] = monotonic()
            if status != 200 or data is None:
                raise SourceError(f"{path}: HTTP {status}")
            return data
        raise SourceUnavailable(
            f"{base_url}{path}: {_ATTEMPTS} attempts exhausted on 429/5xx"
        )

    return get
