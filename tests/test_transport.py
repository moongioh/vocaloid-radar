"""Fixture tests for the shared adapter transport (plan 0001 V2.x).

No network: raw_get / sleep / monotonic are injected, so pacing and retry
policy are exercised deterministically.
"""
import pytest

from src.adapters.transport import SourceError, SourceUnavailable, make_getter


def make_env(responses):
    """responses = list of (status, data) popped per call. Records sleeps."""
    calls = {"n": 0}
    sleeps = []
    clock = {"t": 0.0}

    def raw_get(path, params):
        calls["n"] += 1
        return responses.pop(0)

    def sleep(s):
        sleeps.append(s)
        clock["t"] += s

    def monotonic():
        return clock["t"]

    return raw_get, sleep, monotonic, calls, sleeps


def test_success_returns_parsed_json():
    raw_get, sleep, monotonic, calls, _ = make_env([(200, {"ok": 1})])
    get = make_getter("http://x", min_interval=1.0, raw_get=raw_get,
                      sleep=sleep, monotonic=monotonic)
    assert get("/p", {}) == {"ok": 1}
    assert calls["n"] == 1


def test_429_backs_off_then_succeeds():
    raw_get, sleep, monotonic, calls, sleeps = make_env([(429, None), (200, {"ok": 1})])
    get = make_getter("http://x", min_interval=0.0, raw_get=raw_get,
                      sleep=sleep, monotonic=monotonic)
    assert get("/p", {}) == {"ok": 1}
    assert calls["n"] == 2
    assert 2.0 in sleeps  # backoff base


def test_exhausted_5xx_raises_unavailable():
    raw_get, sleep, monotonic, _, _ = make_env([(500, None)] * 3)
    get = make_getter("http://x", min_interval=0.0, raw_get=raw_get,
                      sleep=sleep, monotonic=monotonic)
    with pytest.raises(SourceUnavailable):
        get("/p", {})


def test_4xx_is_a_request_bug_not_retried():
    raw_get, sleep, monotonic, calls, _ = make_env([(400, {"err": "bad"})])
    get = make_getter("http://x", min_interval=0.0, raw_get=raw_get,
                      sleep=sleep, monotonic=monotonic)
    with pytest.raises(SourceError):
        get("/p", {})
    assert calls["n"] == 1


def test_pacing_sleeps_between_calls():
    raw_get, sleep, monotonic, _, sleeps = make_env([(200, {}), (200, {})])
    get = make_getter("http://x", min_interval=5.0, raw_get=raw_get,
                      sleep=sleep, monotonic=monotonic)
    get("/p", {})
    get("/p", {})
    # second call must wait out the remainder of the 5s interval
    assert any(0 < s <= 5.0 for s in sleeps)
