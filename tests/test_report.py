"""Fixture tests for the report_gen node (plan 0002 R3). No live calls, no DB.

Covers the pure prompt grounding and the flash caller's availability policy
(pacing, same-tier backoff, GatewayUnavailable). report_gen has no quality gate
— a narrative has no id-echo contract — so escalation is intentionally absent.
"""
from src.canon import GatewayUnavailable
from src.report import (
    build_report_prompt,
    generate_report,
    make_report_caller,
)
from src.routing import FLASH

EVIDENCE = {
    "week": "2026-07-13",
    "top_songs": [{"song_id": 1, "title": "テスト曲", "view_velocity": 3.2, "deriv_velocity": 1.1}],
    "tag_deltas": [{"canon": "初音ミク", "share_pct": 12.3, "delta_pp": 4.1}],
    "clusters": [{"cluster_id": 0, "titles": ["A", "B"]}],
    "watchlist": [{"song_id": 9, "title": "新曲"}],
}


# ---------------------------------------------------------------- pure: prompt

def test_build_report_prompt_grounding():
    p = build_report_prompt(EVIDENCE)
    assert "テスト曲" in p and "初音ミク" in p       # evidence is in the prompt
    assert "선행지표" in p                            # framing: signals, not prediction
    assert "지어내지" in p                            # no fabrication beyond evidence
    assert "한국어" in p                              # human-facing report → Korean


# --------------------------------------------------------------- the caller

def make_clock():
    state = {"t": 0.0, "sleeps": []}

    def monotonic():
        return state["t"]

    def sleep(s):
        state["sleeps"].append(s)
        state["t"] += s

    return state, monotonic, sleep


def content_response(text):
    return (200, {"choices": [{"message": {"content": text}}]})


def test_report_caller_happy_path():
    body_log = []

    def post(body):
        body_log.append(body)
        return content_response("이번 주 핵심 신호: ...")

    state, monotonic, sleep = make_clock()
    caller = make_report_caller(post=post, sleep=sleep, monotonic=monotonic)
    out = caller("prompt")
    assert out.startswith("이번 주")
    assert body_log[0]["model"] == FLASH  # fixed on flash, always


def test_report_caller_429_retries_flash():
    codes = [429, 429, 200]
    models = []

    def post(body):
        models.append(body["model"])
        c = codes.pop(0)
        return (c, None) if c != 200 else content_response("ok")

    state, monotonic, sleep = make_clock()
    caller = make_report_caller(post=post, sleep=sleep, monotonic=monotonic)
    assert caller("p") == "ok"
    assert models == [FLASH, FLASH, FLASH]  # never leaves the tier
    assert len(state["sleeps"]) >= 2        # backoff happened


def test_report_caller_exhausted_raises():
    def post(body):
        return 503, None

    state, monotonic, sleep = make_clock()
    caller = make_report_caller(post=post, sleep=sleep, monotonic=monotonic)
    try:
        caller("p")
        assert False, "expected GatewayUnavailable"
    except GatewayUnavailable:
        pass


def test_report_caller_bad_request_is_none():
    def post(body):
        return 400, None

    state, monotonic, sleep = make_clock()
    caller = make_report_caller(post=post, sleep=sleep, monotonic=monotonic)
    assert caller("p") is None


def test_report_caller_paces_to_flash_rpm():
    def post(body):
        return content_response("ok")

    state, monotonic, sleep = make_clock()
    caller = make_report_caller(post=post, sleep=sleep, monotonic=monotonic)
    caller("p")
    before = len(state["sleeps"])
    caller("p")
    paced = state["sleeps"][before:]
    assert paced and paced[0] >= 6.0  # 60 / 10 rpm


# --------------------------------------------------------------- orchestration

def test_generate_report_outage_returns_none():
    def caller(prompt):
        raise GatewayUnavailable("down")

    assert generate_report(EVIDENCE, caller) is None
