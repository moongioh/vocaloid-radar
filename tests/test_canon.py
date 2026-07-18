"""Fixture tests for the tag_canon node (plan 0002 R2). No live calls, no DB.

Covers the three R2 layers: prompt/parse (pure), the availability policy inside
the gateway caller (pacing, same-tier backoff, GatewayUnavailable), and the
batch orchestration (vocab growth, run-level cap, outage deferral).
"""
from src.canon import (
    GatewayUnavailable,
    build_prompt,
    make_gateway_caller,
    normalize_classify,
    parse_response,
)
from src.routing import FLASH, LITE, RoutingConfig, Status, TagInput

VOCAB = {"ロック", "初音ミク"}


# ---------------------------------------------------------------- prompt/parse

def test_build_prompt_contract():
    items = [TagInput(1, "みくみく"), TagInput(2, "ろっく")]
    p = build_prompt(items, VOCAB)
    for term in VOCAB:
        assert term in p          # closed-loop vocabulary is in the prompt
    assert '"id": 1' in p and '"id": 2' in p
    assert "is_new" in p          # new-term proposals must be explicit
    assert "exactly once" in p    # id-echo contract (G1)


def test_parse_response_variants():
    ok = [{"id": 1, "canonical": "ロック", "confidence": 0.9}]
    assert parse_response('{"items": [{"id": 1, "canonical": "ロック", "confidence": 0.9}]}') == ok
    assert parse_response('[{"id": 1, "canonical": "ロック", "confidence": 0.9}]') == ok
    fenced = '```json\n{"items": [{"id": 1, "canonical": "ロック", "confidence": 0.9}]}\n```'
    assert parse_response(fenced) == ok
    assert parse_response("broken {") is None
    assert parse_response('{"items": "not-a-list"}') is None
    assert parse_response('[1, 2]') is None  # items must be dicts


# ------------------------------------------------------------------ the caller

def content_response(items):
    import json
    return (200, {"choices": [{"message": {"content": json.dumps({"items": items})}}]})


def make_clock():
    """Deterministic clock: sleeping advances time, nothing else does."""
    state = {"t": 0.0, "sleeps": []}

    def monotonic():
        return state["t"]

    def sleep(s):
        state["sleeps"].append(s)
        state["t"] += s

    return state, monotonic, sleep


def test_caller_happy_path():
    body_log = []

    def post(body):
        body_log.append(body)
        return content_response([{"id": 1, "canonical": "ロック", "confidence": 0.9}])

    state, monotonic, sleep = make_clock()
    caller = make_gateway_caller(vocab=set(VOCAB), post=post, sleep=sleep, monotonic=monotonic)
    out = caller(LITE, [TagInput(1, "ろっく")])
    assert out == [{"id": 1, "canonical": "ロック", "confidence": 0.9}]
    assert body_log[0]["model"] == LITE
    assert "ロック" in body_log[0]["messages"][0]["content"]


def test_caller_429_retries_same_tier():
    # 429 twice then 200 — retried on the SAME model with backoff, never escalated.
    codes = [429, 429, 200]
    body_log = []

    def post(body):
        body_log.append(body)
        c = codes.pop(0)
        if c != 200:
            return c, None
        return content_response([{"id": 1, "canonical": "ロック", "confidence": 0.9}])

    state, monotonic, sleep = make_clock()
    caller = make_gateway_caller(vocab=set(VOCAB), post=post, sleep=sleep, monotonic=monotonic)
    out = caller(LITE, [TagInput(1, "ろっく")])
    assert out is not None
    assert [b["model"] for b in body_log] == [LITE, LITE, LITE]  # same tier throughout
    assert len(state["sleeps"]) >= 2  # backoff happened


def test_caller_exhausted_raises_unavailable():
    def post(body):
        return 429, None

    state, monotonic, sleep = make_clock()
    caller = make_gateway_caller(vocab=set(VOCAB), post=post, sleep=sleep, monotonic=monotonic)
    try:
        caller(LITE, [TagInput(1, "ろっく")])
        assert False, "expected GatewayUnavailable"
    except GatewayUnavailable:
        pass


def test_caller_bad_request_is_malformed_not_outage():
    # A non-429 4xx = our request is broken -> None (G1 batch failure), no raise.
    def post(body):
        return 400, None

    state, monotonic, sleep = make_clock()
    caller = make_gateway_caller(vocab=set(VOCAB), post=post, sleep=sleep, monotonic=monotonic)
    assert caller(LITE, [TagInput(1, "ろっく")]) is None


def test_caller_paces_to_rpm():
    # Two back-to-back lite calls: the second must wait out the 60/15 = 4s interval.
    def post(body):
        return content_response([{"id": 1, "canonical": "ロック", "confidence": 0.9}])

    state, monotonic, sleep = make_clock()
    caller = make_gateway_caller(vocab=set(VOCAB), post=post, sleep=sleep, monotonic=monotonic)
    caller(LITE, [TagInput(1, "ろっく")])
    before = len(state["sleeps"])
    caller(LITE, [TagInput(1, "ろっく")])
    paced = state["sleeps"][before:]
    assert paced and paced[0] >= 4.0


# ------------------------------------------------------------- orchestration

def test_vocab_grows_across_batches():
    # Batch 1: new term proposed -> flash confirms -> registered. Batch 2: lite
    # maps into the *grown* vocabulary and passes membership on its own.
    vocab = {"ロック"}

    def caller(model, items):
        it = items[0]
        if it.id == 1:
            conf = 0.95 if model == FLASH else 0.90
            return [{"id": 1, "canonical": "新ジャンル", "is_new": True, "confidence": conf}]
        return [{"id": 2, "canonical": "新ジャンル", "confidence": 0.90}]

    items = [TagInput(1, "新語"), TagInput(2, "新語の変形")]
    out = normalize_classify(items, vocab, caller, batch_size=1)
    assert out[0].status == Status.CANON_FLASH and out[0].registers_vocab
    assert out[1].status == Status.CANON_LITE  # membership passed via grown vocab
    assert "新ジャンル" in vocab


def test_run_level_cap_spans_batches():
    # cap=1 across the whole run: first failing item uses it, the rest defer.
    def caller(model, items):
        it = items[0]
        if model == LITE:
            return [{"id": it.id, "canonical": "ロック", "confidence": 0.5}]
        return [{"id": it.id, "canonical": "ロック", "confidence": 0.9}]

    items = [TagInput(i, f"t{i}") for i in (1, 2, 3)]
    out = normalize_classify(
        items, set(VOCAB), caller, batch_size=1,
        config=RoutingConfig(escalation_cap=1),
    )
    assert [r.status for r in out] == [Status.CANON_FLASH, Status.PENDING, Status.PENDING]


def test_outage_defers_everything():
    # Gateway down mid-run: current batch AND all remaining items -> pending.
    def caller(model, items):
        raise GatewayUnavailable("down")

    items = [TagInput(i, f"t{i}") for i in (1, 2, 3)]
    out = normalize_classify(items, set(VOCAB), caller, batch_size=2)
    assert [r.status for r in out] == [Status.PENDING] * 3
