"""Fixture tests for the FinOps rollup (plan 0002 R4). No live calls, no DB.

Exercises the pure aggregation of all three sources: status-count rollup,
Phoenix span aggregation (incl. the paid-model guardrail), and the 승급쌍 일치율.
"""
from src.finops import (
    agreement_rate,
    summarize_spans,
    summarize_status_counts,
)
from src.routing import FLASH, LITE, Status, TagResult


# ------------------------------------------------------------ status rollup

def test_summarize_status_counts():
    counts = {"canon_lite": 70, "canon_flash": 20, "pending": 8, "demoted": 2}
    r = summarize_status_counts(counts)
    assert r["total"] == 100
    assert r["cache_hit_rate"] == 0.9          # 90 accepted / 100
    assert r["pending_backlog"] == 8
    assert r["escalation_rate"] == round(20 / 90, 3)   # flash share of accepted
    assert r["demotion_rate"] == round(2 / 92, 3)      # demoted / (accepted + demoted)
    assert r["tiers"][FLASH] == 20


def test_summarize_status_counts_empty():
    r = summarize_status_counts({})
    assert r["total"] == 0 and r["cache_hit_rate"] == 0.0


# ------------------------------------------------------------- Phoenix spans

def test_summarize_spans_per_tier():
    spans = [
        {"model": LITE, "tokens": 100},
        {"model": LITE, "tokens": 150},
        {"model": FLASH, "tokens": 500},
    ]
    r = summarize_spans(spans)
    assert r["per_tier"][LITE] == {"calls": 2, "tokens": 250, "quota_pct": round(200 / 1000, 1)}
    assert r["per_tier"][FLASH]["calls"] == 1 and r["per_tier"][FLASH]["tokens"] == 500
    assert r["off_tier_calls"] == {}


def test_summarize_spans_flags_paid_model():
    # Any span outside gw-lite/gw-flash is an invariant breach (§3: no paid calls).
    spans = [{"model": LITE, "tokens": 10}, {"model": "gemini-2.5-pro", "tokens": 900}]
    r = summarize_spans(spans)
    assert r["off_tier_calls"] == {"gemini-2.5-pro": 1}


def test_summarize_spans_tolerates_missing_tokens():
    r = summarize_spans([{"model": FLASH, "tokens": None}, {"model": FLASH}])
    assert r["per_tier"][FLASH] == {"calls": 2, "tokens": 0, "quota_pct": round(200 / 250, 1)}


# ------------------------------------------------------------- agreement rate

def test_agreement_rate_flags_wasted_escalation():
    results = [
        # flash agreed with lite's proposal -> the escalation was wasted
        TagResult(1, Status.CANON_FLASH, canon_name="ロック", lite_canon="ロック"),
        # flash overturned lite -> the escalation earned its keep
        TagResult(2, Status.CANON_FLASH, canon_name="初音ミク", lite_canon="ミク違い"),
        # lite pass — no pair, ignored
        TagResult(3, Status.CANON_LITE, canon_name="バラード"),
        # escalated with no parseable lite proposal (G1/G2) — no pair, ignored
        TagResult(4, Status.CANON_FLASH, canon_name="EDM", lite_canon=None),
    ]
    r = agreement_rate(results)
    assert r["escalation_pairs"] == 2
    assert r["agreed"] == 1
    assert r["agreement_rate"] == 0.5


def test_agreement_rate_no_pairs():
    r = agreement_rate([TagResult(1, Status.CANON_LITE, canon_name="ロック")])
    assert r["escalation_pairs"] == 0 and r["agreement_rate"] is None
