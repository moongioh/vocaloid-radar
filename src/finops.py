"""FinOps rollup — the four §8 metrics + 승급쌍 일치율 (plan 0002 R4).

The plan splits the metrics across two sources by what each can answer cheaply:

  - DB rollup (``rollup_db``): cache-hit rate, pending backlog, escalation rate,
    demotion rate. All derivable from ``tags.canon_status`` alone, because the
    final-tier provenance is stamped there (§7) — 0 external calls, deterministic.
  - Phoenix rollup (``rollup_phoenix``): per-tier call count + token sum vs. the
    daily free quota. Only the trace store knows real token usage.
  - agreement rate (``agreement_rate``): the 승급쌍 일치율. An escalation naturally
    leaves a lite/flash output pair for the same tag (routing captures the lite
    proposal on ``TagResult.lite_canon``); if flash lands on the SAME canon lite
    already proposed, that escalation was wasted → the 0.80 threshold is too
    strict. Computed from a run's in-memory results — 0 extra calls (Evolve's
    real-data validation, done for free).

The threshold/cap tuning this feeds is plan 0003's campaign, gated on 4 weeks of
real trace; R4 only builds the measurement.
"""
from __future__ import annotations

from typing import Callable

from .routing import FLASH, LITE, Status, TagResult

# Free-tier daily request quotas (plan 0001 / memory gemini-free-tier-*).
# LITE = 1500 since the 2026-07-29 tier-1 swap to gw-gemma (plan 0003 E2); the old
# gw-lite figure was 1000. quota_pct is reported to the user, so a stale denominator
# here understates headroom rather than failing loudly.
DAILY_QUOTA = {LITE: 1500, FLASH: 250}


# ------------------------------------------------------------------ DB rollup

def rollup_db(conn) -> dict:
    """Cache + tier metrics from ``tags.canon_status`` (metrics §8-2, §8-3)."""
    with conn.cursor() as cur:
        cur.execute("SELECT canon_status, COUNT(*) FROM tags GROUP BY canon_status")
        counts = {r[0]: r[1] for r in cur.fetchall()}
    return summarize_status_counts(counts)


def summarize_status_counts(counts: dict) -> dict:
    """Pure: turn ``{canon_status: n}`` into the cache/tier metric block."""
    lite = counts.get(Status.CANON_LITE.value, 0)
    flash = counts.get(Status.CANON_FLASH.value, 0)
    pending = counts.get(Status.PENDING.value, 0)
    demoted = counts.get(Status.DEMOTED.value, 0)
    total = lite + flash + pending + demoted
    accepted = lite + flash
    return {
        "total": total,
        "tiers": {LITE: lite, FLASH: flash, "demoted": demoted, "pending": pending},
        # §8-2: share of tags already resolved to an immutable canon (cache hits).
        "cache_hit_rate": _ratio(accepted, total),
        "pending_backlog": pending,
        # §8-3: of the accepted tags, how many needed the flash escalation...
        "escalation_rate": _ratio(flash, accepted),
        # ...and how often the quality floor was hit (flash also failed).
        "demotion_rate": _ratio(demoted, accepted + demoted),
    }


# ------------------------------------------------------------- Phoenix rollup

# fetch() -> list of {"model": str, "tokens": int | None} — one entry per LLM span.
SpanFetch = Callable[[], "list[dict]"]


def rollup_phoenix(
    base_url: "str | None" = None,
    api_key: "str | None" = None,
    *,
    project: str = "gateway",  # LiteLLM's Phoenix project (verified live 2026-07-18)
    fetch: "SpanFetch | None" = None,
) -> dict:
    """Per-tier call count + token sum vs. daily quota (metric §8-1)."""
    spans = (fetch or _phoenix_span_fetch(base_url or "", api_key or "", project))()
    return summarize_spans(spans)


def summarize_spans(spans: "list[dict]") -> dict:
    """Pure: aggregate LLM spans into per-tier calls/tokens and quota usage.

    Also surfaces the paid-model guardrail (acceptance criterion §3): any span on
    a model that is not gw-lite/gw-flash is flagged, since no path should ever
    call a paid tier on its own.
    """
    per_tier: dict[str, dict] = {}
    off_tier: dict[str, int] = {}
    for sp in spans:
        model = sp.get("model")
        if model in DAILY_QUOTA:
            t = per_tier.setdefault(model, {"calls": 0, "tokens": 0})
            t["calls"] += 1
            t["tokens"] += sp.get("tokens") or 0
        elif model is not None:
            off_tier[model] = off_tier.get(model, 0) + 1
    for model, t in per_tier.items():
        t["quota_pct"] = round(100.0 * t["calls"] / DAILY_QUOTA[model], 1)
    return {
        "per_tier": per_tier,
        # Non-empty = a paid/unknown model was hit — an invariant breach to inspect.
        "off_tier_calls": off_tier,
    }


def _phoenix_span_fetch(base_url: str, api_key: str, project: str) -> SpanFetch:
    """Live adapter: pull LLM spans from Phoenix's GraphQL, project by tier + tokens.

    Kept behind the injectable ``fetch`` seam so summarize_spans stays testable
    without a live Phoenix. Attribute shape verified against live Phoenix
    (2026-07-18): the tier the FinOps quota is keyed on is the *gateway alias*
    (``llm.response.model`` = gw-lite/gw-flash), not ``llm.model_name`` (the raw
    provider model, e.g. gemini-2.5-flash-lite). Only spanKind=='llm'
    (litellm_request) spans are counted — the sibling raw_gen_ai_request span
    would double-count. ``attributes`` arrives as a JSON string.
    """
    import json

    import httpx  # lazy: keeps fixture tests stdlib-only

    query = """
    query ($project: String!) {
      projects(filter: {col: name, value: $project}) {
        edges { node { spans(first: 1000) { edges { node {
          spanKind
          attributes
        } } } } }
      }
    }
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    def fetch() -> "list[dict]":
        r = httpx.post(
            f"{base_url}/graphql",
            json={"query": query, "variables": {"project": project}},
            headers=headers,
            timeout=30.0,
        )
        r.raise_for_status()
        data = r.json()["data"]["projects"]["edges"]
        spans: list[dict] = []
        for proj in data:
            for edge in proj["node"]["spans"]["edges"]:
                node = edge["node"]
                if node.get("spanKind", "").upper() != "LLM":
                    continue
                attrs = node.get("attributes")
                if isinstance(attrs, str):
                    attrs = json.loads(attrs)
                spans.append({
                    # gateway alias (gw-lite/gw-flash) is the tier axis, not model_name
                    "model": _dig(attrs, "llm", "response", "model")
                    or _dig(attrs, "llm", "model_name"),
                    "tokens": _dig(attrs, "llm", "token_count", "total"),
                })
        return spans

    return fetch


def _dig(d, *keys):
    """Read nested keys tolerating both flat 'a.b.c' and nested {'a': {'b': ...}}."""
    if not isinstance(d, dict):
        return None
    flat = d.get(".".join(keys))
    if flat is not None:
        return flat
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


# ------------------------------------------------------ 승급쌍 일치율 (agreement)

def agreement_rate(results: "list[TagResult]") -> dict:
    """승급쌍 일치율 (§8-4): of escalated items flash accepted, how many did flash
    map to the same canon lite already proposed? A high rate means the threshold
    over-escalated (wasted flash quota) — tune it down. 0 extra calls.
    """
    pairs = [
        r for r in results
        if r.status is Status.CANON_FLASH and r.lite_canon is not None
    ]
    agree = sum(1 for r in pairs if r.canon_name == r.lite_canon)
    return {
        "escalation_pairs": len(pairs),
        "agreed": agree,
        # None when there were no comparable pairs this run.
        "agreement_rate": _ratio(agree, len(pairs)) if pairs else None,
    }


def _ratio(num: int, den: int) -> float:
    return round(num / den, 3) if den else 0.0
