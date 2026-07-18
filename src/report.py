"""report_gen node — the weekly narrative on gw-flash, no escalation (plan 0002 R3).

Tier map (plan 0002 §1): report_gen is FIXED on ``gw-flash`` — narrative judgment
is already the top free tier, so there is nowhere to escalate to. The flash daily
quota this node depends on is protected from tag_canon regressions by the run-level
escalation cap in routing.call_tiered (§3), not by anything here.

Layering mirrors canon.py:
  - pure: build the grounded prompt, no I/O.
  - transport: a caller that owns AVAILABILITY only (flash RPM pacing, same-tier
    429/5xx backoff, GatewayUnavailable on exhaustion). No quality gate — a
    narrative has no id-echo contract to check.
  - orchestration: one call; an outage defers the week's report (returns None),
    it is never partially fabricated.
  - db (thin): assemble the evidence snapshot from the batch tables, upsert the
    narrative + evidence JSON into weekly_reports.

httpx/psycopg stay lazy so the pure logic is testable with a stdlib-only runner.
"""
from __future__ import annotations

import json
import time
from typing import Callable

from .canon import GatewayUnavailable  # same availability contract as the tag_canon caller
from .routing import FLASH

# Free-tier RPM for flash (plan 0001; re-check before relying).
_FLASH_RPM = 10
_ATTEMPTS = 3
_BACKOFF_BASE = 2.0  # seconds; doubles per retry


# ---------------------------------------------------------------- pure: prompt

def build_report_prompt(evidence: dict) -> str:
    """Ground the narrative strictly on the supplied evidence (plan 0001).

    The framing is leading-indicator detection, NOT hit prediction — small samples
    plus exogenous events (remixes, external buzz) make prediction unverifiable, so
    the model is told to describe signals, not forecast outcomes, and to invent
    nothing beyond the evidence block.
    """
    block = json.dumps(evidence, ensure_ascii=False, indent=2)
    return (
        "당신은 보컬로이드 주간 트렌드 레이더의 애널리스트입니다.\n"
        "아래 evidence(이번 주 배치 집계)만 근거로 한국어 주간 리포트를 씁니다.\n\n"
        f"evidence:\n{block}\n\n"
        "규칙:\n"
        "1. evidence에 없는 수치·곡·태그를 지어내지 마십시오. 모든 서술은 위 데이터에 대응해야 합니다.\n"
        "2. 이것은 '히트 예측'이 아니라 '선행지표 관측'입니다. 무엇이 뜰지 단정하지 말고, "
        "어떤 신호(조회 가속·2차창작 가속·태그 점유 변화·군집)가 관측됐는지 기술하십시오.\n"
        "3. 구성: ① 이번 주 핵심 신호 3~5줄 요약 ② top_songs 가속 곡 근거와 함께 ③ "
        "tag_deltas로 본 장르·태그 흐름 ④ clusters(동시 태그 군집)가 시사하는 미시 트렌드 "
        "⑤ watchlist(신곡 워치, 아직 속도 판정 불가) 짚기.\n"
        "4. 데이터가 비어있는 섹션은 '해당 신호 없음'으로 명시하고 넘어가십시오.\n"
    )


# ------------------------------------------------------- transport: the caller

# post(body) -> (http_status, response_json_or_None). Injected in tests.
Post = Callable[[dict], "tuple[int, dict | None]"]


def _httpx_post(base_url: str, api_key: str) -> Post:
    import httpx  # lazy: keeps fixture tests stdlib-only

    client = httpx.Client(
        base_url=base_url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=120.0,  # a full narrative is longer than a tag batch
    )

    def post(body: dict) -> "tuple[int, dict | None]":
        r = client.post("/v1/chat/completions", json=body)
        try:
            return r.status_code, r.json()
        except ValueError:
            return r.status_code, None

    return post


def make_report_caller(
    base_url: "str | None" = None,
    api_key: "str | None" = None,
    *,
    post: "Post | None" = None,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
):
    """Build the flash caller for report_gen. Availability policy only (§6):
    pace to flash RPM, back off on 429/5xx on the SAME tier, raise
    ``GatewayUnavailable`` when retries run out. A non-429 4xx (our request is
    broken) returns None. There is no escalation — flash is the top free tier.
    """
    if post is None:
        post = _httpx_post(base_url or "", api_key or "")
    last_call = {"t": float("-inf")}

    def caller(prompt: str) -> "str | None":
        wait = 60.0 / _FLASH_RPM - (monotonic() - last_call["t"])
        if wait > 0:
            sleep(wait)
        body = {
            "model": FLASH,
            "messages": [{"role": "user", "content": prompt}],
        }
        for attempt in range(_ATTEMPTS):
            status, data = post(body)
            if status == 429 or status >= 500:
                if attempt < _ATTEMPTS - 1:
                    sleep(_BACKOFF_BASE * 2**attempt)
                continue
            last_call["t"] = monotonic()
            if status != 200 or data is None:
                return None
            try:
                return data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                return None
        raise GatewayUnavailable(f"{FLASH}: {_ATTEMPTS} attempts exhausted on 429/5xx")

    return caller


# ------------------------------------------------------- orchestration

def generate_report(evidence: dict, caller) -> "str | None":
    """One flash call. Returns the narrative, or None when the gateway is down
    (the week's report is deferred, never partially fabricated)."""
    try:
        return caller(build_report_prompt(evidence))
    except GatewayUnavailable:
        return None


# ----------------------------------------------------------------- db (thin)

def load_evidence(conn, week, top_n: int = 15) -> dict:
    """Assemble the week's evidence snapshot from the batch tables.

    top_songs / clusters / watchlist come straight from trend_scores (materialized
    by the trend_score + cluster_detect nodes). tag_deltas is canon-level: this
    week's canonical-tag share vs. last week's, computed here from song_tags →
    tags(canon) → songs(publish_date). Everything is read-only.
    """
    with conn.cursor() as cur:
        # top velocity songs this week (cold-start excluded — no velocity yet)
        cur.execute(
            "SELECT ts.song_id, s.title, ts.view_velocity, ts.deriv_velocity, ts.cluster_id "
            "FROM trend_scores ts JOIN songs s ON s.id = ts.song_id "
            "WHERE ts.week = %s AND NOT ts.is_coldstart AND ts.view_velocity IS NOT NULL "
            "ORDER BY ts.view_velocity DESC LIMIT %s",
            (week, top_n),
        )
        top_songs = [
            {"song_id": r[0], "title": r[1], "view_velocity": _round(r[2]),
             "deriv_velocity": _round(r[3]), "cluster_id": r[4]}
            for r in cur.fetchall()
        ]

        # clusters: group this week's top songs by cluster_id
        cur.execute(
            "SELECT ts.cluster_id, array_agg(s.title ORDER BY ts.view_velocity DESC) "
            "FROM trend_scores ts JOIN songs s ON s.id = ts.song_id "
            "WHERE ts.week = %s AND ts.cluster_id IS NOT NULL "
            "GROUP BY ts.cluster_id ORDER BY ts.cluster_id",
            (week,),
        )
        clusters = [{"cluster_id": r[0], "titles": r[1]} for r in cur.fetchall()]

        # watchlist: cold-start new songs (too little history to score)
        cur.execute(
            "SELECT ts.song_id, s.title FROM trend_scores ts JOIN songs s ON s.id = ts.song_id "
            "WHERE ts.week = %s AND ts.is_coldstart ORDER BY s.publish_date DESC LIMIT %s",
            (week, top_n),
        )
        watchlist = [{"song_id": r[0], "title": r[1]} for r in cur.fetchall()]

        tag_deltas = _load_tag_deltas(cur, week)

    return {
        "week": str(week),
        "top_songs": top_songs,
        "tag_deltas": tag_deltas,
        "clusters": clusters,
        "watchlist": watchlist,
    }


def _load_tag_deltas(cur, week, top_n: int = 15) -> list:
    """Canonical-tag share this week vs. the prior week, in percentage points.

    Share = a canon tag's % of songs published that week. Only confirmed canon
    (canon_lite/canon_flash) counts — raw/pending tags never enter the trend axis.
    """
    from .analysis.trend import tag_share, tag_share_delta

    def counts(w):
        cur.execute(
            "SELECT t.canon_name, COUNT(DISTINCT s.id) FROM songs s "
            "JOIN song_tags st ON st.song_id = s.id "
            "JOIN tags t ON t.id = st.tag_id "
            "WHERE date_trunc('week', s.publish_date) = %s "
            "AND t.canon_status IN ('canon_lite', 'canon_flash') AND t.canon_name IS NOT NULL "
            "GROUP BY t.canon_name",
            (w,),
        )
        rows = dict(cur.fetchall())
        cur.execute(
            "SELECT COUNT(*) FROM songs WHERE date_trunc('week', publish_date) = %s", (w,)
        )
        return rows, cur.fetchone()[0]

    this_counts, this_total = counts(week)
    from datetime import timedelta
    prev_counts, prev_total = counts(week - timedelta(days=7))

    deltas = []
    for canon, n in this_counts.items():
        this_pct = tag_share(n, this_total)
        prev_pct = tag_share(prev_counts.get(canon, 0), prev_total)
        deltas.append({
            "canon": canon,
            "share_pct": round(this_pct, 1),
            "delta_pp": round(tag_share_delta(prev_pct, this_pct), 1),
        })
    deltas.sort(key=lambda d: d["delta_pp"], reverse=True)
    return deltas[:top_n]


def _round(x):
    return round(x, 2) if x is not None else None


def persist_report(conn, week, narrative: str, evidence: dict, model: str = FLASH) -> None:
    """Upsert the week's narrative + evidence snapshot (idempotent on re-run)."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO weekly_reports (week, narrative, evidence, model) "
            "VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (week) DO UPDATE SET "
            "narrative = EXCLUDED.narrative, evidence = EXCLUDED.evidence, "
            "model = EXCLUDED.model, generated_at = now()",
            (week, narrative, json.dumps(evidence, ensure_ascii=False), model),
        )
    conn.commit()


def run_report(
    conn,
    week,
    *,
    base_url: "str | None" = None,
    api_key: "str | None" = None,
    post: "Post | None" = None,
    top_n: int = 15,
) -> "dict | None":
    """The node entry point: assemble evidence, generate, persist. Returns a small
    summary, or None if the gateway was down (nothing persisted)."""
    evidence = load_evidence(conn, week, top_n)
    caller = make_report_caller(base_url, api_key, post=post)
    narrative = generate_report(evidence, caller)
    if narrative is None:
        return None
    persist_report(conn, week, narrative, evidence)
    return {"week": str(week), "chars": len(narrative),
            "top_songs": len(evidence["top_songs"]),
            "tag_deltas": len(evidence["tag_deltas"])}
