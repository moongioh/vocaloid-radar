"""Weekly batch — LangGraph assembly (plan 0001 V3.4).

    fetch_validate → normalize_classify → trend_score → report_gen → publish

Two conditional branches, both availability-shaped:
  - fetch_validate halts the run when there is nothing to score (no songs / no
    metrics in the week) — an empty report would be fabrication.
  - report_gen returning None (gateway outage, plan 0002 §6) skips publish: the
    week's report is deferred, never partially fabricated. A canon outage does
    NOT branch — run_normalize already defers cache-misses to ``pending``
    internally, and uncanonized tags simply stay off the trend axis.

This module also owns the trend materialization the V3.3 pure functions were
written for: weekly aggregates come from SQL here, the math stays in
``analysis.trend``, and the result is an idempotent trend_scores upsert.

langgraph is imported lazily (build_graph) so node logic stays testable with
the stdlib-only fixture runner, mirroring canon.py/report.py.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import TypedDict

from .analysis.trend import (
    deriv_velocity,
    detect_clusters,
    is_coldstart,
    view_velocity,
)
from .canon import run_normalize
from .report import run_report


class BatchState(TypedDict, total=False):
    week: date            # ISO week start (Monday), matches date_trunc('week')
    fetch: dict           # sanity counts from fetch_validate
    halted: "str | None"  # reason fetch_validate stopped the run
    canon: dict           # normalize_classify summary (pending>0 = deferred work)
    trend: dict           # trend_score summary
    report: "dict | None"  # report_gen summary; None = deferred (gateway outage)
    published: "list[str]"


def week_start(d: date) -> date:
    """Monday of d's ISO week — the same boundary as SQL date_trunc('week')."""
    return d - timedelta(days=d.weekday())


# ------------------------------------------------- db (thin): trend aggregates

def load_view_deltas(conn, week: date) -> "dict[int, dict]":
    """Per song: cumulative-view snapshots at the three week boundaries.

    Views are cumulative counters, so a week's gain is the difference between
    the highest reading before the boundary and the one a week earlier. Songs
    with no metrics at all produce no row — with no time series there is
    nothing to score or even watch.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT song_id,"
            " MAX(views) FILTER (WHERE metric_date < %(prev)s) AS v_prev,"
            " MAX(views) FILTER (WHERE metric_date < %(start)s) AS v_start,"
            " MAX(views) FILTER (WHERE metric_date < %(end)s) AS v_end,"
            " COUNT(DISTINCT date_trunc('week', metric_date))"
            "   FILTER (WHERE metric_date < %(end)s) AS weeks "
            "FROM metrics_daily GROUP BY song_id",
            {"prev": week - timedelta(days=7), "start": week,
             "end": week + timedelta(days=7)},
        )
        return {
            r[0]: {"v_prev": r[1], "v_start": r[2], "v_end": r[3], "weeks": r[4]}
            for r in cur.fetchall()
        }


def load_deriv_counts(conn) -> "dict[int, dict]":
    """New derived works per original per week (discovered_at is the signal)."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT original_song_id, date_trunc('week', discovered_at)::date, count(*) "
            "FROM derived_works GROUP BY 1, 2"
        )
        out: dict[int, dict] = {}
        for song_id, wk, n in cur.fetchall():
            out.setdefault(song_id, {})[wk] = n
        return out


def load_canon_tag_map(conn) -> "dict[int, set[str]]":
    """Confirmed canon tags per song — the only tags the trend axis sees."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT st.song_id, t.canon_name FROM song_tags st "
            "JOIN tags t ON t.id = st.tag_id "
            "WHERE t.canon_status IN ('canon_lite', 'canon_flash') "
            "AND t.canon_name IS NOT NULL"
        )
        out: dict[int, set[str]] = {}
        for song_id, canon in cur.fetchall():
            out.setdefault(song_id, set()).add(canon)
        return out


# ------------------------------------------------- pure: trend row computation

def compute_trend_rows(
    week: date,
    views: "dict[int, dict]",
    derivs: "dict[int, dict]",
    tag_map: "dict[int, set[str]]",
    *,
    top_n: int = 15,
    cluster_min_songs: int = 6,
) -> "list[tuple]":
    """(song_id, week, view_velocity, deriv_velocity, is_coldstart, cluster_id)."""
    scored: dict[int, "tuple[float | None, float | None]"] = {}
    coldstart: list[int] = []
    for song_id, v in views.items():
        if is_coldstart(v["weeks"]):
            coldstart.append(song_id)
            continue
        vv = None
        if v["v_end"] is not None and v["v_start"] is not None:
            prev_delta = (
                v["v_start"] - v["v_prev"] if v["v_prev"] is not None else 0
            )
            vv = view_velocity(prev_delta, v["v_end"] - v["v_start"])
        by_week = derivs.get(song_id, {})
        prev_4wk = [
            by_week.get(week - timedelta(days=7 * i), 0) for i in range(1, 5)
        ]
        dv = deriv_velocity(prev_4wk, by_week.get(week, 0))
        scored[song_id] = (vv, dv)

    top = sorted(
        (sid for sid, (vv, _) in scored.items() if vv is not None),
        key=lambda sid: scored[sid][0],
        reverse=True,
    )[:top_n]
    clusters = detect_clusters(
        [(sid, tag_map.get(sid, set())) for sid in top],
        min_songs=cluster_min_songs,
    )
    cluster_of: dict[int, int] = {}
    for i, c in enumerate(clusters, start=1):
        for sid in c["song_ids"]:
            cluster_of.setdefault(sid, i)  # a song in several clusters keeps the first

    rows = [
        (sid, week, vv, dv, False, cluster_of.get(sid))
        for sid, (vv, dv) in scored.items()
    ]
    rows += [(sid, week, None, None, True, None) for sid in coldstart]
    return rows


def persist_trend(conn, rows: "list[tuple]") -> None:
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO trend_scores (song_id, week, view_velocity, deriv_velocity, is_coldstart, cluster_id) "
            "VALUES (%s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (song_id, week) DO UPDATE SET "
            "view_velocity = EXCLUDED.view_velocity, deriv_velocity = EXCLUDED.deriv_velocity, "
            "is_coldstart = EXCLUDED.is_coldstart, cluster_id = EXCLUDED.cluster_id, "
            "computed_at = now()",
            rows,
        )
    conn.commit()


def materialize_trend(conn, week: date, **kwargs) -> "dict[str, int]":
    rows = compute_trend_rows(
        week,
        load_view_deltas(conn, week),
        load_deriv_counts(conn),
        load_canon_tag_map(conn),
        **kwargs,
    )
    persist_trend(conn, rows)
    return {
        "scored": sum(1 for r in rows if not r[4]),
        "coldstart": sum(1 for r in rows if r[4]),
        "clustered": sum(1 for r in rows if r[5] is not None),
    }


# ----------------------------------------------------------------- db: publish

def publish_report(conn, week: date, out_dir: "str | Path") -> "list[str]":
    """Regenerate the dashboard's data files for the week (plan: static site —
    the batch writes files, nothing serves dynamically)."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT narrative, evidence, model, generated_at FROM weekly_reports "
            "WHERE week = %s",
            (week,),
        )
        row = cur.fetchone()
    if row is None:
        return []
    payload = {
        "week": str(week),
        "narrative": row[0],
        "evidence": row[1],
        "model": row[2],
        "generated_at": row[3].isoformat(),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name in (f"report-{week}.json", "latest.json"):
        path = out / name
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        written.append(str(path))
    return written


# ----------------------------------------------------------------- graph nodes

def make_nodes(
    conn,
    *,
    gateway_url: "str | None" = None,
    api_key: "str | None" = None,
    publish_dir: "str | Path" = "dashboard/data",
    normalize=None,   # injected in tests; defaults to the real canon node
    report=None,      # injected in tests; defaults to the real report node
    trend=None,       # injected in tests; defaults to materialize_trend
) -> dict:
    """Build the node callables the graph wires together. Each returns a partial
    state update (LangGraph merges it into BatchState)."""
    normalize = normalize or (
        lambda: run_normalize(conn, base_url=gateway_url, api_key=api_key)
    )
    report = report or (
        lambda week: run_report(conn, week, base_url=gateway_url, api_key=api_key)
    )
    trend = trend or (lambda week: materialize_trend(conn, week))

    def fetch_validate(state: BatchState) -> dict:
        week = state["week"]
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM songs")
            n_songs = cur.fetchone()[0]
            cur.execute(
                "SELECT count(*), count(DISTINCT metric_date) FROM metrics_daily "
                "WHERE metric_date >= %s AND metric_date < %s",
                (week, week + timedelta(days=7)),
            )
            n_metrics, n_days = cur.fetchone()
        fetch = {"songs": n_songs, "week_metrics": n_metrics, "week_days": n_days}
        if n_songs == 0:
            return {"fetch": fetch, "halted": "no_songs"}
        if n_metrics == 0:
            return {"fetch": fetch, "halted": "no_metrics_this_week"}
        return {"fetch": fetch, "halted": None}

    def normalize_classify(state: BatchState) -> dict:
        # A gateway outage never surfaces here: run_normalize defers the
        # remainder to 'pending' (plan 0002 §6) and the batch carries on.
        return {"canon": normalize()}

    def trend_score(state: BatchState) -> dict:
        return {"trend": trend(state["week"])}

    def report_gen(state: BatchState) -> dict:
        return {"report": report(state["week"])}

    def publish(state: BatchState) -> dict:
        return {"published": publish_report(conn, state["week"], publish_dir)}

    return {
        "fetch_validate": fetch_validate,
        "normalize_classify": normalize_classify,
        "trend_score": trend_score,
        "report_gen": report_gen,
        "publish": publish,
    }


def after_fetch(state: BatchState) -> str:
    return "halt" if state.get("halted") else "continue"


def after_report(state: BatchState) -> str:
    # None = the gateway was down: defer the week's report, publish nothing.
    return "publish" if state.get("report") else "defer"


def build_graph(nodes: dict):
    from langgraph.graph import END, START, StateGraph  # lazy: fixture tests stay stdlib-only

    g = StateGraph(BatchState)
    for name, fn in nodes.items():
        g.add_node(name, fn)
    g.add_edge(START, "fetch_validate")
    g.add_conditional_edges(
        "fetch_validate", after_fetch,
        {"continue": "normalize_classify", "halt": END},
    )
    g.add_edge("normalize_classify", "trend_score")
    g.add_edge("trend_score", "report_gen")
    g.add_conditional_edges(
        "report_gen", after_report,
        {"publish": "publish", "defer": END},
    )
    g.add_edge("publish", END)
    return g.compile()


def run_weekly(
    conn,
    week: "date | None" = None,
    *,
    gateway_url: "str | None" = None,
    api_key: "str | None" = None,
    publish_dir: "str | Path" = "dashboard/data",
) -> BatchState:
    """The V3.4 entry point: assemble and run the weekly batch graph."""
    week = week or week_start(date.today())
    graph = build_graph(
        make_nodes(conn, gateway_url=gateway_url, api_key=api_key,
                   publish_dir=publish_dir)
    )
    return graph.invoke({"week": week})


if __name__ == "__main__":
    import argparse

    import psycopg

    from .config import DATABASE_URL, GATEWAY_API_KEY, GATEWAY_URL

    ap = argparse.ArgumentParser(description="Run the weekly batch graph")
    ap.add_argument("--week", help="ISO week start (YYYY-MM-DD); default = this week")
    args = ap.parse_args()
    wk = week_start(date.fromisoformat(args.week)) if args.week else None
    with psycopg.connect(DATABASE_URL) as conn:
        final = run_weekly(conn, wk, gateway_url=GATEWAY_URL, api_key=GATEWAY_API_KEY)
        out = dict(final)
        out["week"] = str(out["week"])
        print(json.dumps(out, ensure_ascii=False, default=str))
