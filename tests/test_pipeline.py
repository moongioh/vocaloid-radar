"""Fixture tests for the weekly batch (plan 0001 V3.4).

compute_trend_rows mirrors the plan's 명세-예시 table; graph-wiring tests use
fake nodes and skip when langgraph is absent (the host stdlib runner) — the
container suite always runs them.
"""
from datetime import date, timedelta

import pytest

from src.pipeline import (
    BatchState,
    after_fetch,
    after_report,
    compute_trend_rows,
    week_start,
)

WEEK = date(2026, 7, 20)  # a Monday


def views(v_prev=None, v_start=None, v_end=None, weeks=3):
    return {"v_prev": v_prev, "v_start": v_start, "v_end": v_end, "weeks": weeks}


def by_song(rows):
    return {r[0]: r for r in rows}


def test_week_start_is_monday():
    assert week_start(date(2026, 7, 26)) == WEEK  # Sunday -> that week's Monday
    assert week_start(WEEK) == WEEK


def test_view_velocity_spec_example():
    # 지난주 +10k, 이번주 +30k → 3.0
    rows = by_song(compute_trend_rows(
        WEEK, {1: views(v_prev=0, v_start=10_000, v_end=40_000)}, {}, {}))
    assert rows[1][2] == pytest.approx(3.0)
    assert rows[1][4] is False


def test_coldstart_spec_example():
    # 시계열 2주 미만 곡은 velocity 산출 제외
    rows = by_song(compute_trend_rows(
        WEEK, {1: views(v_start=100, v_end=500, weeks=1)}, {}, {}))
    assert rows[1][4] is True
    assert rows[1][2] is None and rows[1][3] is None


def test_no_baseline_scores_without_velocity():
    # flat previous week: still a scored row, velocity undefined (None)
    rows = by_song(compute_trend_rows(
        WEEK, {1: views(v_prev=500, v_start=500, v_end=900)}, {}, {}))
    assert rows[1][4] is False and rows[1][2] is None


def test_deriv_velocity_spec_example():
    # 4주 평균 2건, 이번주 8건 → 4.0
    derivs = {1: {WEEK - timedelta(days=7 * i): 2 for i in range(1, 5)} | {WEEK: 8}}
    rows = by_song(compute_trend_rows(
        WEEK, {1: views(v_prev=0, v_start=10, v_end=20)}, derivs, {}))
    assert rows[1][3] == pytest.approx(4.0)


def test_cluster_spec_example():
    # 상위 곡 중 6곡이 같은 태그쌍 → 클러스터 1건, cluster_id 부여
    v = {i: views(v_prev=0, v_start=1000, v_end=1000 + 100 * i) for i in range(1, 8)}
    tags = {i: {"ロック", "初音ミク"} for i in range(1, 7)}  # 6 share the pair
    tags[7] = {"バラード"}
    rows = by_song(compute_trend_rows(WEEK, v, {}, tags))
    assert all(rows[i][5] == 1 for i in range(1, 7))
    assert rows[7][5] is None


def test_after_fetch_branch():
    assert after_fetch(BatchState(halted="no_songs")) == "halt"
    assert after_fetch(BatchState(halted=None)) == "continue"


def test_after_report_branch():
    assert after_report(BatchState(report={"week": "2026-07-20"})) == "publish"
    assert after_report(BatchState(report=None)) == "defer"  # outage defers


# ------------------------------------------------- graph wiring (needs langgraph)

def wired_graph(calls, *, halted=None, report_summary):
    """A real compiled graph over fake nodes that record execution order."""
    from src.pipeline import build_graph

    def node(name, update):
        def fn(state):
            calls.append(name)
            return update
        return fn

    return build_graph({
        "fetch_validate": node("fetch", {"halted": halted}),
        "normalize_classify": node("canon", {"canon": {"total": 0}}),
        "trend_score": node("trend", {"trend": {"scored": 0}}),
        "report_gen": node("report", {"report": report_summary}),
        "publish": node("publish", {"published": []}),
    })


def test_graph_full_path_order():
    pytest.importorskip("langgraph")
    calls = []
    final = wired_graph(calls, report_summary={"chars": 1}).invoke({"week": WEEK})
    assert calls == ["fetch", "canon", "trend", "report", "publish"]
    assert final["report"] == {"chars": 1}


def test_graph_halts_on_no_data():
    pytest.importorskip("langgraph")
    calls = []
    wired_graph(calls, halted="no_songs", report_summary=None).invoke({"week": WEEK})
    assert calls == ["fetch"]  # nothing downstream ran


def test_graph_defers_publish_on_outage():
    pytest.importorskip("langgraph")
    calls = []
    wired_graph(calls, report_summary=None).invoke({"week": WEEK})
    assert calls == ["fetch", "canon", "trend", "report"]  # publish skipped
