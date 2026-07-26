"""DB round-trip checks for trend materialization + publish (plan 0001 V3.4).

Synthetic 3-week history on negative ids (never collide with VocaDB ids).
Gated like the other DB suites:

    docker compose -f docker-compose.dev.yml run --rm app \
      sh -c "python -m src.db.migrate && DB_ROUNDTRIP=1 python -m pytest tests/test_pipeline_db.py -q"
"""
import json
import os
from datetime import date, timedelta
from pathlib import Path

import pytest

if os.environ.get("DB_ROUNDTRIP") != "1":
    pytest.skip("needs the dev DB (set DB_ROUNDTRIP=1)", allow_module_level=True)

import psycopg

from src.config import DATABASE_URL
from src.pipeline import materialize_trend, publish_report, week_start

WEEK = week_start(date(2026, 6, 15))  # a past week, clear of live-seeded data
LO, HI = -920999, -920000


@pytest.fixture()
def conn():
    with psycopg.connect(DATABASE_URL) as c:
        with c.cursor() as cur:
            cur.execute("DELETE FROM songs WHERE id BETWEEN %s AND %s", (LO, HI))
            cur.execute("DELETE FROM weekly_reports WHERE week = %s", (WEEK,))
        c.commit()
        yield c
        with c.cursor() as cur:
            cur.execute("DELETE FROM songs WHERE id BETWEEN %s AND %s", (LO, HI))
            cur.execute("DELETE FROM weekly_reports WHERE week = %s", (WEEK,))
        c.commit()


def seed_song(cur, song_id, title):
    cur.execute(
        "INSERT INTO songs (id, title, publish_date) VALUES (%s, %s, %s)",
        (song_id, title, WEEK - timedelta(days=30)),
    )


def seed_series(cur, song_id, weekly_views):
    """One cumulative reading per week, landing INSIDE that week (Saturday).

    The last entry is WEEK itself; earlier entries walk back one week each.
    """
    for i, v in enumerate(weekly_views):
        wk = WEEK - timedelta(days=7 * (len(weekly_views) - 1 - i))
        cur.execute(
            "INSERT INTO metrics_daily (song_id, metric_date, source, views) "
            "VALUES (%s, %s, 'niconico', %s)",
            (song_id, wk + timedelta(days=5), v),
        )


def test_materialize_trend_velocity_and_coldstart(conn):
    with conn.cursor() as cur:
        # 3 weekly readings: gains +10k then +30k inside WEEK -> velocity 3.0
        seed_song(cur, -920001, "가속곡")
        seed_series(cur, -920001, [1_000, 11_000, 41_000])
        # single reading -> < 2 weeks of history -> coldstart
        seed_song(cur, -920002, "신곡")
        seed_series(cur, -920002, [300])
    conn.commit()

    summary = materialize_trend(conn, WEEK)
    assert summary["scored"] >= 1 and summary["coldstart"] >= 1

    with conn.cursor() as cur:
        cur.execute(
            "SELECT view_velocity, is_coldstart FROM trend_scores "
            "WHERE song_id = -920001 AND week = %s", (WEEK,))
        vv, cold = cur.fetchone()
        assert vv == pytest.approx(3.0) and cold is False
        cur.execute(
            "SELECT view_velocity, is_coldstart FROM trend_scores "
            "WHERE song_id = -920002 AND week = %s", (WEEK,))
        vv, cold = cur.fetchone()
        assert vv is None and cold is True

    # idempotent re-run: same rows, no dupes (PK), values stable
    materialize_trend(conn, WEEK)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM trend_scores WHERE song_id BETWEEN %s AND %s "
            "AND week = %s", (LO, HI, WEEK))
        assert cur.fetchone()[0] == 2


def test_publish_report_writes_week_and_latest(conn, tmp_path):
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO weekly_reports (week, narrative, evidence, model) "
            "VALUES (%s, %s, %s, 'gw-flash')",
            (WEEK, "테스트 내러티브", json.dumps({"week": str(WEEK)})),
        )
    conn.commit()

    written = publish_report(conn, WEEK, tmp_path)
    assert {Path(p).name for p in written} == {f"report-{WEEK}.json", "latest.json"}
    payload = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert payload["narrative"] == "테스트 내러티브"
    assert payload["week"] == str(WEEK)


def test_publish_report_missing_week_writes_nothing(conn, tmp_path):
    assert publish_report(conn, WEEK + timedelta(days=700), tmp_path) == []
