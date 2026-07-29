"""DB round-trip for report_gen (plan 0002 R3) + the DB FinOps rollup (R4).

Exercises load_evidence / persist_report against the real dev Postgres with a
scripted narrative (no LLM), and rollup_db against seeded canon_status rows.
Gated behind DB_ROUNDTRIP=1 so the default fixture suite stays DB-free:

    docker compose -f docker-compose.dev.yml run --rm app \
      sh -c "python -m src.db.migrate && DB_ROUNDTRIP=1 python -m pytest tests/test_report_db.py -q"
"""
import json
import os
from datetime import date

import pytest

if os.environ.get("DB_ROUNDTRIP") != "1":
    pytest.skip("needs the dev DB (set DB_ROUNDTRIP=1)", allow_module_level=True)

import psycopg

from src.config import DATABASE_URL
from src.finops import rollup_db
from src.report import load_evidence, persist_report
from src.routing import FLASH, LITE

WEEK = date(2026, 7, 13)       # a Monday
PREV = date(2026, 7, 6)


@pytest.fixture()
def conn():
    with psycopg.connect(DATABASE_URL) as c:
        with c.cursor() as cur:
            for t in ("song_tags", "trend_scores", "weekly_reports", "tags", "songs"):
                cur.execute(f"DELETE FROM {t} WHERE TRUE AND "
                            + ("week = %s" if t in ("trend_scores", "weekly_reports")
                               else "id BETWEEN -930999 AND -930000" if t == "songs"
                               else "id BETWEEN -931999 AND -931000" if t == "tags"
                               else "song_id BETWEEN -930999 AND -930000"),
                            ((WEEK,) if t in ("trend_scores", "weekly_reports") else ()))
        # songs: three this-week, one prior-week, one cold-start this-week
        with c.cursor() as cur:
            cur.execute(
                "INSERT INTO songs (id, title, publish_date) VALUES "
                "(-930001, '加速テスト', '2026-07-14'),"
                "(-930002, '第二曲',   '2026-07-15'),"
                "(-930003, '先週曲',   '2026-07-08'),"
                "(-930004, '新曲ワッチ', '2026-07-16')"
            )
            cur.execute(
                "INSERT INTO tags (id, name, canon_name, canon_confidence, canon_status) VALUES "
                "(-931001, 'ろっく', 'ロック',   0.95, 'canon_lite'),"
                "(-931002, 'ばらーど', 'バラード', 0.92, 'canon_lite'),"
                "(-931003, 'みくみく', '初音ミク', 0.90, 'canon_flash'),"  # escalated cache row
                "(-931004, '未処理',  NULL,      NULL, 'pending')"
            )
            cur.execute(
                "INSERT INTO song_tags (song_id, tag_id) VALUES "
                "(-930001, -931001), (-930002, -931001), (-930003, -931002)"
            )
            cur.execute(
                "INSERT INTO trend_scores (song_id, week, view_velocity, deriv_velocity, "
                "is_coldstart, cluster_id) VALUES "
                "(-930001, %s, 3.5, 1.2, FALSE, 0),"
                "(-930002, %s, 2.1, NULL, FALSE, 0),"
                "(-930004, %s, NULL, NULL, TRUE, NULL)",
                (WEEK, WEEK, WEEK),
            )
        c.commit()
        yield c
        with c.cursor() as cur:
            cur.execute("DELETE FROM song_tags WHERE song_id BETWEEN -930999 AND -930000")
            cur.execute("DELETE FROM trend_scores WHERE week = %s", (WEEK,))
            cur.execute("DELETE FROM weekly_reports WHERE week = %s", (WEEK,))
            cur.execute("DELETE FROM tags WHERE id BETWEEN -931999 AND -931000")
            cur.execute("DELETE FROM songs WHERE id BETWEEN -930999 AND -930000")
        c.commit()


def test_report_roundtrip(conn):
    ev = load_evidence(conn, WEEK)

    # top_songs: velocity desc, cold-start excluded
    assert [s["song_id"] for s in ev["top_songs"]] == [-930001, -930002]
    assert ev["top_songs"][0]["view_velocity"] == 3.5
    # clusters: both top songs share cluster 0
    assert ev["clusters"] and set(ev["clusters"][0]["titles"]) == {"加速テスト", "第二曲"}
    # watchlist: the cold-start song
    assert [w["song_id"] for w in ev["watchlist"]] == [-930004]
    # tag_deltas: ロック jumped from 0% (prior week) to 2/3 of this week's songs
    rock = next(d for d in ev["tag_deltas"] if d["canon"] == "ロック")
    assert rock["delta_pp"] > 0

    # persist a scripted narrative, read it back
    persist_report(conn, WEEK, "이번 주 핵심 신호: 로큰롤 태그 급등.", ev)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT narrative, evidence, model FROM weekly_reports WHERE week = %s", (WEEK,)
        )
        narrative, evidence, model = cur.fetchone()
    assert "로큰롤" in narrative
    assert json.loads(evidence)["top_songs"][0]["song_id"] == -930001 \
        if isinstance(evidence, str) else evidence["top_songs"][0]["song_id"] == -930001
    assert model == "gw-flash"


def test_rollup_db_reflects_seeded_status(conn):
    r = rollup_db(conn)
    # our four seeded tags: 2 canon_lite, 1 canon_flash, 1 pending.
    # Key off the LITE/FLASH constants, not literals: the tier-1 slot is a slot,
    # and hardcoding its current occupant is what broke this test on the
    # 2026-07-29 gw-lite -> gw-gemma swap.
    assert r["tiers"][LITE] >= 2
    assert r["tiers"][FLASH] >= 1
    assert r["pending_backlog"] >= 1
    assert 0.0 <= r["cache_hit_rate"] <= 1.0
