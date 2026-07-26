"""DB round-trip checks for the adapter upsert layer (plan 0001 V2.1/V2.2).

Runs against the real dev Postgres (docker-compose.dev.yml), gated like
test_canon_db.py:

    docker compose -f docker-compose.dev.yml run --rm app \
      sh -c "python -m src.db.migrate && DB_ROUNDTRIP=1 python -m pytest tests/test_adapters_db.py -q"
"""
import os
from datetime import date

import pytest

if os.environ.get("DB_ROUNDTRIP") != "1":
    pytest.skip("needs the dev DB (set DB_ROUNDTRIP=1)", allow_module_level=True)

import psycopg

from src.adapters.niconico import upsert_metrics
from src.adapters.vocadb import ArtistLink, Song, SongBundle, TagLink, upsert_bundles
from src.config import DATABASE_URL

# Negative ids: VocaDB ids are always positive (1M+ by 2026), so these can never collide with seeded rows.
LO, HI = -910999, -910000


def bundle(song_id, title, *, nico=None, original=None, tag_name="ロック"):
    return SongBundle(
        song=Song(song_id, title, date(2026, 6, 1), "Original", nico, None),
        artists=[ArtistLink(-910100, "テストP", "Producer", "Producer")],
        tags=[TagLink(-910200, tag_name, "Genres")],
        original_version_id=original,
    )


@pytest.fixture()
def conn():
    with psycopg.connect(DATABASE_URL) as c:
        yield c
        with c.cursor() as cur:
            cur.execute("DELETE FROM songs WHERE id BETWEEN %s AND %s", (LO, HI))
            cur.execute("DELETE FROM artists WHERE id BETWEEN %s AND %s", (LO, HI))
            cur.execute("DELETE FROM tags WHERE id BETWEEN %s AND %s", (LO, HI))
        c.commit()


def counts(conn):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT (SELECT count(*) FROM songs WHERE id BETWEEN %(lo)s AND %(hi)s),"
            " (SELECT count(*) FROM song_artists WHERE song_id BETWEEN %(lo)s AND %(hi)s),"
            " (SELECT count(*) FROM song_tags WHERE song_id BETWEEN %(lo)s AND %(hi)s),"
            " (SELECT count(*) FROM derived_works WHERE derived_song_id BETWEEN %(lo)s AND %(hi)s)",
            {"lo": LO, "hi": HI},
        )
        return cur.fetchone()


def test_upsert_idempotent_and_updates(conn):
    bundles = [bundle(-910001, "v1", nico="sm910001")]
    upsert_bundles(conn, bundles)
    first = counts(conn)
    # re-run with a changed title: same rows, updated content
    upsert_bundles(conn, [bundle(-910001, "v2", nico="sm910001")])
    assert counts(conn) == first
    with conn.cursor() as cur:
        cur.execute("SELECT title FROM songs WHERE id = -910001")
        assert cur.fetchone()[0] == "v2"


def test_reseed_never_touches_canon_cache(conn):
    upsert_bundles(conn, [bundle(-910002, "曲", tag_name="ろっく")])
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE tags SET canon_name = 'ロック', canon_confidence = 0.95, "
            "canon_status = 'canon_lite' WHERE id = -910200"
        )
    conn.commit()
    # a later seed sees the tag renamed upstream
    upsert_bundles(conn, [bundle(-910002, "曲", tag_name="ROCK")])
    with conn.cursor() as cur:
        cur.execute(
            "SELECT name, canon_name, canon_status FROM tags WHERE id = -910200"
        )
        name, canon_name, canon_status = cur.fetchone()
    assert name == "ROCK"                      # raw name follows the source
    assert (canon_name, canon_status) == ("ロック", "canon_lite")  # cache intact


def test_derived_links_only_when_original_present(conn):
    bundles = [
        bundle(-910003, "原曲"),
        bundle(-910004, "歌ってみた", original=-910003),   # original in this seed
        bundle(-910005, "外部カバー", original=-919999),   # original unknown -> skipped
    ]
    summary = upsert_bundles(conn, bundles)
    assert summary["derived_seen"] == 2
    assert summary["derived_linked"] == 1
    assert counts(conn)[3] == 1


def test_metrics_same_day_updates_next_day_inserts(conn):
    upsert_bundles(conn, [bundle(-910006, "計測曲", nico="sm910006")])
    d1, d2 = date(2026, 7, 25), date(2026, 7, 26)
    upsert_metrics(conn, [(-910006, d1, 100, 5, 1, 10)])
    upsert_metrics(conn, [(-910006, d1, 150, 6, 1, 12)])  # same-day re-run
    upsert_metrics(conn, [(-910006, d2, 200, 7, 2, 15)])
    with conn.cursor() as cur:
        cur.execute(
            "SELECT metric_date, views FROM metrics_daily "
            "WHERE song_id = -910006 ORDER BY metric_date"
        )
        rows = cur.fetchall()
    assert rows == [(d1, 150), (d2, 200)]  # 2 rows, latest reading wins per day
