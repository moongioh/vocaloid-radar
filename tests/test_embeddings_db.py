"""DB round-trip checks for the V3.2 embedding layer (real dev Postgres).

Scope discipline (memory: db-shared-devdb-test-fixtures-negative-ids-scoped-persist):
every write here stays inside the negative fixture band. run_embed itself is NOT
driven against this shared DB — whole-table orchestration would rewrite
live-seeded rows; its loop is covered fixture-side with patched I/O.

    docker compose -f docker-compose.dev.yml run --rm app \
      sh -c "python -m src.db.migrate && DB_ROUNDTRIP=1 python -m pytest tests/test_embeddings_db.py -q"
"""
import os

import pytest

if os.environ.get("DB_ROUNDTRIP") != "1":
    pytest.skip("needs the dev DB (set DB_ROUNDTRIP=1)", allow_module_level=True)

import psycopg

from src.config import DATABASE_URL
from src.embeddings import (
    build_source_text,
    load_existing,
    load_source_rows,
    select_stale,
    similar,
    upsert_embeddings,
)

# Negative ids: real VocaDB ids are always positive — this band can never collide.
LO, HI = -920999, -920000
DIM = 1024


def vec(*head):
    v = [0.0] * DIM
    for i, x in enumerate(head):
        v[i] = float(x)
    return v


@pytest.fixture()
def conn():
    with psycopg.connect(DATABASE_URL) as c:
        with c.cursor() as cur:
            cur.executemany(
                "INSERT INTO songs (id, title, song_type) VALUES (%s, %s, 'Original') "
                "ON CONFLICT (id) DO NOTHING",
                [(-920001, "rock A"), (-920002, "rock B"), (-920003, "ballad C")],
            )
            cur.execute(
                "INSERT INTO artists (id, name) VALUES (-920101, 'zP'), (-920102, 'aP') "
                "ON CONFLICT (id) DO NOTHING"
            )
            # zP appears twice under different roles — one artist, one name in the text
            cur.executemany(
                "INSERT INTO song_artists (song_id, artist_id, role) VALUES (%s, %s, %s) "
                "ON CONFLICT DO NOTHING",
                [
                    (-920001, -920101, "Composer"),
                    (-920001, -920101, "Vocalist"),
                    (-920001, -920102, ""),
                ],
            )
            # one canon-confirmed tag (canon name must win) + one pending (raw name)
            cur.executemany(
                "INSERT INTO tags (id, name, canon_name, canon_status) VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (id) DO NOTHING",
                [
                    (-920201, "ROCK", "ロック", "canon_lite"),
                    (-920202, "謎タグ", None, "pending"),
                ],
            )
            cur.executemany(
                "INSERT INTO song_tags (song_id, tag_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                [(-920001, -920201), (-920001, -920202)],
            )
        c.commit()
        yield c
        with c.cursor() as cur:
            # song_embeddings rows cascade with songs
            cur.execute("DELETE FROM songs WHERE id BETWEEN %s AND %s", (LO, HI))
            cur.execute("DELETE FROM artists WHERE id BETWEEN %s AND %s", (LO, HI))
            cur.execute("DELETE FROM tags WHERE id BETWEEN %s AND %s", (LO, HI))
        c.commit()


def band_rows(conn):
    return {r.song_id: r for r in load_source_rows(conn) if LO <= r.song_id <= HI}


def test_source_rows_artists_sorted_and_canon_tag_wins(conn):
    r = band_rows(conn)[-920001]
    assert r.artists == ["aP", "zP"]                      # ORDER BY a.name
    assert sorted(r.tags) == ["ロック", "謎タグ"]          # canon name replaced ROCK


def test_upsert_select_stale_and_similar(conn):
    rows = band_rows(conn)
    items = [
        (-920001, build_source_text(rows[-920001]), vec(1.0, 0.0)),
        (-920002, build_source_text(rows[-920002]), vec(0.9, 0.1)),
        (-920003, build_source_text(rows[-920003]), vec(0.0, 1.0)),
    ]
    upsert_embeddings(conn, items, "test-embed")

    # idempotence signal: with stored (source_text, model) matching, none are stale
    existing = {sid: v for sid, v in load_existing(conn).items() if LO <= sid <= HI}
    assert select_stale(list(rows.values()), existing, "test-embed") == []
    # model swap or text drift -> stale again
    assert len(select_stale(list(rows.values()), existing, "other-model")) == 3

    # re-upsert (update path) must not duplicate
    upsert_embeddings(conn, items[:1], "test-embed")
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM song_embeddings WHERE song_id BETWEEN %s AND %s", (LO, HI)
        )
        assert cur.fetchone()[0] == 3

    # similar: rock A's nearest fixture neighbour is rock B; ballad C farther
    neighbours = similar(conn, -920001, k=10)
    band = [(sid, dist) for sid, _, dist in neighbours if LO <= sid <= HI]
    assert band and band[0][0] == -920002
    dist_by_id = dict(band)
    assert dist_by_id[-920002] < dist_by_id[-920003]


def test_similar_unknown_song_returns_empty(conn):
    assert similar(conn, -920999) == []
