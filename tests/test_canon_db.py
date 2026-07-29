"""DB round-trip check for the tag_canon cache layer (plan 0002 R2).

Exercises load_work / load_vocab / persist against the real dev Postgres
(docker-compose.dev.yml). No LLM calls — the caller is scripted. Gated behind
DB_ROUNDTRIP=1 so the default fixture suite stays DB-free:

    docker compose -f docker-compose.dev.yml run --rm app \
      sh -c "python -m src.db.migrate && DB_ROUNDTRIP=1 python -m pytest tests/test_canon_db.py -q"
"""
import os

import pytest

if os.environ.get("DB_ROUNDTRIP") != "1":
    pytest.skip("needs the dev DB (set DB_ROUNDTRIP=1)", allow_module_level=True)

import psycopg

from src.canon import load_vocab, load_work, normalize_classify, persist
from src.config import DATABASE_URL
from src.routing import Status

# High ids far away from real VocaDB tag ids seeded by collection.
SEEDED = {
    900001: ("ロックの canon 시드", "canon_lite"),   # existing cache row -> vocab source
    900002: ("ろっく変形", "pending"),               # in-vocab mapping candidate
    900003: ("エレスウィング", "pending"),           # new-term candidate
    900004: ("前回強등タグ", "demoted"),             # demoted retry must be picked up
}


@pytest.fixture()
def conn():
    with psycopg.connect(DATABASE_URL) as c:
        with c.cursor() as cur:
            cur.execute("DELETE FROM tags WHERE id BETWEEN 900000 AND 900010")
            cur.execute(
                "INSERT INTO tags (id, name, canon_name, canon_confidence, canon_status) VALUES "
                "(900001, %s, 'ロック', 0.99, 'canon_lite'),"
                "(900002, %s, NULL, NULL, 'pending'),"
                "(900003, %s, NULL, NULL, 'pending'),"
                "(900004, %s, NULL, NULL, 'demoted')",
                tuple(SEEDED[i][0] for i in sorted(SEEDED)),
            )
        c.commit()
        yield c
        with c.cursor() as cur:
            cur.execute("DELETE FROM tags WHERE id BETWEEN 900000 AND 900010")
        c.commit()


def scripted_caller(model, items):
    out = []
    for it in items:
        if it.id == 900003:  # new term: lite proposes, flash confirms
            out.append({"id": it.id, "canonical": "エレクトロスウィング",
                        "is_new": True, "confidence": 0.9})
        else:  # everything else maps into the existing vocabulary
            out.append({"id": it.id, "canonical": "ロック", "confidence": 0.95})
    return out


def test_db_roundtrip(conn):
    # --- load: only pending/demoted are work; canon_* rows feed the vocabulary
    # limit is deliberately huge: since plan 0004 load_work orders by USAGE, and
    # these fixtures have no song_tags rows, they sort last and the default
    # limit=500 would leave them out of the result entirely.
    items = load_work(conn, limit=100_000)
    work_ids = {it.id for it in items if 900000 <= it.id <= 900010}
    assert work_ids == {900002, 900003, 900004}  # cache hit (900001) excluded

    # Route ONLY the seeded rows: load_work sweeps in any real seeded tags too,
    # and persisting scripted-caller results for those corrupts live data
    # (it did — 2026-07-26, every real tag became canon 'ロック').
    items = [it for it in items if 900000 <= it.id <= 900010]

    vocab = load_vocab(conn)
    assert "ロック" in vocab

    # --- route with a scripted caller (no live LLM), then persist
    results = normalize_classify(items, vocab, scripted_caller)
    persist(conn, results)

    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, name, canon_name, canon_status FROM tags "
            "WHERE id BETWEEN 900000 AND 900010 ORDER BY id"
        )
        rows = {r[0]: r for r in cur.fetchall()}

    # cache row untouched
    assert rows[900001][3] == "canon_lite"
    # in-vocab mapping accepted on lite
    assert rows[900002][2] == "ロック" and rows[900002][3] == "canon_lite"
    # new term escalated + confirmed on flash
    assert rows[900003][2] == "エレクトロスウィング" and rows[900003][3] == "canon_flash"
    # demoted retry re-entered the ladder and was accepted; raw name preserved
    assert rows[900004][3] in ("canon_lite", "canon_flash")
    assert rows[900004][1] == SEEDED[900004][0]


@pytest.fixture()
def usage_conn():
    """Three pending tags with 3 / 1 / 2 song_tags rows, so usage order != id order.

    The bug this guards (plan 0004) was invisible to fixture tests precisely
    because usage lives in a different table than the one load_work used to read.
    """
    with psycopg.connect(DATABASE_URL) as c:
        with c.cursor() as cur:
            cur.execute("DELETE FROM tags WHERE id BETWEEN 900005 AND 900007")
            cur.execute("DELETE FROM songs WHERE id BETWEEN 900001 AND 900003")
            cur.execute(
                "INSERT INTO songs (id, title) VALUES "
                "(900001, 'fixture a'), (900002, 'fixture b'), (900003, 'fixture c')"
            )
            cur.execute(
                "INSERT INTO tags (id, name, canon_status) VALUES "
                "(900005, 'fixture heavy', 'pending'),"   # 3 uses
                "(900006, 'fixture light', 'pending'),"   # 1 use
                "(900007, 'fixture mid',   'pending')"    # 2 uses
            )
            cur.execute(
                "INSERT INTO song_tags (song_id, tag_id) VALUES "
                "(900001, 900005), (900002, 900005), (900003, 900005),"
                "(900001, 900006),"
                "(900001, 900007), (900002, 900007)"
            )
        c.commit()
        yield c
        with c.cursor() as cur:  # song_tags goes with them (ON DELETE CASCADE)
            cur.execute("DELETE FROM tags WHERE id BETWEEN 900005 AND 900007")
            cur.execute("DELETE FROM songs WHERE id BETWEEN 900001 AND 900003")
        c.commit()


def test_load_work_orders_by_usage_not_id(usage_conn):
    """Regression: the founding-vocabulary defect of 2026-07-29 (plan 0004).

    load_work used ORDER BY id, so the closed-loop vocabulary was seeded from
    whichever tags happened to sort first rather than from the ones the corpus
    actually uses. Asserting relative order within the fixture band is enough —
    real tags interleave by their own usage, which is the point.
    """
    order = [it.id for it in load_work(usage_conn, limit=100_000)
             if 900005 <= it.id <= 900007]
    assert order == [900005, 900007, 900006]  # 3 uses, 2 uses, 1 use


def test_load_vocab_includes_seed_before_any_canon_row(usage_conn):
    """The cold start is gone: an allowlist exists even with zero canon rows."""
    from src.canon import SEED_VOCAB

    vocab = load_vocab(usage_conn)
    assert SEED_VOCAB <= vocab          # seed is a floor...
    assert "MMD Model" in vocab         # ...and it carries the family whose absence
    #                                     sent `YYB Kagamine Len` to Beta Voicebank
