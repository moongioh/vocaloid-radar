"""Idempotent schema migration for the vocaloid DB (plan 0001 V1.1).

Runs src/db/schema.sql, which is written to be safe to re-run — every object
uses IF NOT EXISTS. No migration framework: the schema is small and additive.

    python -m src.db.migrate
"""
import pathlib

import psycopg

from src.config import DATABASE_URL

SCHEMA_SQL = pathlib.Path(__file__).with_name("schema.sql")


def migrate(database_url: str = DATABASE_URL) -> None:
    sql = SCHEMA_SQL.read_text(encoding="utf-8")
    # autocommit so the whole multi-statement DDL script runs via the simple
    # query protocol (psycopg3 allows several statements in one execute() call
    # when no parameters are bound).
    with psycopg.connect(database_url, autocommit=True) as conn:
        conn.execute(sql)


if __name__ == "__main__":
    migrate()
    print("migration applied")
