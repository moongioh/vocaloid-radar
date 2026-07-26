"""Niconico Snapshot Search v2 adapter — daily metrics_daily rows (plan 0001 V2.2).

The API is officially undocumented-but-alive (docs 403 since the 2024 incident),
which is exactly why this lives behind the adapter pattern: everything
Niconico-specific is in this file, swappable for YouTube Data API later.

One request covers a batch of known video ids via a jsonFilter or-equal —
no search-by-keyword, so results map 1:1 onto our songs. Videos the API no
longer returns (deleted/private) are reported as ``missing``, never guessed.

Same-day re-run is an update (latest reading wins); the (song_id, metric_date,
source) PK is what makes the V2.3 "0 dupes" gate structural.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date

from .transport import make_getter

BASE_URL = "https://snapshot.search.nicovideo.jp"
SEARCH_PATH = "/api/v2/snapshot/video/contents/search"
BATCH_SIZE = 20  # ids per request; keeps the jsonFilter URL comfortably short
MIN_INTERVAL = 2.0  # courtesy pacing between requests
SOURCE = "niconico"


@dataclass(frozen=True)
class Metric:
    content_id: str
    views: "int | None"
    comments: "int | None"
    mylists: "int | None"
    likes: "int | None"


# --------------------------------------------------------------- pure: request

def build_params(ids: "list[str]") -> dict:
    return {
        "q": "",
        "targets": "title",
        "fields": "contentId,viewCounter,commentCounter,mylistCounter,likeCounter",
        "jsonFilter": json.dumps(
            {
                "type": "or",
                "filters": [
                    {"type": "equal", "field": "contentId", "value": i} for i in ids
                ],
            },
            ensure_ascii=False,
        ),
        "_sort": "-viewCounter",  # required by the API; any stable field works
        "_limit": len(ids),
    }


def chunked(ids: "list[str]", size: int = BATCH_SIZE) -> "list[list[str]]":
    return [ids[i : i + size] for i in range(0, len(ids), size)]


# ---------------------------------------------------------------- pure: parse

def parse_response(data: dict) -> "dict[str, Metric]":
    out = {}
    for row in data.get("data") or []:
        cid = row.get("contentId")
        if not cid:
            continue
        out[cid] = Metric(
            content_id=cid,
            views=row.get("viewCounter"),
            comments=row.get("commentCounter"),
            mylists=row.get("mylistCounter"),
            likes=row.get("likeCounter"),
        )
    return out


def fetch_metrics(get, ids: "list[str]") -> "dict[str, Metric]":
    metrics: dict[str, Metric] = {}
    for batch in chunked(ids):
        metrics.update(parse_response(get(SEARCH_PATH, build_params(batch))))
    return metrics


# ----------------------------------------------------------------- db (thin)

def load_targets(conn) -> "list[tuple[int, str]]":
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, nico_video_id FROM songs "
            "WHERE nico_video_id IS NOT NULL ORDER BY id"
        )
        return [(r[0], r[1]) for r in cur.fetchall()]


def upsert_metrics(conn, rows: "list[tuple]") -> None:
    """Rows = (song_id, metric_date, views, comments, mylists, likes)."""
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO metrics_daily (song_id, metric_date, source, views, comments, mylists, likes) "
            f"VALUES (%s, %s, '{SOURCE}', %s, %s, %s, %s) "
            "ON CONFLICT (song_id, metric_date, source) DO UPDATE SET "
            "views = EXCLUDED.views, comments = EXCLUDED.comments, "
            "mylists = EXCLUDED.mylists, likes = EXCLUDED.likes",
            rows,
        )
    conn.commit()


def collect_daily(conn, *, metric_date: "date | None" = None, get=None) -> dict:
    """The V2.2 entry point: one snapshot of every known nico video."""
    if get is None:
        get = make_getter(BASE_URL, min_interval=MIN_INTERVAL)
    metric_date = metric_date or date.today()
    targets = load_targets(conn)
    metrics = fetch_metrics(get, [vid for _, vid in targets])
    rows = [
        (sid, metric_date, m.views, m.comments, m.mylists, m.likes)
        for sid, vid in targets
        if (m := metrics.get(vid))
    ]
    upsert_metrics(conn, rows)
    missing = [vid for _, vid in targets if vid not in metrics]
    return {"requested": len(targets), "written": len(rows), "missing": missing}


if __name__ == "__main__":
    import psycopg

    from ..config import DATABASE_URL

    with psycopg.connect(DATABASE_URL) as conn:
        summary = collect_daily(conn)
        print({**summary, "missing": len(summary["missing"])})
