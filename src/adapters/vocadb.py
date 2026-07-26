"""VocaDB adapter — seed songs / artists / tags / derived works (plan 0001 V2.1).

Layering mirrors canon.py: pure parsers, the shared paced transport
(adapters.transport, 1 req/sec courtesy + descriptive User-Agent as VocaDB
asks), a thin pagination orchestrator, and idempotent DB upserts.

Incremental = the same upserts re-run with a later --after date. Invariant:
nothing here ever writes the tags table's canon_* columns — re-seeding can
never clobber the tag_canon cache (plan 0002).

Derived works come for free from the listing itself: a song carrying
``originalVersionId`` IS a derived work of that original, so no per-song
/derived calls are needed. Links whose original lies outside the seeded set
are skipped (FK) and pick themselves up when a wider seed brings it in.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from .transport import make_getter

BASE_URL = "https://vocadb.net"
PAGE_SIZE = 50  # /api/songs maxResults cap
MIN_INTERVAL = 1.0  # seconds between requests (cache/courtesy per VocaDB API docs)


@dataclass(frozen=True)
class Song:
    id: int
    title: str
    publish_date: "date | None"
    song_type: "str | None"
    nico_video_id: "str | None"
    youtube_video_id: "str | None"


@dataclass(frozen=True)
class ArtistLink:
    artist_id: int
    name: str
    artist_type: "str | None"
    role: str


@dataclass(frozen=True)
class TagLink:
    tag_id: int
    name: str
    category: "str | None"


@dataclass(frozen=True)
class SongBundle:
    song: Song
    artists: "list[ArtistLink]" = field(default_factory=list)
    tags: "list[TagLink]" = field(default_factory=list)
    original_version_id: "int | None" = None


# --------------------------------------------------------------- pure: parsing

def _parse_date(value: "str | None") -> "date | None":
    if not value:
        return None
    return date.fromisoformat(value[:10])


def _pick_pv(pvs: "list[dict]", service: str) -> "str | None":
    """Prefer the Original upload; fall back to any PV of that service (reprint)."""
    candidates = [p for p in pvs if p.get("service") == service and p.get("pvId")]
    for p in candidates:
        if p.get("pvType") == "Original":
            return p["pvId"]
    return candidates[0]["pvId"] if candidates else None


def parse_song(item: dict) -> SongBundle:
    pvs = item.get("pvs") or []
    song = Song(
        id=item["id"],
        title=item["name"],
        publish_date=_parse_date(item.get("publishDate")),
        song_type=item.get("songType"),
        nico_video_id=_pick_pv(pvs, "NicoNicoDouga"),
        youtube_video_id=_pick_pv(pvs, "Youtube"),
    )
    artists = []
    for entry in item.get("artists") or []:
        artist = entry.get("artist")
        if not artist:  # free-text credit without a VocaDB artist record
            continue
        roles = entry.get("roles") or ""
        role = roles if roles not in ("", "Default") else (entry.get("categories") or "")
        artists.append(
            ArtistLink(
                artist_id=artist["id"],
                name=artist["name"],
                artist_type=artist.get("artistType"),
                role=role,
            )
        )
    tags = [
        TagLink(
            tag_id=t["tag"]["id"],
            name=t["tag"]["name"],
            category=t["tag"].get("categoryName"),
        )
        for t in item.get("tags") or []
        if t.get("tag")
    ]
    return SongBundle(
        song=song,
        artists=artists,
        tags=tags,
        original_version_id=item.get("originalVersionId") or None,
    )


def parse_page(data: dict) -> "tuple[list[SongBundle], int]":
    bundles = [parse_song(item) for item in data.get("items") or []]
    return bundles, data.get("totalCount") or 0


def page_params(after_date: str, start: int, max_results: int = PAGE_SIZE) -> dict:
    return {
        "sort": "PublishDate",
        "afterDate": after_date,
        "start": start,
        "maxResults": max_results,
        "getTotalCount": "true",
        "fields": "Artists,Tags,PVs",
        "onlyWithPvs": "true",  # metrics collection needs a video id
    }


# ------------------------------------------------------- thin: pagination

def fetch_recent(get, after_date: str, *, max_pages: int = 40) -> "tuple[list[SongBundle], int]":
    """Page through songs published after ``after_date`` (YYYY-MM-DD)."""
    bundles: list[SongBundle] = []
    total = 0
    for page_no in range(max_pages):
        data = get("/api/songs", page_params(after_date, page_no * PAGE_SIZE))
        page, total = parse_page(data)
        bundles.extend(page)
        if not page or (page_no + 1) * PAGE_SIZE >= total:
            break
    return bundles, total


# ----------------------------------------------------------------- db (thin)

def upsert_bundles(conn, bundles: "list[SongBundle]") -> "dict[str, int]":
    """Idempotent upserts. Duplicate ids across bundles collapse (last wins)."""
    songs: dict[int, tuple] = {}
    artists: dict[int, tuple] = {}
    tags: dict[int, tuple] = {}
    song_artists: set[tuple] = set()
    song_tags: set[tuple] = set()
    derived: dict[tuple, tuple] = {}
    for b in bundles:
        s = b.song
        songs[s.id] = (s.id, s.title, s.publish_date, s.song_type,
                       s.nico_video_id, s.youtube_video_id)
        for a in b.artists:
            artists[a.artist_id] = (a.artist_id, a.name, a.artist_type)
            song_artists.add((s.id, a.artist_id, a.role))
        for t in b.tags:
            tags[t.tag_id] = (t.tag_id, t.name, t.category)
            song_tags.add((s.id, t.tag_id))
        if b.original_version_id:
            derived[(b.original_version_id, s.id)] = (
                b.original_version_id, s.id, s.song_type, s.publish_date,
            )
    derived_rows: list[tuple] = []
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO songs (id, title, publish_date, song_type, nico_video_id, youtube_video_id) "
            "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (id) DO UPDATE SET "
            "title = EXCLUDED.title, publish_date = EXCLUDED.publish_date, "
            "song_type = EXCLUDED.song_type, nico_video_id = EXCLUDED.nico_video_id, "
            "youtube_video_id = EXCLUDED.youtube_video_id, updated_at = now()",
            list(songs.values()),
        )
        cur.executemany(
            "INSERT INTO artists (id, name, artist_type) VALUES (%s, %s, %s) "
            "ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name, "
            "artist_type = EXCLUDED.artist_type, updated_at = now()",
            list(artists.values()),
        )
        # canon_* columns are the tag_canon cache (plan 0002) — never written here.
        cur.executemany(
            "INSERT INTO tags (id, name, category) VALUES (%s, %s, %s) "
            "ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name, category = EXCLUDED.category",
            list(tags.values()),
        )
        cur.executemany(
            "INSERT INTO song_artists (song_id, artist_id, role) VALUES (%s, %s, %s) "
            "ON CONFLICT DO NOTHING",
            sorted(song_artists),
        )
        cur.executemany(
            "INSERT INTO song_tags (song_id, tag_id) VALUES (%s, %s) "
            "ON CONFLICT DO NOTHING",
            sorted(song_tags),
        )
        if derived:
            cur.execute(
                "SELECT id FROM songs WHERE id = ANY(%s)",
                ([orig for (orig, _) in derived],),
            )
            present = {r[0] for r in cur.fetchall()}
            derived_rows = [row for (orig, _), row in derived.items() if orig in present]
            # ON CONFLICT DO NOTHING keeps the first discovered_at (deriv_velocity input).
            cur.executemany(
                "INSERT INTO derived_works (original_song_id, derived_song_id, relation_type, derived_publish_date) "
                "VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
                derived_rows,
            )
    conn.commit()
    return {
        "songs": len(songs),
        "artists": len(artists),
        "tags": len(tags),
        "song_artists": len(song_artists),
        "song_tags": len(song_tags),
        "derived_linked": len(derived_rows),
        "derived_seen": len(derived),
    }


def seed(conn, *, after_date: str, get=None, max_pages: int = 40) -> "dict[str, int]":
    """The V2.1 entry point: fetch recent songs and upsert everything."""
    if get is None:
        get = make_getter(BASE_URL, min_interval=MIN_INTERVAL)
    bundles, total = fetch_recent(get, after_date, max_pages=max_pages)
    summary = upsert_bundles(conn, bundles)
    summary["fetched"] = len(bundles)
    summary["total_matching"] = total
    return summary


if __name__ == "__main__":
    import argparse

    import psycopg

    from ..config import DATABASE_URL

    ap = argparse.ArgumentParser(description="Seed songs from VocaDB")
    ap.add_argument("--after", required=True, help="publish-date floor, YYYY-MM-DD")
    ap.add_argument("--max-pages", type=int, default=40)
    args = ap.parse_args()
    with psycopg.connect(DATABASE_URL) as conn:
        print(seed(conn, after_date=args.after, max_pages=args.max_pages))
