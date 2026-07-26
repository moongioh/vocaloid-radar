"""V3.2: song_embeddings load + similar-song query (plan 0001).

The embedding input is the song's "grammar": title + type + artist names +
tag names (canon name once confirmed, raw otherwise), joined deterministically
so an unchanged song produces byte-identical ``source_text``. That equality is
the idempotence check — re-runs embed only new or changed songs, so the weekly
cadence costs a handful of calls, not 2000.

Model = ``gw-embed`` (voyage-4-large via the gateway, 1024d — the V3.2 dim
finalization recorded in schema.sql). Transport mirrors canon.py/report.py:
``post(body) -> (status, json|None)`` is injected in tests; the real one is
lazy-httpx against the gateway's /v1/embeddings. Availability contract is the
same: 429/5xx backoff, ``GatewayUnavailable`` on exhaustion.

    python -m src.embeddings                 # backfill / refresh all songs
    python -m src.embeddings --similar 12345 # top-10 neighbours of a song
"""
import time
from typing import Callable, NamedTuple

from .canon import GatewayUnavailable
from .config import EMBED_MODEL, GATEWAY_API_KEY, GATEWAY_URL

BATCH_SIZE = 64
_ATTEMPTS = 3
_BACKOFF_BASE = 2.0  # seconds; doubles per retry

# HNSW search width. pgvector's default (40) measurably under-returns here: the
# corpus is one narrow slice of Vocaloid releases, so neighbours sit in a tight
# distance band (top-10 spanned 0.2302..0.2889 on the 2000-song backfill) and a
# narrow search settles into a local minimum — it missed the exact #1, #2 and a
# known cover. At 200 the index path matched the exact top-10 exactly.
EF_SEARCH = 200


class SourceRow(NamedTuple):
    song_id: int
    title: str
    song_type: "str | None"
    artists: "list[str]"
    tags: "list[str]"


# --------------------------------------------------------------- pure helpers

def build_source_text(row: SourceRow) -> str:
    """Deterministic embedding input — sorted lists, fixed field order."""
    return (
        f"{row.title}\n"
        f"type: {row.song_type or ''}\n"
        f"artists: {', '.join(sorted(row.artists))}\n"
        f"tags: {', '.join(sorted(row.tags))}"
    )


def select_stale(
    rows: "list[SourceRow]",
    existing: "dict[int, tuple[str | None, str | None]]",
    model: str = EMBED_MODEL,
) -> "list[tuple[SourceRow, str]]":
    """Rows whose (source_text, model) differ from what is stored."""
    out = []
    for row in rows:
        text = build_source_text(row)
        if existing.get(row.song_id) != (text, model):
            out.append((row, text))
    return out


def parse_embed_response(data: "dict | None", n: int) -> "list[list[float]] | None":
    """OpenAI-shape /v1/embeddings response -> n vectors in input order."""
    if not isinstance(data, dict) or not isinstance(data.get("data"), list):
        return None
    items = data["data"]
    if len(items) != n:
        return None
    vecs: "list[list[float] | None]" = [None] * n
    for item in items:
        if not isinstance(item, dict):
            return None
        idx, vec = item.get("index"), item.get("embedding")
        if not isinstance(idx, int) or not (0 <= idx < n) or not isinstance(vec, list):
            return None
        vecs[idx] = vec
    if any(v is None for v in vecs):
        return None
    return vecs  # type: ignore[return-value]


# ------------------------------------------------------- transport: the embedder

# post(body) -> (http_status, response_json_or_None). Injected in tests.
Post = Callable[[dict], "tuple[int, dict | None]"]


def _httpx_post(base_url: str, api_key: str) -> Post:
    import httpx  # lazy: keeps fixture tests stdlib-only

    client = httpx.Client(
        base_url=base_url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=60.0,
    )

    def post(body: dict) -> "tuple[int, dict | None]":
        r = client.post("/v1/embeddings", json=body)
        try:
            return r.status_code, r.json()
        except ValueError:
            return r.status_code, None

    return post


def make_embedder(
    base_url: "str | None" = None,
    api_key: "str | None" = None,
    *,
    model: str = EMBED_MODEL,
    post: "Post | None" = None,
    sleep: Callable[[float], None] = time.sleep,
):
    """embed(texts) -> vectors. Retries 429/5xx; a non-429 4xx or malformed
    response is a hard error (our request is wrong — retrying won't help)."""
    if post is None:
        post = _httpx_post(base_url or GATEWAY_URL, api_key or GATEWAY_API_KEY)

    def embed(texts: "list[str]") -> "list[list[float]]":
        body = {"model": model, "input": texts}
        for attempt in range(_ATTEMPTS):
            status, data = post(body)
            if status == 429 or status >= 500:
                if attempt < _ATTEMPTS - 1:
                    sleep(_BACKOFF_BASE * 2**attempt)
                continue
            if status != 200:
                raise RuntimeError(f"embeddings: HTTP {status}")
            vecs = parse_embed_response(data, len(texts))
            if vecs is None:
                raise RuntimeError("embeddings: malformed response")
            return vecs
        raise GatewayUnavailable(f"{model}: {_ATTEMPTS} attempts exhausted on 429/5xx")

    return embed


# ------------------------------------------------------------------ DB layer

_SOURCE_SQL = """
SELECT s.id, s.title, s.song_type,
       COALESCE((SELECT array_agg(DISTINCT a.name ORDER BY a.name)
                 FROM song_artists sa JOIN artists a ON a.id = sa.artist_id
                 WHERE sa.song_id = s.id), '{}'),
       COALESCE((SELECT array_agg(DISTINCT CASE
                     WHEN t.canon_status IN ('canon_lite', 'canon_flash')
                          AND t.canon_name IS NOT NULL THEN t.canon_name
                     ELSE t.name END)
                 FROM song_tags st JOIN tags t ON t.id = st.tag_id
                 WHERE st.song_id = s.id), '{}')
FROM songs s
"""


def load_source_rows(conn) -> "list[SourceRow]":
    with conn.cursor() as cur:
        cur.execute(_SOURCE_SQL)
        return [SourceRow(*r) for r in cur.fetchall()]


def load_existing(conn) -> "dict[int, tuple[str | None, str | None]]":
    with conn.cursor() as cur:
        cur.execute("SELECT song_id, source_text, model FROM song_embeddings")
        return {sid: (text, model) for sid, text, model in cur.fetchall()}


def upsert_embeddings(conn, items: "list[tuple[int, str, list[float]]]", model: str) -> None:
    """items = (song_id, source_text, vector)."""
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO song_embeddings (song_id, embedding, source_text, model, updated_at) "
            "VALUES (%s, %s::vector, %s, %s, now()) "
            "ON CONFLICT (song_id) DO UPDATE SET embedding = EXCLUDED.embedding, "
            "source_text = EXCLUDED.source_text, model = EXCLUDED.model, updated_at = now()",
            [
                (sid, "[" + ",".join(map(str, vec)) + "]", text, model)
                for sid, text, vec in items
            ],
        )
    conn.commit()


def run_embed(
    conn,
    *,
    embed: "Callable[[list[str]], list[list[float]]] | None" = None,
    model: str = EMBED_MODEL,
    batch_size: int = BATCH_SIZE,
) -> dict:
    """The V3.2 entry point: embed every new/changed song, batch-wise.

    Each batch commits on its own, so a mid-run gateway outage keeps the
    finished batches — the next run resumes from what is actually stored.
    """
    if embed is None:
        embed = make_embedder(model=model)
    rows = load_source_rows(conn)
    stale = select_stale(rows, load_existing(conn), model)
    done = 0
    for i in range(0, len(stale), batch_size):
        batch = stale[i : i + batch_size]
        vecs = embed([text for _, text in batch])
        upsert_embeddings(
            conn, [(row.song_id, text, vec) for (row, text), vec in zip(batch, vecs)], model
        )
        done += len(batch)
    return {"songs": len(rows), "stale": len(stale), "embedded": done}


def similar(
    conn, song_id: int, k: int = 10, *, ef_search: int = EF_SEARCH
) -> "list[tuple[int, str, float]]":
    """Top-k nearest songs by cosine distance (self excluded).

    The reference vector is fetched first and passed as a bound parameter on
    purpose: joining it in makes the ORDER BY expression reference another
    relation, and the HNSW index is then unusable (the planner falls back to a
    full scan + sort).

    ``SET LOCAL`` needs a transaction, which is psycopg's default; on an
    autocommit connection Postgres warns and the search falls back to the
    default width rather than failing silently.
    """
    with conn.cursor() as cur:
        cur.execute(f"SET LOCAL hnsw.ef_search = {int(ef_search)}")
        cur.execute("SELECT embedding FROM song_embeddings WHERE song_id = %s", (song_id,))
        row = cur.fetchone()
        if row is None:
            return []
        ref = row[0]
        cur.execute(
            "SELECT e.song_id, s.title, (e.embedding <=> %(ref)s)::float8 AS dist "
            "FROM song_embeddings e JOIN songs s ON s.id = e.song_id "
            "WHERE e.song_id <> %(self)s "
            "ORDER BY e.embedding <=> %(ref)s "
            "LIMIT %(k)s",
            {"ref": ref, "self": song_id, "k": k},
        )
        return list(cur.fetchall())


if __name__ == "__main__":
    import argparse

    import psycopg

    from .config import DATABASE_URL

    ap = argparse.ArgumentParser(description="Embed songs / query similar songs")
    ap.add_argument("--similar", type=int, help="song id: print top-k neighbours instead")
    ap.add_argument("-k", type=int, default=10)
    args = ap.parse_args()
    with psycopg.connect(DATABASE_URL) as conn:
        if args.similar is not None:
            for sid, title, dist in similar(conn, args.similar, args.k):
                print(f"{dist:.4f}  {sid}  {title}")
        else:
            print(run_embed(conn))
