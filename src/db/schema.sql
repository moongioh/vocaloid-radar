-- vocaloid_radar schema (plan 0001 V1.1). Idempotent: safe to re-run.
-- Target: the shared `vocaloid` DB on the llm_gateway platform (PG16 + pgvector).
-- Locally this runs against docker-compose.dev.yml's throwaway PG.
--
-- The plan lists 8 conceptual groups; normalized that is 10 tables (song_artists
-- and song_tags are the many-to-many junctions).

CREATE EXTENSION IF NOT EXISTS vector;

-- 1. songs — master record. Natural key = VocaDB song id.
CREATE TABLE IF NOT EXISTS songs (
    id               BIGINT PRIMARY KEY,
    title            TEXT NOT NULL,
    publish_date     DATE,
    song_type        TEXT,
    nico_video_id    TEXT,
    youtube_video_id TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS songs_publish_date_idx ON songs (publish_date);
CREATE INDEX IF NOT EXISTS songs_nico_video_id_idx ON songs (nico_video_id);

-- 2. artists — producers, vocals (characters). Natural key = VocaDB artist id.
CREATE TABLE IF NOT EXISTS artists (
    id          BIGINT PRIMARY KEY,
    name        TEXT NOT NULL,
    artist_type TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 3. song_artists — many-to-many with role (composer, vocalist, ...).
CREATE TABLE IF NOT EXISTS song_artists (
    song_id   BIGINT NOT NULL REFERENCES songs (id) ON DELETE CASCADE,
    artist_id BIGINT NOT NULL REFERENCES artists (id) ON DELETE CASCADE,
    role      TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (song_id, artist_id, role)
);
CREATE INDEX IF NOT EXISTS song_artists_artist_idx ON song_artists (artist_id);

-- 4. tags — raw VocaDB tag plus LLM canon columns. Raw is preserved; V3.1 fills
--    canon_name / canon_confidence and demotes low-confidence tags via canon_status.
CREATE TABLE IF NOT EXISTS tags (
    id               BIGINT PRIMARY KEY,
    name             TEXT NOT NULL,
    category         TEXT,
    canon_name       TEXT,
    canon_confidence REAL,
    canon_status     TEXT NOT NULL DEFAULT 'pending',
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS tags_canon_name_idx ON tags (canon_name);
CREATE INDEX IF NOT EXISTS tags_canon_status_idx ON tags (canon_status);

-- 5. song_tags — many-to-many (raw VocaDB tagging).
CREATE TABLE IF NOT EXISTS song_tags (
    song_id BIGINT NOT NULL REFERENCES songs (id) ON DELETE CASCADE,
    tag_id  BIGINT NOT NULL REFERENCES tags (id) ON DELETE CASCADE,
    PRIMARY KEY (song_id, tag_id)
);
CREATE INDEX IF NOT EXISTS song_tags_tag_idx ON song_tags (tag_id);

-- 6. derived_works — utaite covers / remixes linked to the original song.
--    discovered_at drives deriv_velocity (new derived works per week).
CREATE TABLE IF NOT EXISTS derived_works (
    original_song_id     BIGINT NOT NULL REFERENCES songs (id) ON DELETE CASCADE,
    derived_song_id      BIGINT NOT NULL,
    relation_type        TEXT,
    derived_publish_date DATE,
    discovered_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (original_song_id, derived_song_id)
);
CREATE INDEX IF NOT EXISTS derived_works_pubdate_idx ON derived_works (derived_publish_date);

-- 7. metrics_daily — the un-backfillable time-series canon.
--    (song_id, metric_date, source) PK blocks duplicate daily rows (V2.3 verify).
CREATE TABLE IF NOT EXISTS metrics_daily (
    song_id     BIGINT NOT NULL REFERENCES songs (id) ON DELETE CASCADE,
    metric_date DATE NOT NULL,
    source      TEXT NOT NULL,
    views       BIGINT,
    comments    BIGINT,
    mylists     BIGINT,
    likes       BIGINT,
    PRIMARY KEY (song_id, metric_date, source)
);
CREATE INDEX IF NOT EXISTS metrics_daily_date_idx ON metrics_daily (metric_date);

-- 8. trend_scores — weekly batch output, per song per ISO week.
--    tag_share_delta is tag-level, not song-level, so it lives in the
--    weekly_reports evidence JSON rather than here.
CREATE TABLE IF NOT EXISTS trend_scores (
    song_id        BIGINT NOT NULL REFERENCES songs (id) ON DELETE CASCADE,
    week           DATE NOT NULL,
    view_velocity  REAL,
    deriv_velocity REAL,
    is_coldstart   BOOLEAN NOT NULL DEFAULT FALSE,
    cluster_id     INTEGER,
    computed_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (song_id, week)
);
CREATE INDEX IF NOT EXISTS trend_scores_week_velocity_idx ON trend_scores (week, view_velocity DESC);
CREATE INDEX IF NOT EXISTS trend_scores_week_cluster_idx ON trend_scores (week, cluster_id);

-- 9. weekly_reports — LLM narrative + evidence snapshot (top songs, tag deltas,
--    clusters, watchlist) captured as JSON for reproducibility.
CREATE TABLE IF NOT EXISTS weekly_reports (
    week         DATE PRIMARY KEY,
    narrative    TEXT,
    evidence     JSONB,
    model        TEXT,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 10. song_embeddings — pgvector. dim is tied to the embedding model
--     (text-embedding-004 = 768) and finalized at V3.2. The HNSW index is
--     deferred until the model and data are locked.
CREATE TABLE IF NOT EXISTS song_embeddings (
    song_id     BIGINT PRIMARY KEY REFERENCES songs (id) ON DELETE CASCADE,
    embedding   vector(768),
    source_text TEXT,
    model       TEXT,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
