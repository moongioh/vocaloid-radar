"""Fixture tests for the VocaDB adapter parsers + pagination (plan 0001 V2.1).

Fixtures mirror the real /api/songs shape (fields=Artists,Tags,PVs). No
network, no DB — upsert semantics are covered by test_adapters_db.py.
"""
from src.adapters.vocadb import (
    PAGE_SIZE,
    fetch_recent,
    page_params,
    parse_page,
    parse_song,
)

FULL_ITEM = {
    "id": 831244,
    "name": "テスト曲",
    "publishDate": "2026-06-01T00:00:00Z",
    "songType": "Original",
    "artists": [
        {"artist": {"id": 11, "name": "某P", "artistType": "Producer"},
         "categories": "Producer", "roles": "Default"},
        {"artist": {"id": 22, "name": "初音ミク", "artistType": "Vocaloid"},
         "categories": "Vocalist", "roles": "Default"},
        {"artist": {"id": 33, "name": "絵師X", "artistType": "Illustrator"},
         "categories": "Other", "roles": "Illustrator"},
        {"name": "無名クレジット", "categories": "Producer", "roles": "Default"},
    ],
    "tags": [
        {"count": 5, "tag": {"id": 481, "name": "ロック", "categoryName": "Genres"}},
        {"count": 2, "tag": {"id": 502, "name": "電波ソング", "categoryName": "Genres"}},
    ],
    "pvs": [
        {"service": "NicoNicoDouga", "pvId": "sm45000001", "pvType": "Reprint"},
        {"service": "NicoNicoDouga", "pvId": "sm45000000", "pvType": "Original"},
        {"service": "Youtube", "pvId": "ytAbc123", "pvType": "Original"},
    ],
}


def test_parse_song_full():
    b = parse_song(FULL_ITEM)
    s = b.song
    assert (s.id, s.title, s.song_type) == (831244, "テスト曲", "Original")
    assert str(s.publish_date) == "2026-06-01"
    assert s.nico_video_id == "sm45000000"  # Original preferred over Reprint
    assert s.youtube_video_id == "ytAbc123"
    assert b.original_version_id is None


def test_parse_song_artists_roles_and_null_artist():
    b = parse_song(FULL_ITEM)
    by_id = {a.artist_id: a for a in b.artists}
    assert set(by_id) == {11, 22, 33}  # free-text credit (no artist record) skipped
    assert by_id[11].role == "Producer"       # roles=Default -> categories
    assert by_id[33].role == "Illustrator"    # explicit roles win
    assert by_id[22].artist_type == "Vocaloid"


def test_parse_song_tags():
    b = parse_song(FULL_ITEM)
    assert [(t.tag_id, t.name, t.category) for t in b.tags] == [
        (481, "ロック", "Genres"),
        (502, "電波ソング", "Genres"),
    ]


def test_parse_song_minimal():
    b = parse_song({"id": 1, "name": "min"})
    assert b.song.publish_date is None
    assert b.song.nico_video_id is None and b.song.youtube_video_id is None
    assert b.artists == [] and b.tags == []


def test_parse_song_reprint_fallback_and_derived():
    item = {
        "id": 900,
        "name": "歌ってみた",
        "songType": "Cover",
        "originalVersionId": 831244,
        "pvs": [{"service": "NicoNicoDouga", "pvId": "sm9", "pvType": "Reprint"}],
    }
    b = parse_song(item)
    assert b.song.nico_video_id == "sm9"  # no Original PV -> reprint fallback
    assert b.original_version_id == 831244


def test_parse_page_and_total():
    bundles, total = parse_page({"items": [FULL_ITEM], "totalCount": 1234})
    assert len(bundles) == 1 and total == 1234
    assert parse_page({}) == ([], 0)


def test_page_params_shape():
    p = page_params("2026-06-01", 100)
    assert p["afterDate"] == "2026-06-01"
    assert p["start"] == 100 and p["maxResults"] == PAGE_SIZE
    assert "Artists" in p["fields"] and "PVs" in p["fields"]


def test_fetch_recent_paginates_until_total():
    def item(i):
        return {"id": i, "name": f"s{i}"}

    pages = {
        0: {"items": [item(i) for i in range(PAGE_SIZE)], "totalCount": PAGE_SIZE + 3},
        PAGE_SIZE: {"items": [item(PAGE_SIZE + i) for i in range(3)],
                    "totalCount": PAGE_SIZE + 3},
    }
    starts = []

    def get(path, params):
        starts.append(params["start"])
        return pages[params["start"]]

    bundles, total = fetch_recent(get, "2026-06-01")
    assert len(bundles) == PAGE_SIZE + 3
    assert total == PAGE_SIZE + 3
    assert starts == [0, PAGE_SIZE]  # stopped after covering totalCount


def test_fetch_recent_respects_max_pages():
    def get(path, params):
        return {"items": [{"id": params["start"], "name": "x"}], "totalCount": 10_000}

    # 1-item pages with a huge total: max_pages is the only brake
    bundles, _ = fetch_recent(get, "2026-06-01", max_pages=3)
    assert len(bundles) == 3
