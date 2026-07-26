"""Fixture tests for the V3.2 embedding layer — pure logic, no DB/network."""
import pytest

from src.canon import GatewayUnavailable
from src.embeddings import (
    SourceRow,
    build_source_text,
    make_embedder,
    parse_embed_response,
    select_stale,
)


def row(song_id=1, title="夜行", song_type="Original", artists=("wowaka",), tags=("ロック",)):
    return SourceRow(song_id, title, song_type, list(artists), list(tags))


def test_build_source_text_deterministic_sorting():
    a = build_source_text(row(artists=["b", "a"], tags=["y", "x"]))
    b = build_source_text(row(artists=["a", "b"], tags=["x", "y"]))
    assert a == b
    assert a == "夜行\ntype: Original\nartists: a, b\ntags: x, y"


def test_build_source_text_none_type():
    assert "type: \n" in build_source_text(row(song_type=None))


def test_select_stale_new_changed_model_unchanged():
    r1, r2, r3 = row(1), row(2, title="t2"), row(3, title="t3")
    existing = {
        1: (build_source_text(r1), "gw-embed"),          # unchanged -> skip
        2: ("old text", "gw-embed"),                      # text changed
        3: (build_source_text(r3), "other-model"),        # model changed
    }                                                     # 4 = absent -> new
    stale = select_stale([r1, r2, r3, row(4)], existing, "gw-embed")
    assert [r.song_id for r, _ in stale] == [2, 3, 4]


def test_parse_embed_response_orders_by_index():
    data = {"data": [{"index": 1, "embedding": [2.0]}, {"index": 0, "embedding": [1.0]}]}
    assert parse_embed_response(data, 2) == [[1.0], [2.0]]


@pytest.mark.parametrize(
    "data",
    [
        None,
        {},
        {"data": [{"index": 0, "embedding": [1.0]}]},                       # count mismatch (n=2)
        {"data": [{"index": 0, "embedding": [1.0]}, {"index": 0, "embedding": [2.0]}]},  # dup index
        {"data": [{"index": 0, "embedding": "bad"}, {"index": 1, "embedding": [2.0]}]},
    ],
)
def test_parse_embed_response_rejects_malformed(data):
    assert parse_embed_response(data, 2) is None


def test_embedder_retries_429_then_succeeds():
    calls = []

    def post(body):
        calls.append(body)
        if len(calls) == 1:
            return 429, None
        return 200, {"data": [{"index": 0, "embedding": [0.5]}]}

    embed = make_embedder(post=post, sleep=lambda s: None, model="gw-embed")
    assert embed(["x"]) == [[0.5]]
    assert len(calls) == 2


def test_embedder_hard_error_on_4xx():
    embed = make_embedder(post=lambda b: (400, None), sleep=lambda s: None)
    with pytest.raises(RuntimeError):
        embed(["x"])


def test_embedder_exhaustion_raises_gateway_unavailable():
    embed = make_embedder(post=lambda b: (503, None), sleep=lambda s: None)
    with pytest.raises(GatewayUnavailable):
        embed(["x"])


# ---------------------------------------------------- run_embed orchestration
# DB I/O is monkeypatched out; the real SQL is covered by test_embeddings_db.py.
# Kept out of the DB suite on purpose: driving run_embed against the shared dev
# DB would rewrite live-seeded rows (the 07-26 isolation-incident class).

import src.embeddings as embeddings  # noqa: E402


def _wire(monkeypatch, rows, existing, stored):
    monkeypatch.setattr(embeddings, "load_source_rows", lambda c: rows)
    monkeypatch.setattr(embeddings, "load_existing", lambda c: existing)
    monkeypatch.setattr(
        embeddings, "upsert_embeddings", lambda c, items, m: stored.extend(items)
    )


def test_run_embed_batches_and_summary(monkeypatch):
    rows = [row(i, title=f"t{i}") for i in range(5)]
    stored, batches = [], []

    def embed(texts):
        batches.append(len(texts))
        return [[0.0]] * len(texts)

    _wire(monkeypatch, rows, {}, stored)
    summary = embeddings.run_embed(None, embed=embed, batch_size=2)
    assert batches == [2, 2, 1]
    assert summary == {"songs": 5, "stale": 5, "embedded": 5}
    assert len(stored) == 5


def test_run_embed_mid_run_failure_keeps_finished_batches(monkeypatch):
    rows = [row(i, title=f"t{i}") for i in range(4)]
    stored, calls = [], []

    def embed(texts):
        calls.append(texts)
        if len(calls) == 2:
            raise GatewayUnavailable("boom")
        return [[0.0]] * len(texts)

    _wire(monkeypatch, rows, {}, stored)
    with pytest.raises(GatewayUnavailable):
        embeddings.run_embed(None, embed=embed, batch_size=2)
    assert len(stored) == 2  # batch 1 persisted; resume embeds only the rest
