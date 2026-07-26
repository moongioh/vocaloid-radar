"""Fixture tests for the Niconico snapshot adapter (plan 0001 V2.2).

Request building, chunking, and response parsing — no network, no DB.
"""
import json

from src.adapters.niconico import (
    BATCH_SIZE,
    Metric,
    build_params,
    chunked,
    fetch_metrics,
    parse_response,
)


def test_build_params_json_filter():
    p = build_params(["sm100", "sm200"])
    f = json.loads(p["jsonFilter"])
    assert f["type"] == "or"
    assert [(x["field"], x["value"]) for x in f["filters"]] == [
        ("contentId", "sm100"), ("contentId", "sm200"),
    ]
    assert p["_limit"] == 2
    assert p["_sort"]  # the API rejects requests without a sort
    assert "viewCounter" in p["fields"] and "likeCounter" in p["fields"]


def test_chunked_splits_at_batch_size():
    ids = [f"sm{i}" for i in range(BATCH_SIZE + 5)]
    chunks = chunked(ids)
    assert [len(c) for c in chunks] == [BATCH_SIZE, 5]
    assert chunked([]) == []


def test_parse_response_maps_by_content_id():
    data = {
        "meta": {"status": 200},
        "data": [
            {"contentId": "sm100", "viewCounter": 5000, "commentCounter": 30,
             "mylistCounter": 12, "likeCounter": 400},
            {"contentId": "sm200", "viewCounter": 77},  # partial fields tolerated
            {"viewCounter": 1},  # no contentId -> dropped
        ],
    }
    m = parse_response(data)
    assert m["sm100"] == Metric("sm100", 5000, 30, 12, 400)
    assert m["sm200"].views == 77 and m["sm200"].likes is None
    assert set(m) == {"sm100", "sm200"}


def test_fetch_metrics_merges_batches():
    ids = [f"sm{i}" for i in range(BATCH_SIZE + 2)]
    seen_limits = []

    def get(path, params):
        f = json.loads(params["jsonFilter"])
        seen_limits.append(params["_limit"])
        return {"data": [{"contentId": x["value"], "viewCounter": 1}
                         for x in f["filters"]]}

    m = fetch_metrics(get, ids)
    assert len(m) == len(ids)
    assert seen_limits == [BATCH_SIZE, 2]


def test_fetch_metrics_missing_ids_absent():
    def get(path, params):
        return {"data": [{"contentId": "sm1", "viewCounter": 9}]}

    m = fetch_metrics(get, ["sm1", "smGONE"])
    assert "smGONE" not in m and m["sm1"].views == 9
