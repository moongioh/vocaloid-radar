"""Spec-by-example tests for the trend pure functions (plan 0001 V3.3).

Each test mirrors one row of the plan's '트렌드 스코어 — 명세 예시' table.
"""
from src.analysis.trend import (
    detect_clusters,
    deriv_velocity,
    is_coldstart,
    tag_share_delta,
    view_velocity,
)


def test_view_velocity():
    # 지난주 +10k, 이번주 +30k → 3.0
    assert view_velocity(10_000, 30_000) == 3.0


def test_view_velocity_no_baseline():
    # 지난주 증가 0 → 나눌 기준이 없음 → None (콜드스타트로 처리)
    assert view_velocity(0, 5_000) is None


def test_deriv_velocity():
    # 4주 평균 2건, 이번주 8건 → 4.0
    assert deriv_velocity([2, 2, 2, 2], 8) == 4.0


def test_tag_share_delta():
    # 태그A 점유율 5% → 9% → +4.0pp
    assert tag_share_delta(5.0, 9.0) == 4.0


def test_detect_clusters():
    # 상위 10곡 중 6곡이 같은 태그쌍 {rock, ballad} → 클러스터 1건
    top = [(i, {"rock", "ballad"}) for i in range(6)]
    top += [(i, {f"misc{i}", f"other{i}"}) for i in range(6, 10)]
    clusters = detect_clusters(top, min_songs=6)
    assert len(clusters) == 1
    assert clusters[0]["tags"] == frozenset({"rock", "ballad"})
    assert len(clusters[0]["song_ids"]) == 6


def test_is_coldstart():
    # 게시 5일 곡(≈0주 관측) → True; 3주 관측 곡 → False
    assert is_coldstart(0) is True
    assert is_coldstart(3) is False
