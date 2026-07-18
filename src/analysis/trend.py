"""Trend-scoring pure functions (plan 0001 V3.3).

All functions here are pure — no I/O, no DB — so they are unit-testable in
isolation (see tests/test_trend.py). The LangGraph weekly batch (V3.4) calls
these after fetching the weekly aggregates from the DB.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations


def view_velocity(prev_week_delta: float, this_week_delta: float) -> float | None:
    """This week's view gain relative to last week's.

    3.0 means views grew three times as fast this week. Returns None when there
    is no positive baseline to divide by (a brand-new song) — such songs are
    handled by the cold-start guard and surfaced on the '신곡 워치' list instead.
    """
    if prev_week_delta <= 0:
        return None
    return this_week_delta / prev_week_delta


def deriv_velocity(prev_4wk_counts: list[int], this_week_count: int) -> float | None:
    """New derived works this week vs. the average of the prior 4 weeks.

    4.0 means four times the usual rate of utaite covers / remixes appearing.
    """
    if not prev_4wk_counts:
        return None
    avg = sum(prev_4wk_counts) / len(prev_4wk_counts)
    if avg == 0:
        return None
    return this_week_count / avg


def tag_share(tag_count: int, total_count: int) -> float:
    """A tag's share of this week's new songs, in percent."""
    if total_count == 0:
        return 0.0
    return 100.0 * tag_count / total_count


def tag_share_delta(prev_share_pct: float, this_share_pct: float) -> float:
    """Week-over-week change in a tag's share, in percentage points."""
    return this_share_pct - prev_share_pct


def detect_clusters(
    top_songs: list[tuple[int, set[str]]],
    min_songs: int = 6,
    pair_size: int = 2,
) -> list[dict]:
    """Group the top-velocity songs by shared tag combinations.

    A cluster is a tag combination (default: a pair) carried by at least
    `min_songs` of the top songs. Returns one dict per cluster:
        {"tags": frozenset(...), "song_ids": [...]}.
    """
    combo_to_songs: dict[frozenset[str], list[int]] = defaultdict(list)
    for song_id, tags in top_songs:
        for combo in combinations(sorted(tags), pair_size):
            combo_to_songs[frozenset(combo)].append(song_id)
    return [
        {"tags": combo, "song_ids": song_ids}
        for combo, song_ids in combo_to_songs.items()
        if len(song_ids) >= min_songs
    ]


def is_coldstart(weeks_of_history: int, min_weeks: int = 2) -> bool:
    """True if a song has too little time-series to score a velocity yet.

    Cold-start songs are excluded from trend_scores and listed on the
    '신곡 워치' watchlist instead.
    """
    return weeks_of_history < min_weeks
