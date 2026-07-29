"""Tiered LLM routing helper — the gate + escalation ladder (plan 0002 R1).

Pure logic. The actual gateway HTTP call is *injected* as ``caller``, so the whole
policy is unit-testable without a live LLM (tests/test_routing.py mirrors the
plan's 명세-예시 table row by row).

Policy (plan 0002, approved 2026-07-18):
  - A task starts on its default tier; tag_canon starts on ``gw-lite``.
  - Code gates decide pass vs. escalate. **Structure is the primary signal**
    (id echo, output shape, canon-vocabulary membership); self-reported
    ``confidence`` is only a secondary hint — LLMs are not calibrated, so a
    confident wrong answer must not ride through on confidence alone.
  - Failing items escalate ``gw-lite`` -> ``gw-flash``. The ladder STOPS at
    flash: it never calls a paid model on its own (plan 0001 "자동 폴백 금지").
  - Escalation is capped per run so a bad lite day can't drain the flash quota
    that ``report_gen`` also depends on (plan A3). Over-cap items stay ``pending``
    for the next run; a flash that also fails the gate is ``demoted``.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

# Logical model names as configured in the gateway's config.yaml (plan 0001).
# LITE is the tier-1 slot, not a claim about which model fills it: it moved from
# gw-lite to gw-gemma on 2026-07-29 (plan 0003 E2). gw-lite was mis-classifying
# with high self-reported confidence, which is exactly the failure the confidence
# threshold below cannot catch — see the G4 note in RoutingConfig.
# The Status values stay ``canon_lite``/``canon_flash``: they are persisted in
# tags.canon_status, so renaming them would be a data migration for no gain.
LITE = "gw-gemma"
FLASH = "gw-flash"


@dataclass(frozen=True)
class TagInput:
    """One raw tag handed to canonicalization (VocaDB entity id + raw name)."""

    id: int
    name: str


class Status(str, Enum):
    """Terminal routing outcome — mirrors ``tags.canon_status`` (schema V1.1)."""

    CANON_LITE = "canon_lite"    # accepted on gw-lite
    CANON_FLASH = "canon_flash"  # accepted after escalation to gw-flash
    DEMOTED = "demoted"          # flash also failed the gate; raw preserved, retry next run
    PENDING = "pending"          # deferred: escalation cap reached, not yet attempted on flash


@dataclass
class TagResult:
    id: int
    status: Status
    canon_name: str | None = None
    confidence: float | None = None
    registers_vocab: bool = False  # a newly *confirmed* canon term to add to the allowlist
    lite_canon: str | None = None  # the lite proposal on an escalated item — the free
    #                                half of the 승급쌍 (escalation pair) the FinOps rollup
    #                                (R4) compares against the flash result; None unless the
    #                                item escalated with a parseable lite canonical.


# (model_name, items) -> the parsed per-item response dicts, or None when the
# batch response could not be parsed at all (e.g. the JSON broke). Each dict is
# one item's raw fields: {"id", "canonical", "confidence", "is_new"?}.
Caller = Callable[[str, "list[TagInput]"], "list[dict] | None"]


@dataclass
class RoutingConfig:
    confidence_threshold: float = 0.80
    # Max items escalated to flash per run. With batched flash calls the item
    # count is the meaningful load knob that protects report_gen's shared flash
    # quota; anything beyond it defers to `pending` rather than being demoted.
    #
    # 20 -> 100 on 2026-07-29 (plan 0004). 20 was set before we had a corpus to
    # size it against, and it turned out to be a convergence stopper: escalations
    # are how NEW canon terms enter the vocabulary, so a 20-item run budget caps
    # vocabulary growth at 20 terms/run. Against ~900 unclassified tags that is
    # ~45 weekly runs. The quota it protects is unaffected at 100 — escalated
    # items go out BATCHED (one flash call per batch, not per item), so 100 items
    # is ~5 flash calls in a run that happens once a week, against
    # finops.DAILY_QUOTA[FLASH] = 250 per day.
    escalation_cap: int = 100


class _Gate(Enum):
    PASS = "pass"
    ESCALATE = "escalate"


def _ids_ok(raw: "list[dict] | None", items: "list[TagInput]") -> bool:
    """G1 — the response echoes *exactly* the requested id set.

    Batching maps results to inputs by id, never by position: a model that drops
    or reorders an item would otherwise bind an answer to the wrong tag. A set
    mismatch (or an unparseable batch) fails the whole batch.
    """
    if raw is None:
        return False
    try:
        got = {r["id"] for r in raw}
    except (TypeError, KeyError):
        return False
    return got == {it.id for it in items}


def _gate_lite(resp: dict, vocab: "set[str]", threshold: float) -> _Gate:
    """G2/G3/G4 on a lite response. Any miss escalates the item to flash."""
    canon = resp.get("canonical")
    if not isinstance(canon, str) or not canon.strip():
        return _Gate.ESCALATE  # G2: broken/empty structure
    if bool(resp.get("is_new", False)):
        return _Gate.ESCALATE  # G3: a new-vocabulary proposal — lite may not finalize it
    if canon not in vocab:
        return _Gate.ESCALATE  # G3: not in the closed-loop allowlist
    conf = resp.get("confidence")
    if not isinstance(conf, (int, float)) or conf < threshold:
        return _Gate.ESCALATE  # G4: secondary confidence hint
    return _Gate.PASS


def _accept_flash(item_id: int, resp: dict, threshold: float) -> "TagResult | None":
    """Flash is the confirmer: accept on valid structure + confidence, else demote.

    Flash is authoritative on vocabulary, so an ``is_new`` proposal it returns is
    a confirmed new canon term to register — membership is not re-checked here.
    """
    canon = resp.get("canonical")
    if not isinstance(canon, str) or not canon.strip():
        return None
    conf = resp.get("confidence")
    if not isinstance(conf, (int, float)) or conf < threshold:
        return None
    return TagResult(
        id=item_id,
        status=Status.CANON_FLASH,
        canon_name=canon,
        confidence=float(conf),
        registers_vocab=bool(resp.get("is_new", False)),
    )


def call_tiered(
    items: "list[TagInput]",
    caller: Caller,
    vocab: "set[str]",
    config: "RoutingConfig | None" = None,
) -> "list[TagResult]":
    """Route ``items`` through the lite→flash gate ladder. Input order preserved.

    ``items`` are assumed to be cache-misses only (the caching layer, R2, filters
    already-canon tags out before this is called).
    """
    config = config or RoutingConfig()
    by_id = {it.id: it for it in items}
    results: dict[int, TagResult] = {}

    # --- lite batch, with one batch-level retry on a G1 failure ---
    # A broken batch is retried on lite first; we never buy a lite hiccup with
    # flash quota.
    raw = caller(LITE, items)
    if not _ids_ok(raw, items):
        raw = caller(LITE, items)

    # lite_canon[id] = what lite proposed for an escalated item, when it was a
    # usable string. It is the free half of the 승급쌍 (R4 agreement rate); a
    # batch-level failure (raw broken) leaves it empty for every item.
    lite_canon: dict[int, str] = {}
    if not _ids_ok(raw, items):
        to_escalate = list(items)  # batch still broken -> every item is a candidate
    else:
        assert raw is not None  # _ids_ok guarantees this
        to_escalate = []
        for r in raw:
            item = by_id[r["id"]]
            if _gate_lite(r, vocab, config.confidence_threshold) is _Gate.PASS:
                results[item.id] = TagResult(
                    id=item.id,
                    status=Status.CANON_LITE,
                    canon_name=r["canonical"],
                    confidence=float(r["confidence"]),
                )
            else:
                to_escalate.append(item)
                canon = r.get("canonical")
                if isinstance(canon, str) and canon.strip():
                    lite_canon[item.id] = canon

    # --- flash escalation, capped per run ---
    escalated = to_escalate[: config.escalation_cap]
    for item in to_escalate[config.escalation_cap :]:
        results[item.id] = TagResult(id=item.id, status=Status.PENDING)

    if escalated:
        fraw = caller(FLASH, escalated)
        fraw_by_id = {r["id"]: r for r in fraw} if _ids_ok(fraw, escalated) else {}
        for item in escalated:
            accepted = (
                _accept_flash(item.id, fraw_by_id[item.id], config.confidence_threshold)
                if item.id in fraw_by_id
                else None
            )
            result = accepted or TagResult(id=item.id, status=Status.DEMOTED)
            result.lite_canon = lite_canon.get(item.id)
            results[item.id] = result

    return [results[it.id] for it in items]
