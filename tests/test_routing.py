"""Spec-by-example tests for the tiered routing helper (plan 0002 R1).

Each test mirrors one row of the plan's '명세 — 예시 (검증 게이트·승급 수용 기준)' table.
No live LLM: the gateway call is injected as a fake ``caller`` returning canned
responses, so the gate + escalation ladder is exercised in full isolation.
"""
from src.routing import (
    FLASH,
    LITE,
    RoutingConfig,
    Status,
    TagInput,
    call_tiered,
)

VOCAB = {"初音ミク", "ロック"}


def caller_from(mapping, *, lite_first_batch_none=False):
    """Build a fake caller from {(model, tag_id): response_dict}.

    Records the sequence of models called on ``.calls`` so tests can assert that
    flash was (or was not) touched. ``lite_first_batch_none`` makes the *first*
    lite batch return None (unparseable) to exercise the G1 retry path.
    """
    calls = []

    def _caller(model, items):
        calls.append(model)
        if model == LITE and lite_first_batch_none and calls.count(LITE) == 1:
            return None
        return [mapping[(model, it.id)] for it in items if (model, it.id) in mapping]

    _caller.calls = calls
    return _caller


def test_lite_pass_in_vocab():
    # 표준/변형 태그가 어휘 내 + 신뢰도 충분 → lite 통과 → canon_lite
    items = [TagInput(1, "初音ミク"), TagInput(2, "みくみく")]
    caller = caller_from({
        (LITE, 1): {"id": 1, "canonical": "初音ミク", "confidence": 0.98},
        (LITE, 2): {"id": 2, "canonical": "初音ミク", "confidence": 0.93},
    })
    out = call_tiered(items, caller, VOCAB)
    assert [r.status for r in out] == [Status.CANON_LITE, Status.CANON_LITE]
    assert out[0].canon_name == "初音ミク"
    assert FLASH not in caller.calls


def test_g1_id_mismatch_retries_lite_not_flash():
    # 응답 id 집합 ≠ 요청 id 집합(첫 배치 malformed) → lite 배치 1회 재시도로 회복
    items = [TagInput(1, "初音ミク")]
    caller = caller_from(
        {(LITE, 1): {"id": 1, "canonical": "初音ミク", "confidence": 0.98}},
        lite_first_batch_none=True,
    )
    out = call_tiered(items, caller, VOCAB)
    assert out[0].status == Status.CANON_LITE
    assert caller.calls.count(LITE) == 2  # tried lite twice
    assert FLASH not in caller.calls      # never bought the hiccup with flash


def test_new_term_escalates_and_registers():
    # 신조어 → is_new 제안 → lite가 확정 못함 → flash 확정 → canon_flash + 어휘 등재
    items = [TagInput(3, "新造語タグ")]
    caller = caller_from({
        (LITE, 3): {"id": 3, "canonical": "新ジャンル", "is_new": True, "confidence": 0.90},
        (FLASH, 3): {"id": 3, "canonical": "新ジャンル", "is_new": True, "confidence": 0.95},
    })
    out = call_tiered(items, caller, VOCAB)
    assert out[0].status == Status.CANON_FLASH
    assert out[0].registers_vocab is True
    assert FLASH in caller.calls


def test_low_confidence_escalates():
    # 어휘 내지만 저신뢰(G4) → flash 재시도 → 통과 → canon_flash
    items = [TagInput(2, "ろく")]
    caller = caller_from({
        (LITE, 2): {"id": 2, "canonical": "ロック", "confidence": 0.55},
        (FLASH, 2): {"id": 2, "canonical": "ロック", "confidence": 0.90},
    })
    out = call_tiered(items, caller, VOCAB)
    assert out[0].status == Status.CANON_FLASH


def test_malformed_item_escalates():
    # 비구조 출력(빈 canonical, G2) → flash 재시도 → 통과
    items = [TagInput(4, "???")]
    caller = caller_from({
        (LITE, 4): {"id": 4, "canonical": ""},
        (FLASH, 4): {"id": 4, "canonical": "ロック", "confidence": 0.88},
    })
    out = call_tiered(items, caller, VOCAB)
    assert out[0].status == Status.CANON_FLASH


def test_flash_failure_demotes():
    # lite 실패 → flash도 저신뢰 → 강등(원본 보존, 다음 런 재시도)
    items = [TagInput(2, "ろく")]
    caller = caller_from({
        (LITE, 2): {"id": 2, "canonical": "ロック", "confidence": 0.55},
        (FLASH, 2): {"id": 2, "canonical": "ロック", "confidence": 0.50},
    })
    out = call_tiered(items, caller, VOCAB)
    assert out[0].status == Status.DEMOTED


def test_escalation_cap_defers_to_pending():
    # 승급 상한 도달 → 승급 없이 보류(pending), flash 미호출
    items = [TagInput(2, "ろく")]
    caller = caller_from({
        (LITE, 2): {"id": 2, "canonical": "ロック", "confidence": 0.55},
        (FLASH, 2): {"id": 2, "canonical": "ロック", "confidence": 0.90},
    })
    out = call_tiered(items, caller, VOCAB, RoutingConfig(escalation_cap=0))
    assert out[0].status == Status.PENDING
    assert FLASH not in caller.calls


def test_batch_malformed_twice_escalates_all():
    # lite 배치가 재시도까지 두 번 다 malformed → 전 항목 flash 승급
    items = [TagInput(1, "初音ミク")]
    calls = []

    def caller(model, items):
        calls.append(model)
        if model == LITE:
            return None  # both lite attempts fail
        return [{"id": 1, "canonical": "初音ミク", "confidence": 0.97}]

    out = call_tiered(items, caller, VOCAB)
    assert out[0].status == Status.CANON_FLASH
    assert calls.count(LITE) == 2  # tried lite twice before escalating
    assert FLASH in calls
