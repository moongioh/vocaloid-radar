"""tag_canon node — cache, closed-loop vocabulary, gateway caller (plan 0002 R2).

Layering (plan 0002 §5/§6):
  - routing.call_tiered owns QUALITY: gates + the lite→flash escalation ladder.
  - the caller built here owns AVAILABILITY: RPM pacing and same-tier 429/5xx
    backoff. When availability retries are exhausted it raises
    ``GatewayUnavailable`` — it never returns a "failure" that the ladder could
    mistake for a quality problem, so an outage can never trigger escalation.
  - this module also owns the CACHE: only ``pending``/``demoted`` tags are sent
    to the LLM at all (canon_* rows are immutable cache hits, 0 calls), and the
    canon vocabulary is loaded from — and grows back into — the ``tags`` table.

httpx and psycopg are imported lazily so the pure logic (prompt build, parsing,
orchestration) stays testable with a stdlib-only runner, mirroring routing.py.
"""
from __future__ import annotations

import json
import time
from typing import Callable

from .routing import (
    FLASH,
    LITE,
    RoutingConfig,
    Status,
    TagInput,
    TagResult,
    call_tiered,
)

# Free-tier requests-per-minute per model (plan 0001; re-check before relying).
RPM = {LITE: 15, FLASH: 10}
_ATTEMPTS = 3
_BACKOFF_BASE = 2.0  # seconds; doubles per retry


# --------------------------------------------------------------- seed vocabulary
#
# The closed loop (plan 0002 §4) grows the vocabulary from what the model itself
# proposes. That works once it is running; it has no answer for the FIRST run.
# With an empty allowlist no item can pass G3, so batch 1 escalated all 20 tags,
# minted the entire founding vocabulary in one shot, and exhausted the run-level
# escalation cap — measured 2026-07-29: 458 of 500 tags deferred without a single
# flash call, and the 20 founding terms covered 111 of 4,304 tag assignments
# (2.6%), which is how `Bicycle Theme` and `Blender Software` became canon
# families while `rock` and `duet` sat in pending. Everything downstream then had
# to squeeze into that slice — `YYB Kagamine Len` (an MMD model author) landed on
# `Beta Voicebank` at confidence 0.9 because no MMD-model family existed to pick.
#
# So the seed is a FLOOR, not a ceiling. It is authored, not generated: the axes
# below come from reading the 195 tags with >=5 uses. Two rules held while
# writing it, both learned from the failure above:
#   1. AXES, NOT INSTANCES. `Bicycle Theme` is an instance; `Nature Theme` is an
#      axis. The eight colour tags (red/blue/pink/...) collapse to `Color Theme`.
#   2. A family only earns a slot if the corpus actually uses it. Nothing here
#      was invented for symmetry.
# It ended up at 97 terms rather than the 40-80 the plan estimated: the tag space
# has 19 real axes, and trimming to a number picked before seeing the data would
# have meant dropping axes that carry usage. Prompt cost is ~2 KB, and a WIDER
# allowlist means FEWER is_new proposals, so it also cuts escalation.
#
# Changing this set changes how every future tag is classified. It is deliberately
# in code, not in the DB, so a vocabulary change is a reviewable diff (plan 0004).
SEED_VOCAB: "frozenset[str]" = frozenset({
    # 장르
    "Rock", "Alternative Rock", "Metal", "Pop", "J-Pop", "K-Pop", "Electronic",
    "EDM", "House", "Trap Music", "Drum and Bass", "Breakcore", "Hyperpop",
    "Chiptune", "Jazz", "Hip Hop", "Ballad", "Ambient Music",
    # 무드
    "Cute", "Dark Mood", "Sad Mood", "Calm Mood", "Funny", "Creepy", "Cool",
    # 테마
    "Love Theme", "Death Theme", "Blood Theme", "Supernatural Theme",
    "Religion Theme", "Animal Theme", "Nature Theme", "Seasonal Theme",
    # PV·영상 제작형태
    "2D Animation", "3D Animation", "Editor PV", "Official Art PV",
    "Live Action PV", "Music Visualization", "Pixel Art", "Kinetic Typography",
    "AI-Generated Art",
    # 색 — 개별 색 태그 8종이 여기로 접힌다
    "Color Theme",
    # 가창·합성 기술
    "Cross-Lingual Synthesis", "Human Vocals", "Robotic Vocals", "Harsh Vocals",
    "Speech Vocals", "Vocal Range", "Rapping", "Choir Vocals", "A Cappella",
    "Tuning Quality", "Voice Genderswap",
    # 보이스뱅크 상태
    "Beta Voicebank", "Unofficial Voicebank", "Imported Voicebank",
    "Voicebank Release", "Voicebank Demo", "Derived Voicebank",
    "Unconfirmed Vocalist",
    # 파생·커버 관계
    "Derivative Work", "Cover", "Self-Cover", "Fan Work", "Parody", "Remix",
    "Changed Lyrics", "Changed Language", "Version Variant", "Debut Work",
    # 편성
    "Duet", "Group Ensemble",
    # 언어·자막
    "Subtitled", "Multilingual", "Unsupported Language", "Instrumental",
    # 데이터 배포
    "Source Data Available", "Karaoke Available",
    # 악기·기법
    "Piano", "Electric Guitar", "Acoustic Guitar", "Sampling",
    # 제작 소프트웨어
    "Production Software",
    # 콘텐츠 경고
    "Flashing Lights Warning", "Explicit Content", "Content Warning",
    # 원작·프랜차이즈
    "Anime Original Song", "Video Game Theme", "Franchise Reference",
    # 캐릭터·모델
    "Vocalist Character", "MMD Model",
    # 시대·스타일
    "Retro Style", "80s Style", "Japanese Traditional Style",
    # 메타
    "Experimental", "Meme",
})


class GatewayUnavailable(Exception):
    """The gateway kept answering 429/5xx. Availability, not quality: the run
    defers remaining work to the next run instead of escalating tiers."""


# ---------------------------------------------------------------- pure: prompt

def build_prompt(items: "list[TagInput]", vocab: "set[str]") -> str:
    """Closed-loop vocabulary prompt (plan 0002 §4).

    The model must map each tag into the existing canon vocabulary or explicitly
    flag a new-term proposal with is_new — free-form canon names are how canon
    families drift apart across model versions, so they are forbidden.
    """
    payload = json.dumps(
        [{"id": it.id, "name": it.name} for it in items], ensure_ascii=False
    )
    vocab_list = json.dumps(sorted(vocab), ensure_ascii=False)
    return (
        "You canonicalize Vocaloid song tags into canonical trend-family names.\n"
        f"Existing canon vocabulary (the ONLY allowed canonical values): {vocab_list}\n"
        f"Input tags: {payload}\n\n"
        "Rules:\n"
        "1. Return ONLY a JSON object: {\"items\": [{\"id\": <int>, \"canonical\": <str>, "
        "\"confidence\": <0..1>, \"is_new\": <bool>}]}.\n"
        "2. Echo every input id exactly once. Do not add, drop, or reorder ids.\n"
        "3. \"canonical\" MUST be one of the existing vocabulary values. If none fits, "
        "propose a new canonical name and set \"is_new\": true.\n"
        "4. \"confidence\" is your honest mapping confidence.\n"
    )


def parse_response(content: str) -> "list[dict] | None":
    """Parse the model's JSON reply into per-item dicts. None = unparseable batch.

    Tolerates a code fence and either a bare array or the {"items": [...]}
    wrapper the prompt asks for. Anything else is a G1 batch failure upstream.
    """
    text = content.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(data, dict):
        data = data.get("items")
    if not isinstance(data, list) or not all(isinstance(d, dict) for d in data):
        return None
    return data


# ------------------------------------------------------- transport: the caller

# post(body) -> (http_status, response_json_or_None). Injected in tests.
Post = Callable[[dict], "tuple[int, dict | None]"]


def _httpx_post(base_url: str, api_key: str) -> Post:
    import httpx  # lazy: keeps fixture tests stdlib-only

    # 180s, not 60s: tier-1 is a thinking model since 2026-07-29 (plan 0003 E2).
    # Measured on the box against a real 20-tag batch — gw-gemma 43.9s / 1961 tok
    # (897 of them reasoning) vs the old gw-lite 3.3s / 998 tok. 60s left only 27%
    # headroom, and the margin SHRINKS as the canon vocabulary grows and lengthens
    # the prompt. A timeout here is not a slow item, it is the whole batch failing
    # G1 and 20 tags going unclassified, so the headroom is deliberately generous.
    client = httpx.Client(
        base_url=base_url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=180.0,
    )

    def post(body: dict) -> "tuple[int, dict | None]":
        r = client.post("/v1/chat/completions", json=body)
        try:
            return r.status_code, r.json()
        except ValueError:
            return r.status_code, None

    return post


def make_gateway_caller(
    base_url: "str | None" = None,
    api_key: "str | None" = None,
    *,
    vocab: "set[str]",
    post: "Post | None" = None,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
):
    """Build the Caller that routing.call_tiered drives.

    ``vocab`` is the SAME set object the orchestrator mutates as new canon terms
    are confirmed, so later batches automatically prompt with the grown
    vocabulary. Availability policy lives here and only here (§6): pace to the
    free-tier RPM, back off on 429/5xx within the SAME tier, and raise
    ``GatewayUnavailable`` when retries run out. A non-429 4xx means the request
    itself is bad → None, which the ladder treats as a malformed batch.
    """
    if post is None:
        post = _httpx_post(base_url or "", api_key or "")
    last_call: dict[str, float] = {}

    def caller(model: str, items: "list[TagInput]") -> "list[dict] | None":
        wait = 60.0 / RPM[model] - (monotonic() - last_call.get(model, float("-inf")))
        if wait > 0:
            sleep(wait)
        body = {
            "model": model,
            "messages": [{"role": "user", "content": build_prompt(items, vocab)}],
            "response_format": {"type": "json_object"},
        }
        for attempt in range(_ATTEMPTS):
            status, data = post(body)
            if status == 429 or status >= 500:
                if attempt < _ATTEMPTS - 1:
                    sleep(_BACKOFF_BASE * 2**attempt)
                continue
            last_call[model] = monotonic()
            if status != 200 or data is None:
                return None
            try:
                content = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                return None
            return parse_response(content)
        raise GatewayUnavailable(f"{model}: {_ATTEMPTS} attempts exhausted on 429/5xx")

    return caller


# ------------------------------------------------------- pure: orchestration

def normalize_classify(
    items: "list[TagInput]",
    vocab: "set[str]",
    caller,
    batch_size: int = 20,
    config: "RoutingConfig | None" = None,
) -> "list[TagResult]":
    """Run cache-miss tags through the ladder in batches. Mutates ``vocab``.

    - The escalation cap is a RUN-level budget (plan A3): flash usage by earlier
      batches shrinks the cap handed to later ones.
    - Confirmed new canon terms are registered into ``vocab`` between batches,
      so the closed loop converges within a single run.
    - On GatewayUnavailable everything not yet processed is deferred to
      ``pending`` — an outage never escalates and never demotes (§6).
    """
    config = config or RoutingConfig()
    remaining_cap = config.escalation_cap
    out: list[TagResult] = []
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        try:
            results = call_tiered(
                batch, caller, vocab,
                RoutingConfig(config.confidence_threshold, remaining_cap),
            )
        except GatewayUnavailable:
            out.extend(TagResult(id=it.id, status=Status.PENDING) for it in items[i:])
            break
        flash_used = sum(
            1 for r in results if r.status in (Status.CANON_FLASH, Status.DEMOTED)
        )
        remaining_cap = max(0, remaining_cap - flash_used)
        for r in results:
            if r.registers_vocab and r.canon_name:
                vocab.add(r.canon_name)
        out.extend(results)
    return out


# ----------------------------------------------------------------- db (thin)

def load_work(conn, limit: int = 500) -> "list[TagInput]":
    """Cache-miss tags (pending + demoted retries), **most-used first**.

    The ordering is load-bearing, not cosmetic. Because the vocabulary is a
    closed loop, whichever tags are processed FIRST decide the canon families
    every later tag must fit into. The original ``ORDER BY id`` was VocaDB tag
    creation order — effectively alphabetical, and uncorrelated with importance:
    the first 20 rows covered 111 of 4,304 tag assignments (2.6%), while the 20
    most-used tags cover 1,124 (26.1%). Same 20 slots, 10x the corpus.

    LEFT JOIN, not INNER: a tag nobody has used must sort LAST, never drop out of
    the work queue entirely. The ``t.id`` tie-break keeps runs reproducible.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT t.id, t.name "
            "FROM tags t LEFT JOIN song_tags st ON st.tag_id = t.id "
            "WHERE t.canon_status IN ('pending', 'demoted') "
            "GROUP BY t.id, t.name "
            "ORDER BY count(st.tag_id) DESC, t.id "
            "LIMIT %s",
            (limit,),
        )
        return [TagInput(id=r[0], name=r[1]) for r in cur.fetchall()]


def load_vocab(conn) -> "set[str]":
    """The G3 allowlist: the authored seed UNION everything earlier runs confirmed.

    Union, not replace — the seed removes the cold start, it does not close the
    loop. Flash-confirmed ``is_new`` proposals still grow the vocabulary exactly
    as plan 0002 §4 specifies.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT canon_name FROM tags "
            "WHERE canon_status IN ('canon_lite', 'canon_flash') AND canon_name IS NOT NULL"
        )
        return set(SEED_VOCAB) | {r[0] for r in cur.fetchall()}


def persist(conn, results: "list[TagResult]") -> None:
    """Apply status transitions (plan 0002 §7). PENDING rows are left untouched."""
    accepted = [
        (r.canon_name, r.confidence, r.status.value, r.id)
        for r in results
        if r.status in (Status.CANON_LITE, Status.CANON_FLASH)
    ]
    demoted = [(r.id,) for r in results if r.status is Status.DEMOTED]
    with conn.cursor() as cur:
        cur.executemany(
            "UPDATE tags SET canon_name = %s, canon_confidence = %s, canon_status = %s "
            "WHERE id = %s",
            accepted,
        )
        # Demotion preserves the raw tag: only the status moves.
        cur.executemany("UPDATE tags SET canon_status = 'demoted' WHERE id = %s", demoted)
    conn.commit()


def run_normalize(
    conn,
    *,
    base_url: "str | None" = None,
    api_key: "str | None" = None,
    post: "Post | None" = None,
    limit: int = 500,
    batch_size: int = 20,
    config: "RoutingConfig | None" = None,
) -> "dict[str, int]":
    """The node entry point: load cache-misses, route, persist, summarize."""
    items = load_work(conn, limit)
    vocab = load_vocab(conn)
    caller = make_gateway_caller(base_url, api_key, vocab=vocab, post=post)
    results = normalize_classify(items, vocab, caller, batch_size, config)
    persist(conn, results)
    summary = {s.value: 0 for s in Status}
    for r in results:
        summary[r.status.value] += 1
    summary["total"] = len(results)
    return summary
