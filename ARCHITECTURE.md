# Architecture

## Overview

vocaloid_radar is a weekly batch that turns public Vocaloid metadata into a
leading-indicator report. The pipeline is a LangGraph state machine:

```
fetch_validate → normalize_classify → trend_score → cluster_detect → report_gen → publish
```

- **fetch_validate** — pull songs/tags/metrics from public sources (VocaDB REST,
  Niconico Snapshot Search) behind an adapter interface.
- **normalize_classify** — canonicalize raw tags into trend families via the FinOps
  routing layer (below). This is the LLM-heavy node.
- **trend_score** — pure functions: view/derivative velocity, tag-share deltas,
  cold-start detection.
- **cluster_detect** — group top-velocity songs by shared tag combinations.
- **report_gen** — generate the weekly narrative, grounded strictly on the computed
  evidence.
- **publish** — persist the report + evidence snapshot.

All LLM traffic goes through a single OpenAI-compatible **LiteLLM gateway**; all calls
are traced to **Arize Phoenix** for FinOps rollups.

## The FinOps routing layer

The problem: canonicalizing thousands of multilingual tags is bulk LLM work that must
stay inside free-tier daily quotas. The design keeps quality high on the hard items
without spending a single call to *decide* what's hard.

### 1. Static tier + escalate-on-failure

| Node | Task | Default tier | Escalates to |
|---|---|---|---|
| `normalize_classify` | tag canonicalization (bulk, mechanical) | cheap tier | flash tier |
| `report_gen` | weekly narrative (judgment) | flash tier | — (already top free tier) |

There is **no per-item difficulty classifier** — that call would itself double quota
usage. Difficulty is inferred *after the fact* from whether the cheap model's output
passes a set of code gates.

### 2. Code gates (0 cost) — structure first, confidence second

Applied in order to each batched response. Structure is the primary signal because
self-reported confidence is uncalibrated (a confident wrong answer must not pass on
confidence alone).

| Gate | Check | Level | On fail |
|---|---|---|---|
| G1 | response id-set == request id-set (map by id, never by position) | batch | retry batch once, then escalate |
| G2 | output shape valid + canonical name non-empty | item | escalate item |
| G3 | canonical ∈ existing vocabulary, **or** `is_new: true` is set | item | escalate item |
| G4 | confidence ≥ threshold (secondary hint) | item | escalate item |

### 3. Escalation ladder

```
tag batch → cheap tier
  ├─ batch broken (G1)        → retry cheap once → still broken → escalate every item
  ├─ item fails (G2–G4)       → that item → flash tier (within per-run cap)
  │                               ├─ passes → accept  (status: canon_flash)
  │                               └─ fails  → demote  (raw preserved, retried next run)
  ├─ over the per-run cap      → leave pending (next run)
  └─ passes                    → accept        (status: canon_lite)
```

The ladder **stops at the flash tier**. A paid model is never called automatically —
that is always an explicit, manual decision.

### 4. Closed-loop vocabulary (drift guard + G3's basis)

Canonical names are the trend aggregation axis, so they must be stable across model
versions. The system owns the vocabulary:

- The prompt includes the existing canonical vocabulary; the model must map into it or
  set `is_new: true`. Existing mappings are checked in code (G3) — stronger than trusting
  confidence.
- A new-term proposal *changes* the vocabulary, so the cheap tier may not finalize it:
  the flash tier confirms it, and on acceptance it is registered into the vocabulary
  (within the same run, so later batches see the grown set).
- Cold-start (initial backfill) converges naturally: canonical families (hundreds) are
  far fewer than raw tags (thousands), so the vocabulary fills fast and cheap-tier
  membership lookups dominate thereafter. Normalization trends toward a deterministic
  lookup — the LLM is progressively *removed* from the hot path.

### 5. Caching & batching (the biggest lever)

- **The cache is the `tags` table itself.** Tag ids are external entity ids, so each tag
  is canonicalized **once in its lifetime**; only `pending`/`demoted` rows are ever sent
  to the LLM. After the first backfill, weekly new tags number in the tens.
- **Batching** — many tags per prompt (the free limit is per-call), with a mandatory
  id-echo (G1) so results bind to inputs by id.

### 6. Invariants (availability boundary)

1. No router-level fallback between tiers in the gateway config — availability fallback
   must not ride the quality ladder.
2. 429/5xx = same-tier backoff, never an escalation trigger.
3. The batch loop paces to each tier's free-tier RPM.
4. No path calls a paid model.

The transport layer owns availability *only*: it paces, backs off on the same tier, and
raises a distinct `GatewayUnavailable` when retries are exhausted. On that signal the
orchestrator defers all remaining work to `pending` — an outage never escalates and
never demotes.

### 7. `canon_status` lifecycle

```
pending ──cheap pass──▶ canon_lite     (immutable cache)
   │  └──flash pass───▶ canon_flash    (immutable cache; escalation provenance kept)
   │  └──flash fail───▶ demoted        (raw preserved; retried next run)
   └──over cap────────▶ pending        (next run)
```

Final-tier provenance lives in the status, so most FinOps metrics can be computed from
the DB alone.

## FinOps observability

Traces feed a rollup with three sources:

- **DB rollup** (0 calls): cache-hit rate, pending backlog, escalation rate, demotion
  rate — all from `canon_status`.
- **Trace rollup**: per-tier call count + token sum vs. daily quota; an off-tier guard
  flags any call to a model outside the two free tiers (a live audit of "no paid calls").
- **Escalation-pair agreement rate** (0 extra calls): an escalation naturally leaves a
  cheap/flash output pair for the same item. If the flash result lands on the same
  canonical the cheap tier already proposed, the escalation was wasted — so the pair
  agreement rate tunes the confidence threshold from real data, for free.

## Data model (10 tables)

`songs`, `artists`, `song_artists`, `tags` (+ canon columns), `song_tags`,
`derived_works`, `metrics_daily` (the un-backfillable time series), `trend_scores`
(weekly), `weekly_reports` (narrative + evidence JSON), `song_embeddings` (pgvector).

## Key decisions & tradeoffs

- **Chosen: static tier + escalate-on-failure.** Replaces a difficulty classifier with
  0-cost code gates, so "automatic difficulty routing" is free.
- **Rejected: dynamic classifier.** Most intuitive, but the per-item scoring call
  ~doubles free-quota use — directly against the FinOps goal.
- **Rejected: pure static (no escalation).** Simplest, but hard items are stuck on the
  cheap tier with no quality floor. One 0-cost gate layer buys the escalation cheaply.
- **Boundary: escalation ≠ fallback.** Paid escalation is prohibited; a paid model is
  always a manual, deliberate choice.
