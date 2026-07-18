# vocaloid_radar

A leading-indicator radar for Vocaloid song trends — it surfaces **early signals**
(view acceleration, derivative-work bursts, tag-share shifts, co-tag clusters) from
public metadata, rather than trying to predict hits (small samples plus exogenous
events like remixes make hit prediction unverifiable, so that goal is deliberately
out of scope).

Its more interesting half is the **FinOps model-routing layer**: a difficulty-based
router that keeps LLM-heavy normalization inside free-tier quotas by spending code —
not a paid classifier — to decide which items are "hard".

> Status: early / active development. The analysis pipeline and routing layer are
> built and tested; live collection runs against a self-hosted host.

## Why this exists

Canonicalizing thousands of raw, multilingual tags into trend families is the kind of
bulk LLM work that quietly burns quota. The naive fix — a classifier that scores each
item's difficulty before routing it — **doubles** the call count (the scoring call is
itself billable). vocaloid_radar takes the opposite approach:

1. **Static tier per task.** Bulk tag canonicalization starts on the cheapest model;
   report narration starts on the top free model. No per-item difficulty call.
2. **Code gates decide pass vs. escalate (0 cost).** Structure checks — id-echo
   integrity, output shape, membership in a closed canonical vocabulary — are the
   primary signal; self-reported confidence is only a secondary hint (LLMs are not
   calibrated, so a confident wrong answer must not ride through on confidence alone).
3. **Only failing items escalate**, one free tier up, under a per-run cap. The ladder
   **stops at the top free tier** — it never calls a paid model on its own.

The result: "easy → cheap model, hard → better model" emerges automatically with
**zero** difficulty-classification calls.

## Design highlights

- **Closed-loop vocabulary.** Canonical names are the trend aggregation axis, so they
  must not drift across model versions. The prompt carries the existing vocabulary and
  the model must either map into it or explicitly flag a new-term proposal; membership
  is verified in code. As the vocabulary fills, normalization converges toward a
  deterministic lookup — fewer LLM calls over time.
- **Availability ≠ quality.** Rate-limit/5xx handling (pacing, same-tier backoff) lives
  entirely in the transport layer and raises a distinct "gateway unavailable" signal, so
  an outage defers work to the next run — it can never be mistaken for a quality failure
  and trigger escalation.
- **Run-level escalation cap.** Report generation and tag escalation share one free
  quota, so a bad batch can't starve the weekly report; over-cap items simply wait.
- **FinOps observability.** Every call is traced; the rollup reports per-tier
  calls/tokens vs. quota, cache-hit rate, escalation/demotion rates, and an
  *escalation-pair agreement rate* that tunes the confidence threshold for free (if the
  better model lands on the same answer the cheap one already proposed, the escalation
  was wasted).

See [ARCHITECTURE.md](./ARCHITECTURE.md) for the full design.

## Stack

- **Python 3.12**, [LangGraph](https://github.com/langchain-ai/langgraph) for the weekly
  batch pipeline
- **PostgreSQL 16 + pgvector** for records and song embeddings
- An **OpenAI-compatible LiteLLM gateway** as the single LLM entry point
- **Arize Phoenix** for OpenTelemetry tracing / FinOps rollups
- Public data via the VocaDB REST API and the Niconico Snapshot Search API (adapter
  pattern, so a source can be swapped)

## Quick start (local)

```sh
cp .env.example .env          # fill in GATEWAY_URL / GATEWAY_API_KEY
docker compose -f docker-compose.dev.yml up -d db
docker compose -f docker-compose.dev.yml run --rm app \
  sh -c "python -m src.db.migrate && python -m pytest -q"
```

`docker-compose.dev.yml` is a throwaway local Postgres (PG16 + pgvector) — **not** a
deployment target.

## Testing

The routing and analysis logic is pure and dependency-injected, so it is verified in
three layers:

1. **Fixtures** — spec-by-example tests with a fake gateway caller (no live LLM, no DB).
2. **DB round-trip** — cache/persist logic against a real Postgres (gated on `DB_ROUNDTRIP=1`).
3. **Live smoke** — one real gateway call to confirm structured output arrives and traces land.

```sh
python -m pytest -q                                   # fixtures
DB_ROUNDTRIP=1 python -m pytest tests/test_*_db.py    # + DB round-trip
```

## License

[MIT](./LICENSE)
