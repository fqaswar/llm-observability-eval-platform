# LLM Observability & Evaluation Platform

A RAG application over the Anthropic documentation, wrapped in distributed tracing and an
automated evaluation suite that gates every pull request. The RAG app is deliberately
small — the point is the instrumentation around it: proving on every change whether the
system hallucinates, what it costs, and where the time goes.

**Status:** Phases 1–4 complete (RAG app + tracing + eval harness + CI gate). Phases 5–6 pending.

## Architecture

```
                  ┌── ingest ──────────────────────────────┐
  docs.claude.com │ fetch_docs → chunk → embed → pgvector  │
                  └────────────────────────────────────────┘
                                    │
  POST /ask ──► embed query ──► pgvector top-k ──► Claude ──► grounded answer
                    └──────────── traced end to end (Phase 2) ──────────┘
                                    │
  eval/run_eval.py ──► RAGAS metrics ──► CI quality gate (Phases 3–4)
```

## Stack

| Component | Choice | Why |
|---|---|---|
| Backend | FastAPI | RAGAS/DeepEval are Python-only; one language end to end |
| Vector store | Supabase Postgres + pgvector 0.8.2 | HNSW index, hosted, no containers |
| Embeddings | `BAAI/bge-base-en-v1.5` (local) | 768-dim, runs on CPU, no API key, no cost |
| Generation | Claude Haiku 4.5 | Cheap enough to run evals on every PR |
| Tracing | Langfuse Cloud | Phase 2 |
| Evaluation | RAGAS | Phase 3 |

Only one paid credential is required for the whole project: `ANTHROPIC_API_KEY`.

## Setup

**Requires Python 3.10 or 3.11.** PyTorch publishes no wheels for 3.13, and none past
2.2.2 for Intel macOS — see the notes in `requirements.txt`.

```bash
python3.10 -m venv .venv310
.venv310/bin/pip install -r requirements.lock   # exact verified versions
# or: .venv310/bin/pip install -r requirements.txt   (constraints + rationale)

cp .env.example .env      # then fill in the five values
```

Prerequisites in Supabase: enable the `vector` extension (Database → Extensions), and copy
the **Session pooler** connection string (the direct connection is IPv6-only on the free
tier and fails from CI).

Verify the extension is live in the database your `DATABASE_URL` points at:

```bash
psql "$DATABASE_URL" -c "SELECT extversion FROM pg_extension WHERE extname='vector';"
```

## Usage

```bash
# 1. Download the corpus (562 pages; --limit N for a subset)
.venv310/bin/python -m ingest.fetch_docs

# 2. Chunk, embed, and index. Idempotent — safe to re-run.
.venv310/bin/python -m ingest.ingest

# 3. Serve
.venv310/bin/uvicorn app.main:app --reload
```

```bash
curl -s localhost:8000/health

curl -s -X POST localhost:8000/ask \
  -H 'content-type: application/json' \
  -d '{"question": "Which Claude models are deprecated?"}' | python3 -m json.tool
```

The response carries the answer, the retrieved sources with similarity scores, the raw
context strings, token usage, a per-stage latency breakdown, and the Langfuse `trace_id`.

## Tracing

Every request produces one Langfuse trace:

```
rag-query                    (span)       ← input, output, tags, total latency
├── retrieve                 (retriever)  ← sources, similarity scores
│   ├── embed-query          (embedding)
│   └── pgvector-search      (span)       ← row count, search time
└── generate                 (generation) ← prompt, tokens, cost in USD
```

Cost is inferred server-side from the model name and the token counts reported by the
Anthropic SDK — Langfuse's fallback tokenizer is inaccurate for Claude, so the app always
sends Anthropic's own numbers.

```bash
# Ask one question and print its trace URL
.venv310/bin/python -m scripts.smoke_trace "your question"

# Fetch a trace back from the API to inspect its structure
.venv310/bin/python -m scripts.show_trace <trace_id>
```

Tracing is **optional at runtime**. If the credentials are absent, rejected, or Langfuse is
unreachable, the app logs a warning and serves normally — observability must not be able to
take down the service it observes. Set `LANGFUSE_ENABLED=false` to disable it explicitly;
the Phase 3 eval runner does this so a 30-question run does not emit 30 traces.

## Evaluation

26 hand-written questions in `eval/golden_set.json`, each with a `reference` answer
(required by `ContextRecall` and `ContextPrecisionWithReference`), across four kinds:

| Tag | What it tests |
|---|---|
| `factual` | the answer is stated plainly in one document |
| `synthesis` | requires combining facts across chunks |
| `edge` | ambiguous, or the docs give a conditional answer |
| `trap` | sounds answerable but the corpus does not contain the answer — the correct behaviour is to **decline** |

```bash
.venv310/bin/python -m eval.run_eval                    # full run
.venv310/bin/python -m eval.run_eval --limit 4          # cheap smoke run
.venv310/bin/python -m eval.run_eval --tags trap        # traps only
.venv310/bin/python -m eval.run_eval --min-faithfulness 0.85   # gate (exit 1 on breach)
```

Scored with RAGAS: **faithfulness** (is the answer supported by the retrieved context),
**answer relevancy**, **context precision**, and **context recall**. The judge is Claude
Sonnet, reusing the same `ANTHROPIC_API_KEY`; embeddings are the app's own local model.

**A measurement subtlety worth knowing.** RAGAS scores a deliberately non-committal answer
("the documentation does not cover this") as **0 relevancy**. On a trap question that
refusal is exactly right, so averaging relevancy across traps would punish the app for
behaving correctly. The runner therefore aggregates relevancy over non-trap questions only,
and reports `trap_faithfulness` / `trap_context_recall` separately. Per-question scores stay
in the report either way.

### Baseline (2026-09-02, 26 questions, top_k=5)

| Metric | Score |
|---|---|
| faithfulness | 0.924 |
| answer relevancy | 0.717 *(non-trap only)* |
| context precision | 0.636 |
| context recall | 0.740 |
| trap faithfulness | 0.822 |

Faithfulness by tag: edge 1.000 · factual 0.958 · synthesis 0.880 · trap 0.822.
Cost: **$0.164** for the run (~$0.006/question on Haiku), 24 minutes wall clock — of which
only 101s is answering; the rest is the judge.

**The baseline found a real retrieval bug, and it is the most interesting number here.**
Context precision sits at 0.636 because retrieval reliably finds the right *document* but
often the wrong *chunk within it*. Concretely: the cache-minimums table lives in
`prompt-caching.md` chunk 9, but a question about it retrieves chunk 0 — the page
introduction, whose generic prose resembles many queries. The app then correctly answers
"the documentation does not cover this" even though the fact is indexed.

That is a chunking and retrieval problem, not a generation problem, and the split between
`faithfulness` (0.924 — the model does not invent) and `context_precision` (0.636 —
retrieval hands it the wrong material) is exactly what separates the two. Candidate fixes
for a later pass: prepend the document title to every chunk, raise `top_k`, or add a
reranking step.

Two helpers for working on the golden set, neither of which calls Claude:

```bash
.venv310/bin/python -m scripts.verify_golden      # do the questions retrieve real context?
.venv310/bin/python -m scripts.peek "question"    # what does retrieval actually return?
.venv310/bin/python -m scripts.probe_ragas        # is the judge wired up correctly?
```

## CI quality gate

`.github/workflows/eval.yml` runs the evaluation on every pull request and blocks the merge
when a gated metric drops.

| Gate | Threshold | Baseline |
|---|---|---|
| faithfulness | ≥ 0.85 | 0.924 |
| context recall | ≥ 0.60 | 0.740 |
| context precision | ≥ 0.50 | 0.636 |

Thresholds sit just below the recorded baseline — low enough not to fire on ordinary judge
variance, high enough to catch a real regression.

Each run spins up a disposable `pgvector/pgvector:pg17` service container and ingests a
40-page subset of the corpus. **CI never points at Supabase:** concurrent PRs would race on
the same tables and exhaust the free tier's connection limit. Langfuse is disabled in CI —
tracing is a local and demo concern.

The workflow then posts a comment on the PR with each metric against `main`, the delta, and
what the run cost. Pushes to `main` refresh the committed baseline that PR runs compare
against.

**Setup:** the repository needs one secret — `ANTHROPIC_API_KEY`, under
Settings → Secrets and variables → Actions.

**Cost:** roughly $0.16 per run. A `concurrency` block cancels superseded runs so a rapid
series of pushes does not bill for every intermediate commit.

## The regression catch

The point of all the preceding machinery is that a change which looks fine in review gets
stopped before it merges. To prove that, `regression/shrink-retrieval-context` makes exactly
one edit:

```diff
- rag_top_k: int = 5
+ rag_top_k: int = 1
```

Retrieval now hands the model a single chunk instead of five. There is no syntax error, no
failing unit test, nothing a reviewer skimming a one-line config change would catch. The
answers stay fluent and confident — they are simply less often grounded in the right
material.

The eval gate catches it, the check goes red, and the PR comment shows the drop against
`main`. The fix is then pushed to the same branch so the break, the detection, and the
recovery all live in one thread.

This is why `top_k` and the system prompt are configuration rather than literals: the demo
is a one-line diff, which makes the causality obvious to anyone reading the PR.

## Layout

```
app/
  config.py      # all tunables — top_k and the system prompt live here, not inline
  db.py          # connection pool + schema DDL (single source of truth)
  embeddings.py  # sentence-transformers singleton
  retrieval.py   # pgvector cosine search
  generation.py  # Claude call + prompt construction
  rag.py         # answer_question() — the seam the API and eval runner share
  main.py        # FastAPI wrapper
ingest/
  fetch_docs.py  # corpus download via docs.claude.com/llms.txt
  chunker.py     # heading-aware markdown chunking
  ingest.py      # chunk → embed → upsert
eval/
  golden_set.json  # 26 questions with reference answers
  run_eval.py      # answer → score → aggregate → gate
scripts/
  peek.py          # what does retrieval return? (no LLM call)
  verify_golden.py # do golden questions retrieve real context? (no LLM call)
  probe_ragas.py   # is the RAGAS judge wired correctly? (1 row)
  smoke_trace.py   # ask one question, print its trace URL
  show_trace.py    # fetch a trace back from the Langfuse API
```

Two design decisions carry the later phases:

- **`answer_question()` is a plain in-process function** returning `contexts` alongside the
  answer. The Phase 3 eval runner imports and calls it directly — no HTTP glue — and RAGAS
  scores faithfulness against exactly those context strings.
- **`top_k` and the system prompt are configuration, never literals.** Phase 5 stages its
  deliberate quality regression by changing one value, which keeps the demo diff to a
  single line.
