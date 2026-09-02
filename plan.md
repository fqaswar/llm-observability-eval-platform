# Project 1 — LLM Observability & Eval Platform: Implementation Plan

## Context

`/Users/mac/Documents/mlOps-aiOps` currently contains only the roadmap markdown — this is a
greenfield build. The goal is a portfolio artifact for an LLMOps / AI Quality Engineer role:
a small RAG app whose real value is the instrumentation around it — every request traced,
every PR gated by an automated eval score, cost per query visible.

Decisions taken (differ from the roadmap where noted, to cut cost and credential surface):

| Concern | Roadmap says | We do | Why |
|---|---|---|---|
| Vector store | Supabase cloud | **Supabase Postgres + pgvector** (project `ai-knowledge-base`) | Already provisioned; pgvector 0.8.2 available as a one-click extension. No containers. |
| Tracing | Self-hosted Langfuse | **Langfuse Cloud (free tier)** | v3 self-hosting is a 5-service stack needing ~16 GiB RAM. Cloud removes that entirely. |
| Embeddings | OpenAI/Voyage/Cohere | **Local `sentence-transformers`** | Zero cost, zero credentials, runs on the Mac. |
| Eval judge | RAGAS default (OpenAI) | **Claude, same API key** | Avoids a second paid account. |
| Corpus | any docs | **Anthropic/Claude public docs (markdown)** | Public, technical, easy to write trap questions against. |

**Net: exactly one paid credential for the whole project — `ANTHROPIC_API_KEY`.**
**No Docker in local development** (CI uses one throwaway container — see §Phase 4).

---

## Step 0 — Credentials & prerequisites (do this BEFORE any code)

### 0.1 The one thing you must buy
- **Anthropic API key** — console.anthropic.com → API Keys. Load ~$10 of credit; this
  project will not come close to using it. Used for *both* generation (Haiku) and the
  RAGAS judge (Sonnet).
- Store as `ANTHROPIC_API_KEY` in a **gitignored `.env`**. Commit a `.env.example` with
  empty values instead.

### 0.2 Free accounts to create
- **Langfuse Cloud** — cloud.langfuse.com → sign up → create an organization and a
  project. It issues `LANGFUSE_PUBLIC_KEY` (`pk-lf-…`) and `LANGFUSE_SECRET_KEY`
  (`sk-lf-…`). Note which region you picked: `LANGFUSE_HOST` is
  `https://cloud.langfuse.com` (US) or `https://eu.cloud.langfuse.com` (EU) — the wrong
  one authenticates against the wrong tenant and silently drops traces.
- **GitHub account + a new empty repo** (`llm-observability-eval-platform`). Needed by
  Phase 4. This becomes repo 1 of the 5-repo MLOps org.
- **GitHub Actions secret**: repo → Settings → Secrets and variables → Actions → New
  repository secret → `ANTHROPIC_API_KEY`. Do this at Phase 4 time, not now.

### 0.3 Supabase Postgres — the complete requirement
Project: **`ai-knowledge-base`** (org `fqaswar@techtiz.co`, free tier). Two steps only:

1. **Enable pgvector** — Dashboard → Database → **Extensions** → search `vector` →
   toggle **on** (accept the default `extensions` schema). Version available: **0.8.2**,
   with both `ivfflat` and `hnsw` access methods. *(Not the Integrations/Wrappers page —
   "S3 Vectors Wrapper" there is an unrelated AWS product.)*
2. **Copy the connection string** — top bar → **Connect** → **Session pooler** URI →
   substitute your database password → `DATABASE_URL` in `.env`.

**Nothing further.** Do not create tables or indexes by hand — `app/db.py` creates the
`rag_chunks` table and the HNSW index at startup (§1.2), so the schema stays in version
control rather than living only in the dashboard.

Verify before Phase 1:
```bash
psql "$DATABASE_URL" -c "SELECT extversion FROM pg_extension WHERE extname='vector';"
```
A version number means ready. Extensions are per-database, so run this with the exact
`DATABASE_URL` the app will use.

**Why Session pooler:** Supabase's direct connection is IPv6-only on the free tier, which
fails from many networks and from GitHub Actions runners. The pooler is IPv4 and is what
CI will need in Phase 4. Note it runs in transaction mode — fine for this app, but avoid
server-side prepared statements (`psycopg` autocommit + simple queries is safe).

**Namespacing:** the project name suggests other data may live here. All objects for this
project are prefixed `rag_` (table `rag_chunks`) so nothing can collide.

### 0.4 Local tooling
- Python 3.11+ and `uv` (or venv + pip).
- **No Docker required** — the database is hosted. (Phase 4 CI uses a pgvector container
  only as a disposable test DB; it needs nothing from you.)

### 0.5 What you do NOT need
No OpenAI, Voyage, Cohere, AWS, or HuggingFace account, and no ClickHouse/Redis/MinIO.
`sentence-transformers` downloads model weights anonymously.

### 0.6 Final `.env` — use these exact names
```bash
ANTHROPIC_API_KEY=sk-ant-…            # rotate the one pasted in chat first
DATABASE_URL=postgresql://postgres.llzhg…:PASSWORD@aws-…pooler.supabase.com:5432/postgres
LANGFUSE_PUBLIC_KEY=pk-lf-…
LANGFUSE_SECRET_KEY=sk-lf-…
LANGFUSE_HOST=https://cloud.langfuse.com
```
**Names matter.** Both SDKs auto-read their config from the environment, but only under
these names: the Anthropic SDK wants `ANTHROPIC_API_KEY` (not `CLAUDE_API_KEY`) and the
Langfuse SDK wants `LANGFUSE_HOST` (not `LANGFUSE_BASE_URL`). Using other names means
`Anthropic()` and `get_client()` silently pick up nothing and fail at call time.

Gitignore `.env`; commit `.env.example` with the values blank.

---

## Note on Langfuse: why Cloud, not self-hosted

Roadmap Step 2.1 says "self-host Langfuse pointed at your existing Postgres." That was
true for Langfuse v2. **v3 is a five-service stack** — `langfuse-web`, `langfuse-worker`,
Postgres, ClickHouse (mandatory traces OLAP), Redis, and MinIO — wanting ~16 GiB RAM.
Using Langfuse Cloud's free tier removes that stack entirely; the SDK code in Phase 2 is
byte-for-byte identical either way, so self-hosting stays available later by changing
`LANGFUSE_HOST`.

**Privacy note:** traces sent to Cloud include your questions, full prompts, retrieved
doc chunks, and answers. Fine for a public-docs corpus — worth remembering before pointing
this at anything proprietary.

---

## Phase 1 — RAG app (Days 1–4)

### Layout
```
├─ app/
│  ├─ main.py                  # FastAPI: POST /ask, GET /health
│  ├─ config.py                # pydantic-settings, reads .env
│  ├─ db.py                    # psycopg pool, schema bootstrap
│  ├─ embeddings.py            # sentence-transformers singleton
│  ├─ retrieval.py             # embed query -> pgvector top-k
│  ├─ generation.py            # Anthropic call
│  └─ rag.py                   # answer_question(): the orchestrator
├─ ingest/
│  ├─ fetch_docs.py            # download corpus -> data/raw/
│  └─ ingest.py                # chunk -> embed -> upsert
├─ eval/
│  ├─ golden_set.json
│  └─ run_eval.py
├─ .github/workflows/eval.yml
├─ .env.example
└─ README.md
```

`app/rag.py::answer_question(question) -> AnswerResult` is the **single seam the whole
project hangs off**. It must be importable and callable in-process — the eval runner in
Phase 3 calls it directly (no HTTP), and `main.py` is a thin wrapper over it. Return a
dataclass carrying `answer`, `contexts: list[str]`, `usage`, `latencies`, `trace_id` —
RAGAS needs `contexts`, so do not let the API shape discard them.

### 1.1 Corpus
`ingest/fetch_docs.py` pulls Anthropic docs markdown into `data/raw/` (gitignore the
raw docs; commit the fetch script so the corpus is reproducible).

### 1.2 Schema
Created by `app/db.py` at startup against your existing `DATABASE_URL` — idempotent, so
it is safe on every boot:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS rag_chunks (
  id BIGSERIAL PRIMARY KEY,
  source TEXT NOT NULL,
  chunk_index INT NOT NULL,
  content TEXT NOT NULL,
  embedding vector(768) NOT NULL,
  UNIQUE (source, chunk_index)
);
CREATE INDEX IF NOT EXISTS rag_chunks_embedding_idx
  ON rag_chunks USING hnsw (embedding vector_cosine_ops);
```
HNSW (not IVFFlat) — better recall, no training step. `vector(768)` matches the model below.

### 1.3 Embeddings
`BAAI/bge-base-en-v1.5` — 768-dim, 512-token window, strong CPU retrieval quality.
Load the `SentenceTransformer` **once at module import** in `app/embeddings.py`; loading
per-request will dominate your latency traces and make Phase 2 numbers meaningless.
Put the dimension in config so swapping to `bge-small` (384) is a one-line change.

### 1.4 Ingestion
Chunk ~800 tokens with ~100 overlap, splitting on markdown headings first so chunks stay
semantically whole. Batch-encode (`model.encode(list_of_chunks)`), then upsert with
`ON CONFLICT (source, chunk_index) DO UPDATE` so re-running is idempotent.

### 1.5 Query flow
`POST /ask {"question": "..."}` → embed → `ORDER BY embedding <=> $1 LIMIT k` (k=5) →
build prompt → `claude-haiku-*` → return answer + contexts + usage.

Two things to get right now because Phase 5 depends on them:
- **`k` and the system prompt must be config values**, not literals — Phase 5's
  "intentional regression" is just changing one of them.
- The system prompt must instruct grounding ("answer only from the context; if the
  context does not contain the answer, say so") — otherwise the trap questions in the
  golden set will hallucinate from the start and faithfulness never has room to drop.

**Checkpoint:** `curl -X POST localhost:8000/ask -d '{"question":"..."}'` returns a
grounded answer.

---

## Phase 2 — Tracing (Days 5–7)

### 2.1 Connect to Langfuse Cloud
No infrastructure step — the three env vars from §0.2 are the entire setup. The SDK reads
`LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST` from the environment
automatically; `get_client()` takes no arguments.

Smoke-test auth before instrumenting anything, so a 401 doesn't get misread later as a
tracing bug:
```python
from langfuse import get_client
assert get_client().auth_check()
```

### 2.2–2.4 Instrument
`pip install langfuse` (v4 — an OpenTelemetry rewrite; **v2 tutorials using
`langfuse.trace()` / `.generation()` will not work**).

```python
from langfuse import observe, get_client
langfuse = get_client()

@observe()                                   # parent trace, one per question
def answer_question(question: str) -> AnswerResult:
    with langfuse.start_as_current_observation(
            as_type="span", name="retrieve", input={"question": question}) as span:
        chunks = retrieve(question, k=settings.top_k)
        span.update(output={"n": len(chunks),
                            "scores": [c.score for c in chunks]})

    with langfuse.start_as_current_observation(
            as_type="generation", name="generate",
            model=settings.model, input=prompt) as gen:
        resp = client.messages.create(...)
        gen.update(output=resp.content[0].text,
                   usage_details={"input":  resp.usage.input_tokens,
                                  "output": resp.usage.output_tokens})
```
Pass **Anthropic's own token counts** — Langfuse's fallback tokenizer is inaccurate for
Claude. Omit `cost_details`; Langfuse infers USD from its built-in Anthropic price table
by model name. Call `langfuse.flush()` on FastAPI shutdown and at the end of the eval run,
or short-lived processes will drop traces.

**Checkpoint:** each question shows one trace in Langfuse: retrieve → generate, with
similarity scores, tokens, cost, latency.

---

## Phase 3 — Eval harness (Days 8–11)

### 3.1 Golden set — write `reference` answers
`eval/golden_set.json`, 20–30 items:
```json
{"id":"q001","question":"...","reference":"...","tags":["factual"]}
```
**Include a `reference` for every item.** `LLMContextRecall` and
`LLMContextPrecisionWithReference` cannot run without it — reference-free evals can only
do faithfulness + response relevancy + `LLMContextPrecisionWithoutReference`, which loses
the retrieval-recall signal the roadmap's Step 3.3 asks for.

Mix: ~15 easy factual, ~8 edge/ambiguous, ~5 traps (plausible-sounding questions the docs
do not answer; the reference is "the documentation does not cover this").

### 3.2–3.3 RAGAS
`pip install ragas` (0.4.x). **The API changed** — `LangchainLLMWrapper` /
`LangchainEmbeddingsWrapper` are the legacy path and most tutorials online still use them.
Current form:
```python
from anthropic import Anthropic
from ragas import evaluate, EvaluationDataset
from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.metrics import (Faithfulness, ResponseRelevancy,
                           LLMContextPrecisionWithReference, LLMContextRecall)

judge = llm_factory("claude-sonnet-4-5", provider="anthropic",
                    client=Anthropic())          # `client` is required in 0.4
emb   = HuggingFaceEmbeddings(model="BAAI/bge-base-en-v1.5")   # reuse app model
```
Note `answer_relevancy` is now `ResponseRelevancy`.

### 3.4 Runner
`eval/run_eval.py`:
1. Load golden set.
2. For each item call `app.rag.answer_question` **in-process** — same repo, same language,
   no HTTP glue. Collect `user_input`, `response`, `retrieved_contexts`, `reference`.
3. `evaluate(...)` with the four metrics.
4. Sum tokens/cost across the run from the returned usage.
5. Write `eval/results/latest.json` — aggregate scores, per-question rows, total tokens,
   total USD, wall time — and print a markdown summary table to stdout.
6. Exit non-zero if any gated metric is below threshold. Accept `--threshold-faithfulness`
   etc. as flags so CI owns the numbers, not the script.

Run it against the golden set once and **read the per-question output** before setting
thresholds — thresholds picked blind will either never fire or fire constantly.

**Checkpoint:** `python -m eval.run_eval` produces a score report.

---

## Phase 4 — CI gate (Days 12–14)

`.github/workflows/eval.yml`, `on: pull_request`:
1. `services:` block running `pgvector/pgvector:pg17` as a **throwaway test database**.
   Do *not* point CI at Supabase: concurrent PRs would race on the same tables, and the
   free tier's connection limit is easily exhausted by parallel runs. A disposable
   container per run keeps evals isolated and reproducible.
2. Cache pip **and** the HuggingFace model cache (`~/.cache/huggingface`) — the embedding
   model download is the slowest step by far.
3. Run ingestion against a **small committed subset** of the corpus, not the full docs
   set, to keep CI under a few minutes.
4. `python -m eval.run_eval --threshold-faithfulness 0.85 …` → non-zero exit fails the check.
5. Post the markdown summary as a PR comment via `actions/github-script`, comparing
   against `main`'s stored `eval/results/latest.json` for a score **and cost delta**.

Secret needed: `ANTHROPIC_API_KEY` (Step 0.2). Langfuse is *not* run in CI — tracing is a
local/demo concern; don't try to boot a five-service stack per PR.

---

## Phase 5 — Prove the regression catch (Days 15–17)

Because `top_k` and the system prompt are config (§1.5), the regression PR is a one-line
diff — which makes the demo much more legible than a sprawling change.

1. Branch, set `top_k: 5 → 1` (or blunt the grounding instruction). Open PR.
2. CI fails; PR comment shows faithfulness/context-recall dropping below threshold.
3. Revert on the same PR; CI goes green. The PR now contains the whole story in one thread.
4. Capture: Langfuse trace view, failed check + comment, cost report, before/after scores.

---

## Phase 6 — Packaging (Days 18–21)

README: problem statement → architecture diagram (ingest → retrieve → generate → trace →
eval → CI gate) → real numbers pulled from your own runs (`$X / 1000 queries`, `p95 Yms`,
"caught 1 regression pre-merge") → 30–90s demo clip → one-page case study PDF.
Push to the GitHub repo from Step 0.2.

---

## Verification

| Phase | Command | Expect |
|---|---|---|
| 0 | `psql "$DATABASE_URL" -c "SELECT extversion FROM pg_extension WHERE extname='vector'"` | a version number |
| 1 | `python -m ingest.ingest` | row count > 0 in `rag_chunks` |
| 1 | `curl -XPOST :8000/ask -d '{"question":"..."}'` | grounded answer + contexts |
| 2 | ask a question, open cloud.langfuse.com | one trace, 2 nested obs, non-zero cost |
| 3 | `python -m eval.run_eval` | 4 scores + per-question JSON + cost total |
| 4 | open a no-op PR | check passes, comment posted |
| 5 | open the `top_k=1` PR | check **fails**, comment shows the drop |

## Risks

- **Supabase (dev) vs. container (CI) drift.** Two different databases. Keep *all* DDL in
  `app/db.py` and never hand-run SQL in the Supabase editor — that is what keeps them
  identical.
- **Supabase free tier pauses after ~1 week idle.** If ingestion or `/ask` suddenly fails
  with a connection error after a break, un-pause the project in the dashboard first
  before debugging code.
- **Langfuse Cloud free-tier limits.** Generous, but the eval runner traces every golden-set
  question on every run. If ingestion volume becomes a concern, disable tracing in the eval
  runner via an env flag and keep it on for the interactive demo path.
- **RAGAS 0.4 + Anthropic judge is a less-travelled path.** Metrics parse structured
  output from the judge; if a metric returns NaN, first try Sonnet over Haiku as judge.
  Smoke-test the judge wiring on 2 rows before running all 30.
- **Judge cost** is the dominant spend: ~4 metrics × 30 questions per run, and CI runs it
  per PR. Keep the golden set at 20–30, not 200.
