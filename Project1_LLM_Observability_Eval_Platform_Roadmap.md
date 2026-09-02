# Project 1 — LLM Observability & Evaluation Platform
## Step-by-Step Implementation Roadmap (Built From Scratch)

**Role Target:** LLMOps Engineer / AI Quality Engineer
**Difficulty:** Beginner–Intermediate
**Duration:** 2–3 weeks (part-time)

---

## 0. Goal of This Project

Build a small RAG (Retrieval-Augmented Generation) application from zero, then wrap it in tracing and automated evaluation so it proves — on every code change — whether it hallucinates, how much it costs, and where time is spent.

**End deliverable:** A public dashboard/demo showing (1) a regression caught before merge, (2) a cost breakdown, (3) a before/after eval score for a prompt change.

---

## Phase 1 — Build the RAG App (Days 1–4)

### Step 1.1 — Define the domain
Pick one narrow, real dataset to answer questions about. Keep scope small on purpose — this app is a vehicle for the observability work, not the star of the show.
- Example: a documentation Q&A bot over a set of PDFs/markdown docs (e.g., a product's help docs, or a public dataset like a company's policy documents).

### Step 1.2 — Set up the stack
- **Backend:** FastAPI (Python)
- **Vector store:** Supabase/Postgres + pgvector
- **LLM:** Claude API (Haiku for cost during dev, can swap later)
- **Embeddings:** any standard embedding model (OpenAI, Voyage, or Cohere)
- **Why FastAPI here:** the evaluation tooling in Phase 3 (RAGAS/DeepEval) is Python-only, so building the whole app in Python from the start avoids a two-language split and keeps every later project (2–5) in the same ecosystem.

### Step 1.3 — Build the ingestion pipeline
1. Chunk source documents (500–1000 tokens per chunk, with overlap).
2. Generate embeddings for each chunk.
3. Store chunks + embeddings in pgvector.

### Step 1.4 — Build the RAG query flow
1. Accept a user question via API endpoint.
2. Embed the question.
3. Retrieve top-k similar chunks from pgvector.
4. Construct a prompt with retrieved context + question.
5. Call Claude API to generate the answer.
6. Return the answer to the user.

**Checkpoint:** You should be able to POST a question to your API and get a grounded answer back.

---

## Phase 2 — Add Distributed Tracing (Days 5–7)

### Step 2.1 — Stand up Langfuse
- Self-host Langfuse using Docker Compose, pointed at your existing Supabase/Postgres instance (or a separate Postgres DB).
- Get your Langfuse public/secret API keys.

### Step 2.2 — Instrument the retrieval step
Wrap your pgvector similarity search call in a Langfuse span:
- Log: query text, number of chunks retrieved, retrieval latency, similarity scores.

### Step 2.3 — Instrument the generation step
Wrap your Claude API call in a Langfuse generation trace:
- Log: full prompt sent, model used, tokens in/out, latency, cost, raw response.

### Step 2.4 — Tie spans into one trace per request
Ensure retrieval span + generation span are nested under a single parent trace per user question, so a full request is viewable end-to-end in the Langfuse dashboard.

**Checkpoint:** Every question asked to your app now produces a visual trace in Langfuse showing retrieval → generation → cost → latency.

---

## Phase 3 — Build the Evaluation Harness (Days 8–11)

### Step 3.1 — Create a golden dataset
Write 20–30 question/answer pairs by hand, covering:
- Easy factual questions (answer clearly exists in your docs)
- Edge cases (ambiguous questions, questions with no good answer in the docs)
- At least a few "trap" questions designed to tempt hallucination

Store this as a JSON/CSV file in your repo (e.g., `eval/golden_set.json`).

### Step 3.2 — Install RAGAS or DeepEval
Choose one (RAGAS is more RAG-specific; DeepEval is broader). Install via pip.

### Step 3.3 — Define your eval metrics
At minimum, implement:
- **Faithfulness** — is the answer supported by the retrieved context, or invented?
- **Answer relevance** — does the answer actually address the question?
- **Context precision/recall** — did retrieval pull the right chunks?

### Step 3.4 — Write the eval runner script
A Python script (`eval/run_eval.py`) that:
1. Loops through the golden dataset.
2. Calls your FastAPI app's endpoint directly (same language, same repo — no cross-language HTTP glue needed).
3. Scores the response using RAGAS/DeepEval metrics.
4. Outputs an aggregate score + per-question breakdown (JSON/CSV).

**Checkpoint:** Running one command produces a score report for your whole app against the golden set.

---

## Phase 4 — Wire Evals into CI/CD (Days 12–14)

### Step 4.1 — Add a GitHub Actions workflow
- Trigger: on every pull request.
- Steps: spin up the app (or point at a staging deployment) → run the eval script → parse the output score.

### Step 4.2 — Set a quality gate
- Define a minimum acceptable score (e.g., faithfulness ≥ 0.85).
- If the eval score drops below threshold, fail the CI check and block merge.

### Step 4.3 — Add cost/token tracking to CI output
- Have the eval script also report total tokens used and estimated cost for the run, posted as a PR comment.

**Checkpoint:** Opening a PR that changes a prompt automatically triggers an eval run and shows pass/fail + cost delta directly in the PR.

---

## Phase 5 — Prove the Regression Catch (Days 15–17)

### Step 5.1 — Intentionally break something
Make a deliberate bad change — e.g., shrink the retrieved context window, or edit the system prompt to be vaguer — that should lower answer quality.

### Step 5.2 — Open a PR with the bad change
Let the CI eval gate run and fail, demonstrating the safety net working as intended.

### Step 5.3 — Fix it and show the recovery
Revert/fix the change, re-run, show the score back above threshold.

### Step 5.4 — Capture the artifact
Screenshot or screen-record:
- The Langfuse trace view for a sample request
- The failed CI check with the eval score drop
- The cost dashboard/report
- The before/after score comparison

---

## Phase 6 — Packaging & Showcase (Days 18–21)

### Step 6.1 — Write the README
Follow this structure (per the roadmap's Section 6 format):
1. Problem statement (1 paragraph)
2. Architecture diagram (retrieval → generation → tracing → eval → CI gate)
3. Before/after metrics (e.g., "caught 1 regression pre-merge," "$X per 1000 queries," "p95 latency Yms")
4. 30–90 second demo recording
5. One-page case-study PDF version

### Step 6.2 — Publish
- Push to a dedicated GitHub repo (this becomes repo 1 of your 5-repo MLOps org).
- Record a short demo clip for LinkedIn.
- Add "LLM Evaluation & Observability" as a service line on your Upwork profile if relevant.

---

## Tool Checklist

| Tool | Purpose |
|---|---|
| FastAPI (Python) | RAG app backend |
| Supabase + pgvector | Vector store |
| Claude API | Generation |
| Langfuse (self-hosted) | Tracing/observability |
| RAGAS or DeepEval | Automated evaluation |
| GitHub Actions | CI/CD eval gating |

## Success Criteria (Definition of Done)
- [ ] RAG app answers questions end-to-end
- [ ] Every request produces a full trace in Langfuse
- [ ] Golden dataset of 20–30 Q&A pairs exists
- [ ] Eval script scores faithfulness, relevance, context precision
- [ ] CI blocks a merge on eval regression
- [ ] Recorded demo shows a caught regression + recovery
- [ ] README + case study published to GitHub
