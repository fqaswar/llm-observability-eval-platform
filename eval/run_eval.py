"""Score the RAG app against the golden set.

Calls app.rag.answer_question in-process — same repo, same language, no HTTP
glue — then scores each response with RAGAS using Claude as the judge and the
app's own embedding model.

    .venv310/bin/python -m eval.run_eval
    .venv310/bin/python -m eval.run_eval --limit 4          # cheap smoke run
    .venv310/bin/python -m eval.run_eval --tags trap        # only trap questions
    .venv310/bin/python -m eval.run_eval --min-faithfulness 0.85

Exits non-zero if any gated metric falls below its threshold, which is what
lets CI block a merge in Phase 4.
"""

import argparse
import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

# Tracing off by default: a 26-question run would otherwise emit 26 traces on
# every CI run. Opt in with --trace.
os.environ.setdefault("LANGFUSE_ENABLED", "false")

from app.config import settings  # noqa: E402
from app.rag import answer_question  # noqa: E402

GOLDEN = Path("eval/golden_set.json")
RESULTS_DIR = Path("eval/results")

JUDGE_MODEL = "claude-sonnet-4-5"

# USD per million tokens, for the app model (not the judge).
PRICE_PER_MTOK = {"claude-haiku-4-5": (1.00, 5.00)}

METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def build_judge():
    """A RAGAS LLM backed by Claude, plus the app's local embedding model."""
    from anthropic import AsyncAnthropic
    from ragas.embeddings import HuggingFaceEmbeddings
    from ragas.llms import llm_factory

    llm = llm_factory(
        JUDGE_MODEL,
        provider="anthropic",
        client=AsyncAnthropic(api_key=settings.anthropic_api_key),
    )
    # RAGAS sends temperature/top_p by default; the Anthropic SDK 1.x removed
    # both from messages.create(), so passing them raises a TypeError.
    llm.model_args.pop("temperature", None)
    llm.model_args.pop("top_p", None)

    return llm, HuggingFaceEmbeddings(model=settings.embedding_model)


async def score_row(row: dict, metrics: dict) -> dict:
    """Score one answered question. Each metric takes a different subset."""
    args = {
        "faithfulness": ("user_input", "response", "retrieved_contexts"),
        "answer_relevancy": ("user_input", "response"),
        "context_precision": ("user_input", "reference", "retrieved_contexts"),
        "context_recall": ("user_input", "retrieved_contexts", "reference"),
    }
    out = {}
    for name, metric in metrics.items():
        try:
            res = await metric.ascore(**{f: row[f] for f in args[name]})
            value = res.value
            # A metric can legitimately return None (e.g. no claims to verify).
            out[name] = float(value) if value is not None else None
        except Exception as exc:  # noqa: BLE001 — one bad metric shouldn't abort
            out[name] = None
            out.setdefault("errors", []).append(f"{name}: {type(exc).__name__}: {exc}")
    return out


def estimate_cost(in_tok: int, out_tok: int, model: str) -> float:
    rate_in, rate_out = PRICE_PER_MTOK.get(model, (0.0, 0.0))
    return (in_tok / 1e6) * rate_in + (out_tok / 1e6) * rate_out


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="only the first N questions")
    ap.add_argument("--tags", nargs="*", help="only questions carrying these tags")
    ap.add_argument("--top-k", type=int, help="override retrieval k")
    ap.add_argument("--trace", action="store_true", help="send traces to Langfuse")
    ap.add_argument("--out", default="eval/results/latest.json")
    for m in METRIC_NAMES:
        ap.add_argument(f"--min-{m.replace('_', '-')}", type=float, default=None)
    args = ap.parse_args()

    if args.trace:
        os.environ["LANGFUSE_ENABLED"] = "true"

    data = json.loads(GOLDEN.read_text())
    questions = data["questions"]
    if args.tags:
        questions = [q for q in questions if set(q["tags"]) & set(args.tags)]
    if args.limit:
        questions = questions[: args.limit]
    if not questions:
        print("No questions selected.")
        return 1

    top_k = args.top_k if args.top_k is not None else settings.rag_top_k
    print(
        f"Evaluating {len(questions)} questions\n"
        f"  app model : {settings.rag_model} (top_k={top_k})\n"
        f"  judge     : {JUDGE_MODEL}\n"
    )

    # --- 1. Answer every question ---
    started = time.time()
    rows = []
    for i, q in enumerate(questions, 1):
        r = answer_question(q["question"], k=top_k)
        rows.append(
            {
                "id": q["id"],
                "tags": q["tags"],
                "user_input": q["question"],
                "response": r.answer,
                "retrieved_contexts": r.contexts,
                "reference": q["reference"],
                "sources": [c.source for c in r.chunks],
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "latency_ms": round(r.total_ms, 1),
            }
        )
        print(f"  [{i}/{len(questions)}] {q['id']} answered ({r.total_ms:.0f}ms)")
    answer_secs = time.time() - started

    # --- 2. Score them ---
    print("\nScoring…")
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecisionWithReference,
        ContextRecall,
        Faithfulness,
    )

    llm, emb = build_judge()
    metrics = {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=emb),
        "context_precision": ContextPrecisionWithReference(llm=llm),
        "context_recall": ContextRecall(llm=llm),
    }

    scored = 0
    for row in rows:
        row.update(await score_row(row, metrics))
        scored += 1
        abbrev = {
            "faithfulness": "faith",
            "answer_relevancy": "relev",
            "context_precision": "cprec",
            "context_recall": "crecall",
        }
        vals = " ".join(
            f"{abbrev[n]}={row[n]:.2f}" if row.get(n) is not None else f"{abbrev[n]}=--"
            for n in METRIC_NAMES
        )
        print(f"  [{scored}/{len(rows)}] {row['id']:5} {vals}")

    # --- 3. Aggregate ---
    # RAGAS scores a deliberately non-committal answer ("the docs don't cover
    # this") as 0 relevancy. On trap questions that refusal is the CORRECT
    # behaviour, so averaging relevancy over traps would penalise the app for
    # doing the right thing. Relevancy is therefore aggregated over non-trap
    # questions only; the per-question values remain in the report.
    answerable = [r for r in rows if "trap" not in r["tags"]]

    aggregates = {}
    for name in METRIC_NAMES:
        pool = answerable if name == "answer_relevancy" else rows
        vals = [r[name] for r in pool if r.get(name) is not None]
        aggregates[name] = round(mean(vals), 4) if vals else None

    # Traps get their own headline number: did the app decline when it should?
    traps = [r for r in rows if "trap" in r["tags"]]
    trap_faith = [r["faithfulness"] for r in traps if r.get("faithfulness") is not None]
    trap_recall = [r["context_recall"] for r in traps if r.get("context_recall") is not None]
    aggregates["trap_faithfulness"] = round(mean(trap_faith), 4) if trap_faith else None
    aggregates["trap_context_recall"] = round(mean(trap_recall), 4) if trap_recall else None

    tok_in = sum(r["input_tokens"] for r in rows)
    tok_out = sum(r["output_tokens"] for r in rows)
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "model": settings.rag_model,
            "judge": JUDGE_MODEL,
            "top_k": top_k,
            "embedding_model": settings.embedding_model,
            "questions": len(rows),
        },
        "scores": aggregates,
        "usage": {
            "input_tokens": tok_in,
            "output_tokens": tok_out,
            "app_cost_usd": round(estimate_cost(tok_in, tok_out, settings.rag_model), 6),
        },
        "timing": {
            "answer_secs": round(answer_secs, 1),
            "total_secs": round(time.time() - started, 1),
        },
        "questions": rows,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    # A compact companion file. The full report embeds every retrieved chunk and
    # runs to hundreds of KB, which makes a poor thing for CI to diff on every
    # PR; this one holds just the numbers.
    summary = {k: v for k, v in report.items() if k != "questions"}
    summary["per_question"] = [
        {
            "id": r["id"],
            "tags": r["tags"],
            **{m: r.get(m) for m in METRIC_NAMES},
        }
        for r in rows
    ]
    summary_path = out_path.with_name(out_path.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))

    # --- 4. Report ---
    print(f"\n{'':-<58}")
    print(f"{'metric':22} {'score':>8}")
    print(f"{'':-<58}")
    for name, val in aggregates.items():
        note = ""
        if name == "answer_relevancy" and len(answerable) != len(rows):
            note = f"  (excludes {len(rows) - len(answerable)} traps)"
        print(f"{name:22} {val if val is not None else 'n/a':>8}{note}")
    print(f"{'':-<58}")

    by_tag: dict[str, list[float]] = {}
    for r in rows:
        f = r.get("faithfulness")
        if f is not None:
            by_tag.setdefault(r["tags"][0], []).append(f)
    if by_tag:
        print("faithfulness by tag:")
        for tag, vals in sorted(by_tag.items()):
            print(f"  {tag:10} {mean(vals):.3f}  (n={len(vals)})")

    print(
        f"\ntokens: {tok_in} in / {tok_out} out"
        f"   app cost: ${report['usage']['app_cost_usd']:.4f}"
        f"   wall: {report['timing']['total_secs']}s"
    )
    print(f"report: {out_path}\nsummary: {summary_path}")

    # --- 5. Gate ---
    failures = []
    for name in METRIC_NAMES:
        threshold = getattr(args, f"min_{name}")
        if threshold is None:
            continue
        actual = aggregates[name]
        if actual is None:
            failures.append(f"{name}: no score produced (threshold {threshold})")
        elif actual < threshold:
            failures.append(f"{name}: {actual:.4f} < {threshold}")

    if failures:
        print("\nQUALITY GATE FAILED")
        for f in failures:
            print(f"  {f}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
