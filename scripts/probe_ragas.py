"""Smoke-test the RAGAS wiring on one hand-made row before running a full eval.

Confirms the Anthropic judge and local embeddings actually produce scores — the
least-travelled part of the stack. Cheap: one row, four metrics.

    .venv310/bin/python -m scripts.probe_ragas
"""

import asyncio

from anthropic import AsyncAnthropic

from app.config import settings

JUDGE_MODEL = "claude-sonnet-4-5"


async def main() -> int:
    from ragas.embeddings import HuggingFaceEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecisionWithReference,
        ContextRecall,
        Faithfulness,
    )

    llm = llm_factory(
        JUDGE_MODEL,
        provider="anthropic",
        client=AsyncAnthropic(api_key=settings.anthropic_api_key),
    )
    # RAGAS sends temperature/top_p by default, but the Anthropic SDK 1.x
    # removed both from messages.create() — passing them is a TypeError.
    llm.model_args.pop("temperature", None)
    llm.model_args.pop("top_p", None)
    emb = HuggingFaceEmbeddings(model=settings.embedding_model)
    print(f"judge      : {JUDGE_MODEL}\nembeddings : {settings.embedding_model}\n")

    row = dict(
        user_input="What is the context window of Claude Haiku 4.5?",
        response="Claude Haiku 4.5 has a 200K token context window.",
        retrieved_contexts=[
            "Claude Haiku 4.5 | claude-haiku-4-5 | 200K context | $1.00 input per MTok."
        ],
        reference="Claude Haiku 4.5 has a 200K token context window.",
    )

    # Each metric takes a different subset of the row — see ascore() signatures.
    jobs = [
        ("faithfulness", Faithfulness(llm=llm),
         ("user_input", "response", "retrieved_contexts")),
        ("answer_relevancy", AnswerRelevancy(llm=llm, embeddings=emb),
         ("user_input", "response")),
        ("context_precision", ContextPrecisionWithReference(llm=llm),
         ("user_input", "reference", "retrieved_contexts")),
        ("context_recall", ContextRecall(llm=llm),
         ("user_input", "retrieved_contexts", "reference")),
    ]

    for name, metric, fields in jobs:
        try:
            result = await metric.ascore(**{f: row[f] for f in fields})
            print(f"  {name:18} {result.value:.3f}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {name:18} FAILED: {type(exc).__name__}: {str(exc)[:150]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
